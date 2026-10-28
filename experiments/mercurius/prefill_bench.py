"""Native prefill: speed at ~1K / 8K / 32K and agreement with the deployed transformers prefill (snapshots): first
token, last-position logits, GDN-2 states, MLA codes; then decode after the native prefill vs after the hand-off, and
MTP losslessness from the native state. Writes results/mercurius/prefill.json."""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'src'))
from cairn.mercurius_engine import MercuriusEngine
from cairn.mercurius_mtp import MTPHead, SpecDecoder
from cairn.mercurius_prefill import Prefill

OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")
RES = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'results', 'mercurius')


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@torch.no_grad()
def main():
    ctxs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["chat0", "long1024", "long8192", "long32768"]
    E = MercuriusEngine(os.path.join(OUT, "weights.pt"), Tmax=33800 + 400)
    P = Prefill(E, chunk=int(os.environ.get("PREFILL_CHUNK", 2048)))
    if os.environ.get("PREFILL_ONLY"):
        for c in ctxs:
            snap = torch.load(os.path.join(OUT, f"snap_{c}.pt"), weights_only=False)
            ids = snap["ids"].cuda()
            P(ids); P.profile = True; torch.cuda.synchronize(); t0 = time.time(); P(ids); torch.cuda.synchronize()
            print(c, f"chunk {P.chunk}: {time.time() - t0:.2f} s (profiled run)", {k: round(v, 3) for k, v in P.last_profile.items()}, flush=True)
            P.profile = False
        return
    H = MTPHead(E, draft_vocab=torch.load(os.path.join(OUT, "draft_vocab.pt"))[:32768])
    ref_t = json.load(open(os.path.join(OUT, "ref_timing.json")))
    res = {}
    for c in ctxs:
        snap = torch.load(os.path.join(OUT, f"snap_{c}.pt"), weights_only=False)
        ref = torch.load(os.path.join(OUT, f"ref_{c}.pt"), weights_only=False)
        ids = snap["ids"].cuda()
        T = ids.shape[0]
        P(ids)                                                  # warm-up at the same length (Triton autotuning keys)
        P.profile = True
        P(ids)
        prof = P.last_profile
        P.profile = False
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        nat = P(ids)
        torch.cuda.synchronize()
        dt = time.time() - t0
        r = {"T": T, "prefill_s": dt, "prefill_tok_s": T / dt, "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
             "transformers_prefill_s": ref_t[c]["prefill_s"]}
        lg = E.logits[0, :E.V].clone()
        rl = snap["last_logits"].cuda().float()[:E.V]
        r["first_token_equal"] = bool(int(lg.argmax()) == int(rl.argmax()))
        top = rl.topk(2).values
        r["reference_first_token_margin"] = float(top[0] - top[1])
        r["profile_s"] = {k: round(v, 3) for k, v in prof.items()}
        r["last_logits_rel_vs_reference"] = rel(lg, rl)
        st_nat = {i: (E.conv[i].clone(), E.rec[i].clone()) for i in E.lin_ids}
        codes_nat = {i: E.layers[i]["cache"][:T].clone() for i in E.attn_ids}
        r["rec_state_rel_vs_reference_mean"] = sum(rel(st_nat[i][1], snap["lin"][i][1].cuda()) for i in E.lin_ids) / len(E.lin_ids)
        r["conv_state_rel_vs_reference_mean"] = sum(rel(st_nat[i][0][:, 1:], snap["lin"][i][0].cuda().float()[:, -3:]) for i in E.lin_ids) / len(E.lin_ids)
        same = tot = 0
        for i in E.attn_ids:
            a = snap["attn"][i]
            rr = E.layers[i]["RW"]
            lc = codes_nat[i][:, 24 + 32:24 + 32 + a["lat_codes"].shape[1]]
            same += int((lc.cpu() == a["lat_codes"]).sum()); tot += lc.numel()
        r["latent_code_bytes_equal_frac"] = same / tot
        # decode after the native prefill vs reference tokens and vs decode after the snapshot hand-off
        n = 64
        E.step()
        E.load(snap); toks_h = E.generate(n)
        P(ids); toks_n = E.generate(n)
        rt = ref["tokens"][:n].cuda()
        first_div = lambda x, y: int(n if bool((x == y).all()) else int((x != y).nonzero()[0]))
        r["decode_after_native_vs_handoff_first_div"] = first_div(toks_n, toks_h)
        r["decode_after_native_vs_reference_first_div"] = first_div(toks_n, rt)
        r["decode_after_handoff_vs_reference_first_div"] = first_div(toks_h, rt)
        # MTP from the native state: lossless vs plain from the native state
        nat = P(ids); toks_p = E.generate(128)
        S = SpecDecoder(E, H, k=2)
        nat = P(ids); S.start(nat); ts, cyc, hist = S.generate(128)
        r["mtp_k2_lossless_from_native"] = bool(torch.equal(ts[:128], toks_p[:128]))
        r["mtp_k2_tau_from_native"] = (int(S.count) - 1) / cyc
        res[c] = r
        print(c, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        del snap
        torch.cuda.empty_cache()
    os.makedirs(RES, exist_ok=True)
    old = json.load(open(os.path.join(RES, "prefill.json"))) if os.path.exists(os.path.join(RES, "prefill.json")) else {}
    old.update(res)
    json.dump(old, open(os.path.join(RES, "prefill.json"), "w"), indent=1)
    print("PREFILL_DONE")


if __name__ == "__main__":
    main()
