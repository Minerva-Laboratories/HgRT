"""HgRT Mercurius benchmarks: decode speed (plain and MTP k = 1..3) at several context depths, losslessness of the
speculative path against the engine's own greedy decode, agreement with the deployed transformers reference, per-kernel
time breakdown, peak memory. Writes results/mercurius/bench.json.

    python bench.py [--ctx chat0,chat1,...,long1024,long8192,long32768] [--n 256] [--ks 1,2,3]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'src'))
from hgrt.mercurius_engine import DEV, MercuriusEngine
from hgrt.mercurius_mtp import MTPHead, SpecDecoder

OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")
RES = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'results', 'mercurius')


def sync_time():
    torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def plain(E, snap, n):
    E.load(snap)
    E.step()                                   # capture (first call)
    for _ in range(48):                        # warm-up: DVFS ramps the GPU / memory clocks under load
        E.step()
    E.load(snap)
    t = sync_time()
    toks = E.generate(n)
    dt = sync_time() - t
    return toks, dt


@torch.no_grad()
def teacher_forced(E, snap, ref_tokens):
    """feed the reference tokens; fraction of positions where the engine's argmax equals the reference's next token"""
    E.load(snap)
    first = int(E.tok_in[0])
    agree = [first == int(ref_tokens[0])]
    for t in range(len(ref_tokens) - 1):
        E.tok_in[0] = ref_tokens[t]
        E.step()
        agree.append(int(E.tok_in[0]) == int(ref_tokens[t + 1]))
    return sum(agree) / len(agree)


@torch.no_grad()
def profile_breakdown(E, snap, reps=3):
    """kernel time by name over `reps` eager plain steps (CUDA activity), plus the graph step time."""
    from torch.profiler import ProfilerActivity, profile
    E.load(snap)
    E.step(graph=False)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            E.step(graph=False)
        torch.cuda.synchronize()
    rows = {}
    for ev in prof.key_averages():
        if ev.device_type is not None and "cuda" in str(ev.device_type).lower():
            t = getattr(ev, "device_time_total", None) or getattr(ev, "cuda_time_total", 0)
            rows[ev.key] = rows.get(ev.key, 0) + t / reps / 1e3
    groups = {}
    for k, v in rows.items():
        name = k.split("<")[0].split("(")[0].strip()
        for tag, g in (("nf4_gemv_k", "NF4 GEMV"), ("mla_attn_k", "MLA attention (codes)"), ("mla_reduce", "MLA attention (codes)"),
                       ("gdn2_rec", "GDN-2 recurrence+norm"), ("gdn2_gates", "GDN-2 gates (VeRA B)"),
                       ("dgemv", "small bf16 GEMV (gates A/base, rotations, k_rope)"), ("gdn2_conv", "GDN-2 conv"),
                       ("mla_query", "MLA query map"), ("mla_vup", "MLA V up-proj+gate"), ("mla_encode", "MLA encode"),
                       ("rms_rows", "norms+quant"), ("quant_rows", "norms+quant"), ("silu_mul", "norms+quant"),
                       ("argmax", "argmax"), ("emb_nf4", "embedding")):
            if tag in name:
                groups[g] = groups.get(g, 0) + v
                break
        else:
            groups["other"] = groups.get("other", 0) + v
    return {k: round(v, 3) for k, v in sorted(groups.items(), key=lambda x: -x[1])}, \
        {k: round(v, 3) for k, v in sorted(rows.items(), key=lambda x: -x[1])[:25]}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="chat0,chat1,chat2,chat3,chat4,chat5,long1024,long8192,long32768")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--vocab", type=int, default=0, help="draft head vocabulary subset size (0 = full)")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--gemv", default="dp4a", choices=["dp4a", "mma"])
    ap.add_argument("--attn", default="v2", choices=["fd", "v2"])
    ap.add_argument("--mtp-kv", default="bf16", choices=["bf16", "int8", "int4"])
    ap.add_argument("--no-fused", action="store_true", help="separate GDN-2 commit kernels (the first version)")
    ap.add_argument("--adaptive", action="store_true", help="also run adaptive k")
    ap.add_argument("--out", default=os.path.join(RES, "bench.json"))
    a = ap.parse_args()
    ctxs = a.ctx.split(",")
    ks = [int(k) for k in a.ks.split(",") if k]
    t0 = time.time()
    E = MercuriusEngine(os.path.join(OUT, "weights.pt"), Tmax=33800 + a.n + 64, gemv=a.gemv, attn=a.attn)
    dv = None
    if a.vocab:
        dv = torch.load(os.path.join(OUT, "draft_vocab.pt"))[:a.vocab]
    H = MTPHead(E, draft_vocab=dv, kv=a.mtp_kv) if ks else None
    print(f"engine + MTP built in {time.time() - t0:.0f}s, mem {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
    res = {"n": a.n, "gemv": a.gemv, "attn": a.attn, "mtp_kv": a.mtp_kv, "fused_commit": not a.no_fused, "draft_vocab": a.vocab or "full", "ctx": {}}
    for c in ctxs:
        snap = torch.load(os.path.join(OUT, f"snap_{c}.pt"), weights_only=False)
        ref = torch.load(os.path.join(OUT, f"ref_{c}.pt"), weights_only=False)
        T = int(snap["ids"].shape[0])
        torch.cuda.reset_peak_memory_stats()
        toks, dt = plain(E, snap, a.n)
        r = {"T": T, "plain": {"tok_s": a.n / dt, "ms_per_token": dt / a.n * 1e3}}
        rt = ref["tokens"].to(DEV)
        m = min(len(rt), a.n)
        eq = (toks[:m] == rt[:m])
        r["vs_reference"] = {"first_divergence": int(m if bool(eq.all()) else int((~eq).nonzero()[0])), "compared": m,
                             "free_running_match": float(eq.float().mean()),
                             "teacher_forced_argmax_agree": teacher_forced(E, snap, rt[:m]),
                             "reference_ms_per_token": ref["ms_per_token"]}
        print(f"{c} T={T}: plain {r['plain']['tok_s']:.1f} tok/s ({r['plain']['ms_per_token']:.2f} ms) | vs reference: "
              f"first divergence {r['vs_reference']['first_divergence']}/{m}, teacher-forced agree "
              f"{r['vs_reference']['teacher_forced_argmax_agree']:.3f}", flush=True)
        if a.profile:
            g, top = profile_breakdown(E, snap)
            r["profile_ms"] = g
            r["profile_top_kernels_ms"] = top
            print("  profile (ms/token, eager kernel time):", g, flush=True)
        runs = [(f"mtp_k{k}", dict(k=k)) for k in ks] + ([("mtp_adaptive", dict(adaptive=True))] if a.adaptive else [])
        for name, kw in runs:
            S = SpecDecoder(E, H, fused=not a.no_fused, **kw)
            E.load(snap)
            S.start(snap)
            S.warm()                                           # capture
            S.run(16, kw.get("k", 2))                          # warm-up (DVFS)
            E.load(snap)
            S.start(snap)
            t = sync_time()
            st, cyc, hist = S.generate(a.n)
            dts = sync_time() - t
            emitted = int(S.count)
            lossless = bool(torch.equal(st[:a.n], toks[:a.n]))
            tau = (emitted - 1) / cyc
            kk = kw.get("k") or (sum(S.k_log) / max(1, len(S.k_log)))
            r[name] = {"tok_s": emitted / dts, "ms_per_token": dts / emitted * 1e3, "lossless": lossless,
                       "tau": tau, "accept_hist": hist.tolist(), "cycles": cyc, "emitted": emitted,
                       "accept_rate_per_draft": (tau - 1) / kk}
            if S.k_log:
                r[name]["k_choices"] = {str(k): S.k_log.count(k) for k in set(S.k_log)}
            print(f"  {name}: {emitted / dts:.1f} tok/s, tau {tau:.2f}, accepted-draft histogram {hist.tolist()}"
                  + (f", k groups {r[name]['k_choices']}" if S.k_log else "") + f", lossless {lossless}" + ("" if lossless else
                  f" (first diff at {int((st[:a.n] != toks[:a.n]).nonzero()[0])})"), flush=True)
            del S
        r["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        res["ctx"][c] = r
        del snap
        torch.cuda.empty_cache()
    os.makedirs(RES, exist_ok=True)
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    old.update({k: v for k, v in res.items() if k != "ctx"})
    old.setdefault("ctx", {}).update(res["ctx"])
    json.dump(old, open(a.out, "w"), indent=1)
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
