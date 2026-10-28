"""Diagnostics: (1) teacher-forced disagreements with the transformers reference vs the reference's top-1/top-2 margin;
(2) per-component decode time (each piece captured alone in a CUDA graph, replayed) -> results/mercurius/profile.json."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'src'))
from cairn.mercurius_engine import DEV, HID, MercuriusEngine
from cairn.mercurius_kernels import ext

OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")
RES = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'results', 'mercurius')


@torch.no_grad()
def margins(E, c):
    snap = torch.load(os.path.join(OUT, f"snap_{c}.pt"), weights_only=False)
    ref = torch.load(os.path.join(OUT, f"ref_{c}.pt"), weights_only=False)
    rt, rm = ref["tokens"].to(DEV), ref["margins"]
    E.load(snap)
    rows = []
    for t in range(len(rt)):
        lg = E.logits[0, :E.V]
        top = lg.topk(2)
        eng_tok = int(top.indices[0])
        if eng_tok != int(rt[t]):
            rows.append({"t": t, "ref_margin": float(rm[t]), "eng_margin": float(top.values[0] - top.values[1]),
                         "eng_logit_gap_ref_tok": float(top.values[0] - lg[rt[t]])})
        E.tok_in[0] = rt[t]
        E.step()
    allm = rm.float()
    return {"n": len(rt), "disagree": rows, "ref_margin_median": float(allm.median()),
            "ref_margin_frac_below_0.5": float((allm < 0.5).float().mean())}


def gtime(fn, it=20, warm_s=0.5):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it):
            fn()
    t0 = time.time()
    while time.time() - t0 < warm_s:                   # DVFS: let the GPU / memory clocks ramp up
        g.replay()
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return (time.time() - t) / (3 * it) * 1e3


@torch.no_grad()
def profile(E, c, M=1):
    """ms per token by component (state is restored after timing; times are for the snapshot's context length)."""
    snap = torch.load(os.path.join(OUT, f"snap_{c}.pt"), weights_only=False)
    E.load(snap)
    X = ext()
    st = E.save_state()
    caches = {i: E.layers[i]["cache"].clone() for i in E.attn_ids}
    i_g, i_a = E.lin_ids[0], E.attn_ids[-1]
    dg, da = E.layers[i_g], E.layers[i_a]
    xq, xd = E.xq(HID)
    out = {}
    # whole GDN-2 / MLA / MLP blocks (mode: plain for GDN writes states -> restored after)
    out["gdn2 layer (x24)"] = gtime(lambda: E._gdn(i_g, dg, M, 0 if M == 1 else 1)) * len(E.lin_ids)
    out["mla layer (x8, all layers)"] = sum(gtime(lambda i=i: E._mla(i, E.layers[i], M)) for i in E.attn_ids)
    out["mlp (x32)"] = gtime(lambda: E._mlp(dg, M)) * 32
    out["head + argmax"] = gtime(lambda: (E.head(xq, xd, M, E.logits, E.Vrows), X.argmax_rows(E.logits, E.Vrows, E.V, E.tok_out, M)))
    # sub-pieces of a GDN-2 layer
    sub = {}
    sub["rms+quant"] = gtime(lambda: X.rms_rows(E.h, HID, dg["ln1"], E.x, HID, xq, xd, M, HID, 1e-6))
    sub["qkv|z NF4 GEMV"] = gtime(lambda: dg["qkvz"](xq, xd, M, E.qkvz, E.qkvz.shape[1]))
    sub["gates: base|A|loraA (tensor cores)"] = gtime(lambda: X.tc_skinny(dg["Ws"], HID, dg["Ws"].shape[0], HID, E.x, HID, E.sg, E.NSG, M, True))
    sub["gates: VeRA B (tensor cores) + decay/sigmoid"] = gtime(lambda: X.gdn2_gates_tc(E.sg, E.NSG, 32, E.rank, dg["lr"], dg["B"], dg["Bl"], dg["vd"], dg["vb"],
                                                                      dg["A_log"], dg["dt_bias"], E.W2, E.T, E.gd[i_g], E.be[i_g], E.bw[i_g], M, 128))
    sub["conv"] = gtime(lambda: X.gdn2_conv(E.qkvz, E.qkvz.shape[1], E.conv[i_g], dg["conv_w"], E.mixed[i_g], E.xsave[i_g], M, False))
    xq4, xd4 = E.xq(4096)
    sub["recurrence+gated norm"] = gtime(lambda: X.gdn2_rec(E.mixed[i_g], E.qkvz[:, 8192:], E.qkvz.shape[1], E.gd[i_g], E.be[i_g], E.bw[i_g],
                                                            dg["gn"], E.rec[i_g], E.gdn_out, xq4, xd4, 16, 32, 1e-6, M, 1, None, None))
    sub["out NF4 GEMV"] = gtime(lambda: dg["out"](xq4, xd4, M, E.h, HID, E.h, HID))
    out["gdn2 layer breakdown (one layer)"] = sub
    msub = {}
    d = da
    NQD = E.NQD[i_a]
    msub["qdk NF4 GEMV"] = gtime(lambda: d["qdk"](xq, xd, M, E.yq, NQD))
    msub["k_rope / R0 / R small GEMVs"] = gtime(lambda: (X.dgemv(d["Wr"], HID, 64, HID, E.x, HID, E.kt0, 64, M, True),
                                                         X.dgemv(d["R0"], 64, 64, 64, E.kt0, 64, E.kr, 64, M, True),
                                                         X.dgemv(d["R"], d["r"], d["r"], d["r"], E.yq[:, 8192:], NQD, E.crot, d["r"], M, True)))
    Ep = d["Ep"]
    Q = E.Q.view(-1)[:M * 16 * Ep]
    msub["query map"] = gtime(lambda: X.mla_query(E.yq, NQD, d["qn"], d["q_eps"], d["P"], d["E"], Ep, E.pos, E.inv_freq, Q, M))
    po = E.part_o.view(-1)[:E.NS * M * 16 * (Ep - 64)]
    pm = E.part_ml.view(-1)[:E.NS * M * 16 * 2].view(E.NS, -1)
    olat = E.olat.view(-1)[:M * 16 * d["r"]]
    msub["attention from codes (+reduce)"] = gtime(lambda: X.mla_attn(Q, Ep, d["cache"], d["RW"], d["R0h"], d["cbl"], d["cbr"], E.pos,
                                                                      E.inv_freq, po, pm, olat, d["r"], d["scale"], M))
    msub["V up + gate"] = gtime(lambda: X.mla_vup(olat, d["r"], d["Wv"], E.yq, NQD, E.yattn, M))
    xq2, xd2 = E.xq(4096)
    msub["o NF4 GEMV"] = gtime(lambda: d["o"](xq2, xd2, M, E.h, HID, E.h, HID))
    out[f"mla layer breakdown (layer {i_a}, r={d['r']})"] = msub
    E.load_state(st)
    for i in E.attn_ids:
        E.layers[i]["cache"].copy_(caches[i])
    # the real graph step
    E.load(snap)
    E.step()
    E.load(snap)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(64):
        E.step()
    torch.cuda.synchronize()
    out["graph step (measured)"] = (time.time() - t) / 64 * 1e3
    return out


def main():
    E = MercuriusEngine(os.path.join(OUT, "weights.pt"), Tmax=33900, gemv=os.environ.get("CAIRN_GEMV", "dp4a"))
    res = {}
    for c in sys.argv[1].split(",") if len(sys.argv) > 1 else ["chat0", "long8192", "long32768"]:
        m = margins(E, c)
        print(c, "disagreements:", [(r["t"], round(r["ref_margin"], 3), round(r["eng_margin"], 3)) for r in m["disagree"]],
              f"| ref median margin {m['ref_margin_median']:.2f}, frac < 0.5: {m['ref_margin_frac_below_0.5']:.3f}", flush=True)
        p = profile(E, c)
        print(c, json.dumps(p, indent=1), flush=True)
        res[c] = {"margins": m, "profile_ms": p}
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, "diag.json"), "w"), indent=1)
    print("DIAG_DONE")


if __name__ == "__main__" and not os.environ.get("DIAG_CYCLE"):
    main()


@torch.no_grad()
def profile_cycle(E, c, ks=(1, 2, 3), vocab=32768):
    """ms of the pieces of one speculation cycle (each captured alone)."""
    from cairn.mercurius_mtp import MTPHead, SpecDecoder
    snap = torch.load(os.path.join(OUT, f"snap_{c}.pt"), weights_only=False)
    H = MTPHead(E, draft_vocab=torch.load(os.path.join(OUT, "draft_vocab.pt"))[:vocab])
    out = {}
    for k in ks:
        M = k + 1
        E.load(snap)
        S = SpecDecoder(E, H, k)
        S.start(snap)
        st = E.save_state()
        r = {}
        r["verify block (trunk, M rows, mode 1)"] = gtime(lambda: E._block(M, 1), 5)
        r["  of which: one GDN-2 layer"] = gtime(lambda: E._gdn(E.lin_ids[0], E.layers[E.lin_ids[0]], M, 1), 10)
        r["  of which: one MLP"] = gtime(lambda: E._mlp(E.layers[0], M), 10)
        r["  of which: one MLA layer"] = gtime(lambda: E._mla(E.attn_ids[-1], E.layers[E.attn_ids[-1]], M), 10)
        xq, xd = E.xq(HID)
        r["  of which: head"] = gtime(lambda: E.head(xq, xd, M, E.logits, E.Vrows), 10)
        r["  of which: gates (tensor cores)"] = gtime(lambda: (ext().tc_skinny(E.layers[0]["Ws"], HID, E.layers[0]["Ws"].shape[0], HID, E.x, HID, E.sg, E.NSG, M, True),
            ext().gdn2_gates_tc(E.sg, E.NSG, 32, E.rank, E.layers[0]["lr"], E.layers[0]["B"], E.layers[0]["Bl"], E.layers[0]["vd"], E.layers[0]["vb"],
                             E.layers[0]["A_log"], E.layers[0]["dt_bias"], E.W2, E.T, E.gd[0], E.be[0], E.bw[0], M, 128)), 10)
        E.nacc.fill_(k)
        r["commit (24 GDN layers)"] = gtime(lambda: [E._gdn_commit(i) for i in E.lin_ids], 5)
        r["MTP step 1 (M rows) + draft"] = gtime(lambda: (H.rows(S.hbuf, S.abuf, S.P0, M, row=S.nrow), H.draft()), 5)
        r["MTP step j (1 row) + draft"] = gtime(lambda: (H.rows(S.hrec, S.drafts[0:1], S.P0, 1), H.draft()), 5)
        r["  of which: draft head + argmax"] = gtime(lambda: H.draft(), 10)
        E.load_state(st)
        E.load(snap)
        S.start(snap)
        S.warm()                                         # every graph (both commit parities) captured before timing
        S.run(8)
        E.load(snap)
        S.start(snap)
        torch.cuda.synchronize()
        t = time.time()
        S.run(16)
        torch.cuda.synchronize()
        r["full cycle (graph, measured)"] = (time.time() - t) / 16 * 1e3
        out[f"k={k}"] = r
        print(c, f"k={k}", json.dumps({kk: round(v, 3) for kk, v in r.items()}, indent=1), flush=True)
        del S
    return out


if __name__ == "__main__" and os.environ.get("DIAG_CYCLE"):
    E = MercuriusEngine(os.path.join(OUT, "weights.pt"), Tmax=33900, gemv=os.environ.get("CAIRN_GEMV", "dp4a"))
    res = {c: profile_cycle(E, c) for c in os.environ["DIAG_CYCLE"].split(",")}
    json.dump(res, open(os.path.join(RES, "diag_cycle.json"), "w"), indent=1)
    print("CYCLE_DONE")
