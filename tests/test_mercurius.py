"""Mercurius HgRT engine: kernel correctness against PyTorch references, engine vs a PyTorch re-implementation of the same
decode step, graph == eager, and the M-row verification pass == plain decoding bit for bit.

    PYTHONPATH=src python tests/test_mercurius.py [weights.pt] [snapshot.pt]
"""
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'), 'src'))
from cairn.mercurius_engine import (CONV, DEV, DKH, FF, HD, HID, NG, NH, NK, NV, RD, VALD, MercuriusEngine, pad16)
from cairn.mercurius_kernels import ext

OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")
torch.manual_seed(0)
torch.set_grad_enabled(False)
FAILS = []


def check(name, ok, info=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAILS.append(name)


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def bfr(x):
    return x.to(torch.bfloat16).float()


# ------------------------------------------------------------------------------------------------ references
def quant_ref(x):
    """int8 groups of 32 like warp_quant32: (q (M, K) float, d (M, K/32))"""
    M, K = x.shape
    g = x.float().view(M, K // 32, 32)
    d = g.abs().amax(-1) / 127
    q = torch.where(d[..., None] > 0, torch.round(g / d[..., None]), torch.zeros_like(g))
    return q.view(M, K), d


def nf4_ref(lin, levels, x, res=None, exact_act=True):
    """x (M, K) bf16 -> fp64 reference of the kernel (int8 activations, int16 levels) or of plain fp32 math."""
    W = lin.dequant(levels).double()
    if exact_act:
        q, d = quant_ref(x)
        xa = (q.view(x.shape[0], -1, 32) * d[..., None]).view(x.shape).double()
    else:
        xa = x.double()
    y = xa @ W.T
    if res is not None:
        y = y + res.double()
    return y


def rms_ref(x, gain, eps=1e-6):
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return bfr(xf * r * gain)


# ------------------------------------------------------------------------------------------------ kernel tests
def test_kernels(E):
    X = ext()
    M = 3
    # NF4 GEMV (all row tilings, bf16 + residual, fp32 out)
    lin = E.layers[0]["gu"]
    x = torch.randn(M, HID, device=DEV).to(torch.bfloat16)
    res = torch.randn(M, lin.N, device=DEV).to(torch.bfloat16)
    xq, xd = E.xq(HID)
    X.quant_rows(x, HID, xq, xd, M, HID)
    q, d = quant_ref(x)
    check("quant_rows codes", torch.equal(xq[:M * HID].view(M, HID).float(), q) and torch.allclose(xd[:M * HID // 32].view(M, -1), d))
    ref = nf4_ref(lin, E.levels, x, res)
    if E.gemv == "dp4a":
        for rows in (21, 22, 41, 42, 81, 12, 14, 24, 1021, 1041):
            y = torch.empty(M, lin.N, dtype=torch.bfloat16, device=DEV)
            X.nf4_gemv(lin.codes, lin.scales, xq, xd, res, lin.N, y, lin.N, M, rows)
            check(f"nf4_gemv rows={rows}", rel(y.float(), ref) < 4e-3, f"rel {rel(y.float(), ref):.2e}")
    y = torch.empty(M, lin.N, dtype=torch.bfloat16, device=DEV)
    lin(xq, xd, M, y, lin.N, res, lin.N)
    check(f"NF4 linear ({E.gemv}) + residual", rel(y.float(), ref) < 4e-3, f"rel {rel(y.float(), ref):.2e}")
    yf = torch.empty(M, E.head.N, device=DEV)
    E.head(xq, xd, M, yf, E.head.N)
    r0 = nf4_ref(E.head, E.levels, x)
    check(f"NF4 head fp32 out ({E.gemv})", rel(yf, r0) < 1e-5, f"rel {rel(yf, r0):.2e}")
    check("nf4 levels int16-representation error <= 3.2e-5", float((E.levels - E.levels_true).abs().max()) <= 3.2e-5,
          f"{float((E.levels - E.levels_true).abs().max()):.2e}")
    # single-row results equal the M-row results (bit for bit)
    y1 = torch.empty(1, lin.N, dtype=torch.bfloat16, device=DEV)
    X.quant_rows(x[1:2], HID, xq, xd, 1, HID)
    lin(xq, xd, 1, y1, lin.N, res[1:2], lin.N)
    X.quant_rows(x, HID, xq, xd, M, HID)
    yM = torch.empty(M, lin.N, dtype=torch.bfloat16, device=DEV)
    lin(xq, xd, M, yM, lin.N, res, lin.N)
    check(f"NF4 linear row 1 of M=3 == M=1 ({E.gemv})", torch.equal(y1[0], yM[1]))
    # RMSNorm + quant
    g = E.layers[0]["ln1"]
    y = torch.empty(M, HID, dtype=torch.bfloat16, device=DEV)
    X.rms_rows(x, HID, g, y, HID, xq, xd, M, HID, 1e-6)
    yr = rms_ref(x, g)
    check("rms_rows", (y.float() - yr).abs().max() <= 2 ** -7 * yr.abs().max(), f"max|d| {(y.float() - yr).abs().max():.2e}")
    # dgemv
    d0 = E.layers[0]
    sg = torch.empty(M, d0["Ws"].shape[0], device=DEV)
    X.dgemv(d0["Ws"], HID, d0["Ws"].shape[0], HID, y, HID, sg, sg.shape[1], M, False)
    sr = y.float() @ d0["Ws"].float().T
    check("dgemv vec", rel(sg, sr) < 1e-5, f"rel {rel(sg, sr):.2e}")
    a3 = E.layers[E.attn_ids[0]]
    rr = a3["r"]
    xc = torch.randn(M, rr, device=DEV).to(torch.bfloat16)
    yc = torch.empty(M, rr, device=DEV)
    X.dgemv(a3["R"], rr, rr, rr, xc, rr, yc, rr, M, False)
    check("dgemv scalar (odd K ok)", rel(yc, xc.float() @ a3["R"].float().T) < 1e-5)
    # gates
    i0 = E.lin_ids[0]
    X.dgemv(d0["Ws"], HID, d0["Ws"].shape[0], HID, y, HID, sg, sg.shape[1], M, True)
    gd = torch.empty(M, VALD, device=DEV); be = torch.empty_like(gd); bw = torch.empty_like(gd)
    X.gdn2_gates(sg, sg.shape[1], NV, E.rank, d0["lr"], d0["B"], d0["Bl"], d0["vd"], d0["vb"], d0["A_log"], d0["dt_bias"],
                 gd, be, bw, M, DKH)
    base = sg[:, :3 * NV].view(M, 3, NV)
    u = sg[:, 3 * NV:3 * NV + E.rank]
    tg = [bfr(d0["vb"][k] * (bfr(d0["vd"][k] * u) @ d0["B"].float().T)) for k in range(3)]
    pre = [bfr(base[:, k].repeat_interleave(DKH, -1) + tg[k]) for k in range(3)]
    a = pre[0]
    if d0["lr"]:
        a = bfr(a + bfr(sg[:, 3 * NV + E.rank:3 * NV + E.rank + d0["lr"]] @ d0["Bl"].float().T))
    gref = -d0["A_log"].exp() * F.softplus(a + d0["dt_bias"])
    check("gdn2_gates decay", rel(gd, gref) < 1e-5, f"rel {rel(gd, gref):.2e}")
    check("gdn2_gates erase/write", rel(be, bfr(torch.sigmoid(pre[1]))) < 1e-5 and rel(bw, bfr(torch.sigmoid(pre[2]))) < 1e-5)
    # tensor-core path (skinny GEMM for base|A|lora and for the VeRA B product)
    x16 = torch.zeros(16, HID, dtype=torch.bfloat16, device=DEV); x16[:M] = y
    sg2 = torch.empty(M, d0["Ws"].shape[0], device=DEV)
    X.tc_skinny(d0["Ws"], HID, d0["Ws"].shape[0], HID, x16, HID, sg2, sg2.shape[1], M, True)
    check("tc_skinny (base|A|lora)", rel(sg2, bfr(y.float() @ d0["Ws"].float().T)) < 3e-3, f"rel {rel(sg2, bfr(y.float() @ d0['Ws'].float().T)):.2e}")
    gd2 = torch.empty_like(gd); be2 = torch.empty_like(gd); bw2 = torch.empty_like(gd)
    X.gdn2_gates_tc(sg, sg.shape[1], NV, E.rank, d0["lr"], d0["B"], d0["Bl"], d0["vd"], d0["vb"], d0["A_log"], d0["dt_bias"],
                    E.W2, E.T, gd2, be2, bw2, M, DKH)
    check("gdn2_gates_tc decay / erase / write", rel(gd2, gref) < 1e-4 and rel(be2, be) < 1e-3 and rel(bw2, bw) < 1e-3,
          f"rel {rel(gd2, gref):.2e} {rel(be2, be):.2e} {rel(bw2, bw):.2e}")
    # M invariance of the tensor-core path: row 1 alone == row 1 of M = 3
    x1 = torch.zeros(16, HID, dtype=torch.bfloat16, device=DEV); x1[0] = y[1]
    sg1 = torch.empty(1, d0["Ws"].shape[0], device=DEV)
    X.tc_skinny(d0["Ws"], HID, d0["Ws"].shape[0], HID, x1, HID, sg1, sg1.shape[1], 1, True)
    g1 = torch.empty(1, VALD, device=DEV); b1 = torch.empty_like(g1); w1 = torch.empty_like(g1)
    X.gdn2_gates_tc(sg1, sg1.shape[1], NV, E.rank, d0["lr"], d0["B"], d0["Bl"], d0["vd"], d0["vb"], d0["A_log"], d0["dt_bias"],
                    E.W2, E.T, g1, b1, w1, 1, DKH)
    X.gdn2_gates_tc(sg2, sg2.shape[1], NV, E.rank, d0["lr"], d0["B"], d0["Bl"], d0["vd"], d0["vb"], d0["A_log"], d0["dt_bias"],
                    E.W2, E.T, gd2, be2, bw2, M, DKH)
    check("tc gates: row of M=3 == M=1 (bit)", torch.equal(sg1[0], sg2[1]) and torch.equal(g1[0], gd2[1]) and torch.equal(b1[0], be2[1]))
    # conv + recurrence (vs fla's naive GDN-2 recurrence + gated norm), plain and verify/commit equivalence
    st = torch.randn(CONV, 4, device=DEV).to(torch.bfloat16).float()
    xin = (torch.randn(M, CONV + VALD, device=DEV) * 2).to(torch.bfloat16)
    S0 = torch.randn(NV, DKH, DKH, device=DEV) * 0.05
    mixed = torch.empty(M, CONV, device=DEV)
    xs = torch.empty(M, CONV, dtype=torch.bfloat16, device=DEV)
    st1 = st.clone()
    X.gdn2_conv(xin, CONV + VALD, st1, d0["conv_w"], mixed, xs, M, True)
    win = torch.cat([st, xin[:, :CONV].float().T], 1)                        # (C, 4 + M)
    cref = torch.stack([bfr(F.silu((win[:, t + 1:t + 5] * d0["conv_w"]).sum(-1))) for t in range(M)])
    check("gdn2_conv", rel(mixed, cref) < 1e-6 and torch.equal(st1, win[:, -4:]), f"rel {rel(mixed, cref):.2e}")
    out = torch.empty(M, VALD, dtype=torch.bfloat16, device=DEV)
    xq4, xd4 = E.xq(VALD)
    S1 = S0.clone()
    X.gdn2_rec(mixed, xin[:, CONV:], CONV + VALD, gd, be, bw, d0["gn"], S1, out, xq4, xd4, NK, NV, 1e-6, M, 0, None, d0["pe_c"])
    from fla.ops.gdn2.naive import naive_recurrent_gdn2
    q, k, v = torch.split(mixed, [NK * DKH, NK * DKH, VALD], -1)
    rep = NV // NK
    q = q.view(1, M, NK, DKH).repeat_interleave(rep, 2); k = k.view(1, M, NK, DKH).repeat_interleave(rep, 2)
    q = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt(); k = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    o, Sf = naive_recurrent_gdn2(q, k, v.view(1, M, NV, DKH), gd.view(1, M, NV, DKH), be.view(1, M, NV, DKH),
                                 bw.view(1, M, NV, DKH), initial_state=S0[None].clone(), output_final_state=True)
    o = bfr(o[0].float())
    on = bfr(o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6))
    on = bfr(on * d0["gn"].float()) * F.silu(xin[:, CONV:].float().view(M, NV, DKH))
    check("gdn2_rec output vs fla naive", rel(out.float(), on.view(M, -1)) < 5e-3, f"rel {rel(out.float(), on.view(M, -1)):.2e}")
    check("gdn2_rec state vs fla naive", rel(S1, Sf[0]) < 1e-4, f"rel {rel(S1, Sf[0]):.2e}")
    # verify (no state write) + commit(n) == plain run of n + 1 tokens
    for n in range(M):
        st2, S2 = st.clone(), S0.clone()
        mixed2 = torch.empty_like(mixed)
        X.gdn2_conv(xin, CONV + VALD, st2, d0["conv_w"], mixed2, xs, M, False)
        out2 = torch.empty_like(out)
        X.gdn2_rec(mixed2, xin[:, CONV:], CONV + VALD, gd, be, bw, d0["gn"], S2, out2, xq4, xd4, NK, NV, 1e-6, M, 1, None, d0["pe_c"])
        ok_unchanged = torch.equal(S2, S0) and torch.equal(st2, st)
        nacc = torch.tensor([n], dtype=torch.int32, device=DEV)
        X.gdn2_conv_commit(st2, xs, nacc)
        X.gdn2_rec(mixed2, xin[:, CONV:], CONV + VALD, gd, be, bw, d0["gn"], S2, out2, xq4, xd4, NK, NV, 1e-6, 1, 2, nacc, d0["pe_c"])
        st3, S3 = st.clone(), S0.clone()
        mixed3 = torch.empty(n + 1, CONV, device=DEV)
        X.gdn2_conv(xin[:n + 1], CONV + VALD, st3, d0["conv_w"], mixed3, None, n + 1, True)
        out3 = torch.empty(n + 1, VALD, dtype=torch.bfloat16, device=DEV)
        X.gdn2_rec(mixed3, xin[:n + 1, CONV:], CONV + VALD, gd[:n + 1], be[:n + 1], bw[:n + 1], d0["gn"], S3, out3, xq4, xd4,
                   NK, NV, 1e-6, n + 1, 0, None, d0["pe_c"])
        check(f"gdn2 verify+commit(n={n}) == plain", ok_unchanged and torch.equal(S2, S3) and torch.equal(st2, st3)
              and torch.equal(out2[:n + 1], out3))
    # embedding
    tok = torch.tensor([0, 1234, E.V - 1], device=DEV)
    he = torch.empty(3, HID, dtype=torch.bfloat16, device=DEV)
    E.embed(tok, he, 3)
    w8 = E.head.packed()[tok].long()
    idx = torch.stack([w8 & 15, w8 >> 4], -1).view(3, HID)
    eref = bfr(E.emb_code[idx].view(3, -1, 64) * E.emb_absmax[tok][..., None]).view(3, HID)
    check("emb_nf4", torch.equal(he.float(), eref))
    # argmax
    lg = torch.randn(2, 1000, device=DEV)
    lg[1, 7] = lg[1, 9] = 50.0
    am = torch.empty(2, dtype=torch.long, device=DEV)
    X.argmax_rows(lg, 1000, 900, am, 2)
    check("argmax (first max, vocab limit)", am.tolist() == [int(lg[0, :900].argmax()), 7])
    test_mla_kernels(E)


def tq_ref_encode(x, cb):
    """x (n, r) fp32 values -> (idx (n, r), norm fp16) exactly as mercurius_cache.tq_encode"""
    bnd = cb[16:]
    n = x.norm(dim=-1, keepdim=True).half()
    idx = torch.bucketize(x / n.float().clamp_min(1e-12), bnd)
    return idx, n.squeeze(-1)


def decode_cache_ref(d, cache, n):
    """rows [0, n) -> pos, inv_r (n, 4), rope (n, 64), latent (n, r) dequantized (fp32)"""
    r = d["r"]
    rows = cache[:n]
    pos = rows[:, 0:4].contiguous().view(torch.int32).view(n).long()
    inv = rows[:, 4:20].contiguous().view(torch.float32).view(n, 4)
    nl = rows[:, 20:22].contiguous().view(torch.float16).view(n).float().clamp_min(1e-12)
    nr = rows[:, 22:24].contiguous().view(torch.float16).view(n).float().clamp_min(1e-12)
    unpack = lambda b, w: torch.stack([b & 15, b >> 4], -1).view(n, -1)[:, :w].long()
    rc = unpack(rows[:, 24:24 + RD // 2], RD)
    lc = unpack(rows[:, 24 + RD // 2:24 + RD // 2 + (r + 1) // 2], r)
    return pos, inv, d["cbr"][:16][rc] * nr[:, None], d["cbl"][:16][lc] * nl[:, None]


def rope_ref(x, pos, inv_freq):
    ang = pos.float()[:, None] * inv_freq[None]
    c, s = bfr(ang.cos()), bfr(ang.sin())
    x0, x1 = x[..., :32], x[..., 32:64]
    return torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], -1)


def attn_ref(E, d, Q, cache, p0, NQ):
    """fp32 absorbed attention over the decoded cache: (NQ, 16, r)"""
    n = p0 + NQ
    pos, inv, kr, lat = decode_cache_ref(d, cache, n)
    kt = bfr(bfr(kr) @ d["R0"].float())                      # un-rotate (kernel: fp16 decode, fp32 GEMM, bf16 round)
    k = torch.cat([rope_ref(kt, pos, E.inv_freq), lat], -1)
    out = []
    for qi in range(NQ):
        q = Q[qi, :, :d["E"]].float()                          # (16, E)
        s = (q @ k.T) * inv[:, torch.arange(NH, device=DEV) // 4].T * d["scale"]
        s = s.masked_fill(pos[None] > p0 + qi, float("-inf"))
        out.append(torch.softmax(s, -1) @ lat)
    return torch.stack(out)


def test_mla_kernels(E):
    X = ext()
    i = E.attn_ids[0]
    d = E.layers[i]
    r, Ep, RW = d["r"], d["Ep"], d["RW"]
    # encode: random rows at positions 0..n-1
    n = 100
    cache = torch.zeros(256, RW, dtype=torch.uint8, device=DEV)
    kk = torch.randn(n, NG * HD, device=DEV).to(torch.bfloat16)
    kr = bfr(torch.randn(n, RD, device=DEV))
    cr = bfr(torch.randn(n, r, device=DEV) * 3)
    for p0 in range(0, n, 4):
        m = min(4, n - p0)
        pos = torch.tensor([p0], device=DEV)
        X.mla_encode(kk[p0:p0 + m], NG * HD, kr[p0:p0 + m].contiguous(), cr[p0:p0 + m].contiguous(), r, d["cbl"], d["cbr"],
                     pos, cache, RW, d["k_eps"], m)
    pos_, inv, kd, ld = decode_cache_ref(d, cache, n)
    ir, nr_ = tq_ref_encode(kr, d["cbr"])
    il, nl_ = tq_ref_encode(cr, d["cbl"])
    inv_ref = torch.rsqrt(kk.float().view(n, NG, HD).pow(2).mean(-1) + d["k_eps"])
    kd_ref = d["cbr"][:16][ir] * nr_.float()[:, None]
    ld_ref = d["cbl"][:16][il] * nl_.float()[:, None]
    check("mla_encode positions", torch.equal(pos_, torch.arange(n, device=DEV)))
    check("mla_encode inv_r", rel(inv, inv_ref) < 1e-6)
    check("mla_encode rope codes == tq_encode", torch.equal(kd, kd_ref), f"mismatch {(kd != kd_ref).float().mean():.2e}")
    check("mla_encode latent codes == tq_encode", torch.equal(ld, ld_ref), f"mismatch {(ld != ld_ref).float().mean():.2e}")
    # query map
    M = 2
    yq = (torch.randn(M, E.NQD[i], device=DEV)).to(torch.bfloat16)
    Q = torch.zeros(M, NH, Ep, dtype=torch.float16, device=DEV)
    p0 = torch.tensor([37], device=DEV)
    X.mla_query(yq, E.NQD[i], d["qn"], d["q_eps"], d["P"], d["E"], Ep, p0, E.inv_freq, Q, M)
    qh = yq[:, :NH * 2 * HD].float().view(M, NH, 2 * HD)[..., :HD]
    y = bfr(qh * torch.rsqrt(qh.pow(2).mean(-1, keepdim=True) + d["q_eps"]) * d["qn"])
    qq = torch.einsum("mhd,hed->mhe", y, d["P"].float())
    qref = torch.cat([rope_ref(qq[..., :RD].reshape(-1, RD), torch.tensor([37, 38], device=DEV)[:, None].expand(M, NH).reshape(-1),
                               E.inv_freq).view(M, NH, RD), qq[..., RD:]], -1)
    check("mla_query", rel(Q[..., :d["E"]].float(), qref) < 1e-3 and Q[..., d["E"]:].abs().max() == 0,
          f"rel {rel(Q[..., :d['E']].float(), qref):.2e}")
    # attention from the codes: 1..4 query tokens, several lengths (chunk tails, multi-chunk per block)
    for L in (1, 31, 33, 100, 2000, 5000):
        cache2 = torch.zeros(L + 8, RW, dtype=torch.uint8, device=DEV)
        nkk = torch.randn(L + 4, NG * HD, device=DEV).to(torch.bfloat16)
        nkr = bfr(torch.randn(L + 4, RD, device=DEV) * 2)
        ncr = bfr(torch.randn(L + 4, r, device=DEV) * 4)
        for a in range(0, L + 4, 4):
            X.mla_encode(nkk[a:a + 4], NG * HD, nkr[a:a + 4].contiguous(), ncr[a:a + 4].contiguous(), r, d["cbl"], d["cbr"],
                         torch.tensor([a], device=DEV), cache2, RW, d["k_eps"], 4)
        Qr = (torch.randn(4, NH, Ep, device=DEV) * 0.3)
        Qr[..., d["E"]:] = 0
        Qr = Qr.half()
        for fname in ("mla_attn", "mla_attn_fd"):
          fn = getattr(X, fname)
          outs = {}
          for NQ in (1, 2, 3, 4):
              p = L - NQ
              if p < 0:
                  continue
              po = torch.empty(E.NS, NQ, NH, Ep - RD, device=DEV)
              pm = torch.empty(E.NS, NQ * NH * 2, device=DEV)
              o = torch.empty(NQ, NH, r, device=DEV)
              fn(Qr[:NQ].contiguous(), Ep, cache2, RW, d["R0h"], d["cbl"], d["cbr"], torch.tensor([p], device=DEV),
                         E.inv_freq, po, pm, o, r, d["scale"], NQ)
              ref = attn_ref(E, d, Qr, cache2, p, NQ)
              check(f"{fname} L={L} NQ={NQ}", rel(o, ref) < 3e-3, f"rel {rel(o, ref):.2e}")
              outs[NQ] = (p, o)
          # bit identity: query token qi of an NQ-block == the single-token call at that position
          for NQ, (p, o) in outs.items():
              for qi in range(NQ):
                  po = torch.empty(E.NS, 1, NH, Ep - RD, device=DEV)
                  pm = torch.empty(E.NS, NH * 2, device=DEV)
                  o1 = torch.empty(1, NH, r, device=DEV)
                  fn(Qr[qi:qi + 1].contiguous(), Ep, cache2, RW, d["R0h"], d["cbl"], d["cbr"],
                             torch.tensor([p + qi], device=DEV), E.inv_freq, po, pm, o1, r, d["scale"], 1)
                  if not torch.equal(o1[0], o[qi]):
                      check(f"{fname} bit identity L={L} NQ={NQ} qi={qi}", False, f"max|d| {(o1[0] - o[qi]).abs().max():.2e}")
                      break
              else:
                  continue
              break
          else:
              check(f"{fname} bit identity (all NQ, all rows) L={L}", True)
    # V up-projection + gate
    olat = torch.randn(M, NH, r, device=DEV)
    ya = torch.empty(M, NH * HD, dtype=torch.bfloat16, device=DEV)
    X.mla_vup(olat, r, d["Wv"], yq, E.NQD[i], ya, M)
    o = torch.einsum("mgkc,gdc->mgkd", olat.view(M, NG, 4, r), d["Wv"].float()).reshape(M, NH, HD)
    gate = yq[:, :NH * 2 * HD].float().view(M, NH, 2 * HD)[..., HD:]
    yr = bfr(bfr(o) * bfr(torch.sigmoid(gate)))
    check("mla_vup", rel(ya.float().view(M, NH, HD), yr) < 3e-3, f"rel {rel(ya.float().view(M, NH, HD), yr):.2e}")


# ------------------------------------------------------------------------------------------------ engine tests
def ref_layer(E, i, h, pos, st, act8=False):
    """PyTorch re-implementation of layer i (attention / GDN-2 mixer + MLP) for one token: h (1, HID) float (bf16
    values) -> new h. fp32 math at the kernels' bf16 rounding points, attention by decoding the cache codes; act8: GEMV
    inputs quantized to int8 groups as the kernels do. st: {"conv", "rec", "cache"} copies, updated in place."""
    X = ext()
    lv = E.levels.to(DEV).float()

    def deq8(x):
        q, d = quant_ref(x.float())
        return (q.view(x.shape[0], -1, 32) * d[..., None]).view(x.shape)
    lin = lambda L_, x: (deq8(x) if act8 else x).double() @ L_.dequant(lv).double().T
    d = E.layers[i]
    x = rms_ref(h, d["ln1"])
    if i in E.lin_ids:
        qkvz = bfr(lin(d["qkvz"], x).float())
        sg = bfr(x @ d["Ws"].float().T)
        base = sg[:, :3 * NV]
        u = sg[:, 3 * NV:3 * NV + E.rank]
        tg = [bfr(d["vb"][k] * (bfr(d["vd"][k] * u) @ d["B"].float().T)) for k in range(3)]
        pre = [bfr(base[:, k * NV:(k + 1) * NV].repeat_interleave(DKH, -1) + tg[k]) for k in range(3)]
        a = pre[0]
        if d["lr"]:
            a = bfr(a + bfr(sg[:, 3 * NV + E.rank:3 * NV + E.rank + d["lr"]] @ d["Bl"].float().T))
        gdec = -d["A_log"].exp() * F.softplus(a + d["dt_bias"])
        be, bw = bfr(torch.sigmoid(pre[1])), bfr(torch.sigmoid(pre[2]))
        cs = st["conv"][i]
        cs.copy_(torch.cat([cs[:, 1:], qkvz[0, :CONV, None]], 1))
        mixed = bfr(F.silu((cs * d["conv_w"]).sum(-1)))
        q, k, v = torch.split(mixed, [NK * DKH, NK * DKH, VALD])
        q = q.view(NK, DKH).repeat_interleave(2, 0); k = k.view(NK, DKH).repeat_interleave(2, 0)
        q = q / (q.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt() * DKH ** -0.5
        k = k / (k.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
        S = st["rec"][i]
        S.mul_(gdec.view(NV, DKH, 1).exp())
        er = torch.einsum("hkv,hk->hv", S, be.view(NV, DKH) * k)
        vn = bw.view(NV, DKH) * v.view(NV, DKH) - er
        S.add_(k[:, :, None] * vn[:, None, :])
        o = bfr(torch.einsum("hkv,hk->hv", S, q))
        on = bfr(bfr(o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + d["gn_eps"])) * d["gn"].float())
        on = bfr(on * F.silu(qkvz[0, CONV:].view(NV, DKH)))
        h = bfr(lin(d["out"], on.view(1, -1)).float() + h)
    else:
        r = d["r"]
        yq = bfr(lin(d["qdk"], x).float())
        kt0 = bfr(x @ d["Wr"].float().T)
        kr = bfr(kt0 @ d["R0"].float().T)
        c = yq[:, NH * 2 * HD:NH * 2 * HD + r]
        crot = bfr(c @ d["R"].float().T)
        X.mla_encode(yq[:, NH * 2 * HD + r:].to(torch.bfloat16).contiguous(), NG * HD, kr.contiguous(), crot.contiguous(),
                     r, d["cbl"], d["cbr"], torch.tensor([pos], device=DEV), st["cache"][i], d["RW"], d["k_eps"], 1)
        qh = yq[:, :NH * 2 * HD].view(NH, 2 * HD)[:, :HD]
        y = bfr(qh * torch.rsqrt(qh.pow(2).mean(-1, keepdim=True) + d["q_eps"]) * d["qn"])
        qq = torch.einsum("hd,hed->he", y, d["P"].float())
        qf = torch.cat([rope_ref(qq[:, :RD], torch.full((NH,), pos, device=DEV), E.inv_freq), qq[:, RD:]], -1)
        Qp = torch.zeros(1, NH, d["Ep"], device=DEV)
        Qp[0, :, :d["E"]] = qf
        olat = attn_ref(E, d, Qp.half(), st["cache"][i], pos, 1)[0]
        o = torch.einsum("gkc,gdc->gkd", olat.view(NG, 4, r), d["Wv"].float()).reshape(NH, HD)
        gate = yq[:, :NH * 2 * HD].view(NH, 2 * HD)[:, HD:]
        ya = bfr(bfr(o) * bfr(torch.sigmoid(gate)))
        h = bfr(lin(d["o"], ya.view(1, -1)).float() + h)
    x = rms_ref(h, d["ln2"])
    gu = bfr(lin(d["gu"], x).float())
    act = bfr(bfr(F.silu(gu[:, :FF])) * gu[:, FF:])
    return bfr(lin(d["down"], act).float() + h)


def ref_engine_step(E, tok, pos, st, act8=False):
    """one plain decode step with ref_layer for every layer."""
    lv = E.levels.to(DEV).float()
    h = E.emb_rows(tok).float()
    for i in range(len(E.layers)):
        h = ref_layer(E, i, h, pos, st, act8)
    x = rms_ref(h, E.norm_f)
    if act8:
        q, dd = quant_ref(x.float())
        x = (q.view(1, -1, 32) * dd[..., None]).view(x.shape)
    return (x.double() @ E.head.dequant(lv).double().T).float()[0]


def local_layer_errors(E, snap, tok):
    """engine layer i vs ref_layer i, both from the engine's own input (and state copies): local relative error."""
    E.load(snap)
    pos = int(E.pos)
    E.tok_in[0] = tok
    E.embed(E.tok_in, E.h, 1)
    out = []
    for i, d in enumerate(E.layers):
        h_in = E.h[:1].float().clone()
        st = {"conv": {j: E.conv[j].clone() for j in E.lin_ids}, "rec": {j: E.rec[j].clone() for j in E.lin_ids},
              "cache": {j: E.layers[j]["cache"].clone() for j in E.attn_ids}}
        if i in E.lin_ids:
            E._gdn(i, d, 1, 0)
        else:
            E._mla(i, d, 1)
        E._mlp(d, 1)
        hr = ref_layer(E, i, h_in, pos, st, act8=True)
        out.append((i, "gdn" if i in E.lin_ids else "mla", rel(E.h[:1].float() - h_in, hr - h_in)))
    return out


def test_engine(E, snap, ref):
    st0 = E.save_state()
    E.load(snap)
    s_loaded = E.save_state()
    caches = {i: E.layers[i]["cache"].clone() for i in E.attn_ids}
    # engine (eager) vs PyTorch re-implementation, teacher-forced on the reference tokens
    toks = ref["tokens"][:8].to(DEV)
    st = {"conv": {i: E.conv[i].clone() for i in E.lin_ids}, "rec": {i: E.rec[i].clone() for i in E.lin_ids},
          "cache": {i: caches[i].clone() for i in E.attn_ids}}
    pos = int(E.pos)
    st8 = {kk: {i: v.clone() for i, v in vv.items()} for kk, vv in st.items()}
    errs, errs8, agree, agree8 = [], [], 0, 0
    for t in range(6):
        E.tok_in[0] = toks[t]
        E._block(1, 0)
        lg = E.logits[0, :E.V].clone()
        lr = ref_engine_step(E, toks[t:t + 1], pos + t, st)[:E.V]
        lr8 = ref_engine_step(E, toks[t:t + 1], pos + t, st8, act8=True)[:E.V]
        errs.append(rel(lg, lr)); errs8.append(rel(lg, lr8))
        agree += int(lg.argmax() == lr.argmax()); agree8 += int(lg.argmax() == lr8.argmax())
    # end to end the per-layer differences (GDN-2 ~0, MLA ~1e-2 from the fp16 tensor-core attention vs the fp32-key
    # reference; see the per-layer check below) accumulate over 32 layers to a few % of the logits: bound 6e-2 + argmax
    check("engine step vs PyTorch re-implementation with int8 activations emulated (6 tokens)",
          max(errs8) < 6e-2 and agree8 == 6, f"max rel logits {max(errs8):.2e}, argmax agree {agree8}/6")
    print(f"  [info] vs exact-activation re-implementation: max rel logits {max(errs):.2e}, argmax agree {agree}/6 "
          f"(the int8 activation cost)", flush=True)
    errs_l = local_layer_errors(E, snap, toks[0])
    worst = max(errs_l, key=lambda x: x[2])
    check("per-layer local error (update of h, engine vs int8-emulated reference) < 2e-2", worst[2] < 2e-2,
          f"worst layer {worst[0]} ({worst[1]}) {worst[2]:.2e}; " + " ".join(f"{i}:{e:.1e}" for i, _, e in errs_l))
    # graphs == eager
    E.load(snap)
    a = E.generate(24, graph=False)
    E.load(snap)
    b = E.generate(24, graph=True)
    check("graph == eager (24 greedy tokens)", torch.equal(a, b))
    # verify pass (M rows, mode 1) + commit == plain decoding, bit for bit (logits and states)
    for M in (2, 3, 4):
        E.load(snap)
        plain_logits = []
        seq = []
        for t in range(M):
            seq.append(int(E.tok_in[0]))
            E._block(1, 0)
            plain_logits.append(E.logits[0].clone())
        plain_state = E.save_state()
        plain_cache = {i: E.layers[i]["cache"][:int(E.pos)].clone() for i in E.attn_ids}
        E.load(snap)
        E.tok_in[:M] = torch.tensor(seq, device=DEV)
        E._block(M, 1)
        same_logits = all(torch.equal(E.logits[t], plain_logits[t]) for t in range(M))
        E.nacc.fill_(M - 1)
        for i in E.lin_ids:
            E._gdn_commit(i)
        E.pos.add_(M)
        vs = E.save_state()
        same_state = all(torch.equal(vs["conv"][i], plain_state["conv"][i]) and torch.equal(vs["rec"][i], plain_state["rec"][i])
                         for i in E.lin_ids)
        same_cache = all(torch.equal(E.layers[i]["cache"][:int(E.pos)], plain_cache[i]) for i in E.attn_ids)
        check(f"verify M={M} + commit == plain (logits, GDN states, MLA cache)", same_logits and same_state and same_cache,
              f"logits {same_logits} states {same_state} cache {same_cache}")
    # fused-commit speculative decoding: tokens == plain, and after finalize the GDN states / MLA cache == plain decoding
    from cairn.mercurius_mtp import MTPHead, SpecDecoder
    H = MTPHead(E, Tmax=E.Tmax)
    for k, adaptive in ((1, False), (3, False), (2, True)):
        S = SpecDecoder(E, H, k=k, adaptive=adaptive)
        E.load(snap)
        S.start(snap)
        toks_s, cyc, hist = S.generate(48)
        n_spec = int(E.pos) - int(snap["ids"].shape[0])
        st_spec = E.save_state()
        cache_spec = {i: E.layers[i]["cache"][:int(E.pos)].clone() for i in E.attn_ids}
        E.load(snap)
        toks_p = E.generate(n_spec)
        st_p = E.save_state()
        m_ = min(48, n_spec)
        same_tok = torch.equal(toks_s[:m_], toks_p[:m_])
        same_state = all(torch.equal(st_spec["conv"][i], st_p["conv"][i]) and torch.equal(st_spec["rec"][i], st_p["rec"][i])
                         for i in E.lin_ids) and torch.equal(st_spec["pos"], st_p["pos"])
        same_cache = all(torch.equal(cache_spec[i], E.layers[i]["cache"][:int(E.pos)]) for i in E.attn_ids)
        check(f"speculative (fused commit, k={'adaptive' if adaptive else k}) == plain: tokens, GDN states, MLA cache "
              f"after {n_spec} positions", same_tok and same_state and same_cache,
              f"tokens {same_tok} states {same_state} cache {same_cache} cycles {cyc}")
    E.load_state(st0)


def main():
    wpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "weights.pt")
    spath = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUT, "snap_chat0.pt")
    t0 = time.time()
    E = MercuriusEngine(wpath, Tmax=8192, gemv=os.environ.get("CAIRN_GEMV", "dp4a"))
    E.levels_true = E.emb_code.cpu().double()
    print(f"engine built in {time.time() - t0:.0f}s", flush=True)
    test_kernels(E)
    snap = torch.load(spath, weights_only=False)
    ref = torch.load(spath.replace("snap_", "ref_"), weights_only=False)
    test_engine(E, snap, ref)
    print("FAILS:", FAILS)
    print("TESTS_DONE")


if __name__ == "__main__":
    main()
