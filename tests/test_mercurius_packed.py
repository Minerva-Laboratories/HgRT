"""mercurius-packed-v1 support (hgrt.packed_kernels, hgrt.mercurius_packed).

Kernel tests run on synthetic tensors (no checkpoint): the Q4_0 / Q8_0 GEMVs against the dequantized weights and their
bit identity across M (speculative verification), the Q4_0 embedding, and the latent MLA attention against PyTorch.
The checkpoint test needs HGRT_MERCURIUS05 = a mercurius-packed-v1 directory (e.g. Mercurius-0.5-4B-Preview).
"""
import os

import pytest
import torch

pytestmark = pytest.mark.gpu
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _q4(N, K, g):
    words = torch.randint(-2 ** 31, 2 ** 31 - 1, (N, K // 8), generator=g, dtype=torch.int64).to(torch.int32)
    scales = (torch.rand(N, K // 32, generator=g) * 0.02 + 1e-3).half()
    return {"weight_packed": words, "weight_scale": scales}


def _q8(N, K, g):
    return {"weight_q8": torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8),
            "weight_scale": (torch.rand(N, K // 32, generator=g) * 0.002 + 1e-4).half()}


@cuda
@pytest.mark.parametrize("fmt", ["q4_0", "q8_0"])
@pytest.mark.parametrize("rounded", [False, True])
def test_packed_gemv(fmt, rounded):
    """both numerics: unrounded (q - 8) d (default, GGUF) and bf16((q - 8) d) (the QAT forward), each against its own
    weights; rows bit-identical across M = 1..4."""
    from hgrt import packed_kernels as pk
    g = torch.Generator().manual_seed(0)
    N, K = 1000, 2560                                        # N not a multiple of the rows per block
    lin = pk.PackedLin([(_q4 if fmt == "q4_0" else _q8)(N, K, g)], fmt)
    W = lin.dequant_exact() if not rounded else lin.dequant().float()
    x = torch.randn(4, K, generator=g).to("cuda", torch.bfloat16)
    ref = x.float() @ W.T
    pk.ROUND["bf16"] = rounded
    try:
        for R in (1, 2, 4):
            lin.R = R
            full = lin(x, 4, torch.empty(4, N, device="cuda"))
            assert ((full - ref).abs().max() / ref.abs().max()).item() < 1e-5
            for M in (1, 2, 3):
                assert torch.equal(lin(x, M, torch.empty(M, N, device="cuda")), full[:M])
    finally:
        pk.ROUND["bf16"] = False


@cuda
def test_emb_q4():
    from hgrt.packed_kernels import PackedLin, ext
    g = torch.Generator().manual_seed(1)
    lin = PackedLin([_q4(300, 512, g)], "q4_0")
    tok = torch.tensor([0, 7, 299], device="cuda")
    y = torch.empty(3, 512, device="cuda", dtype=torch.bfloat16)
    ext().emb_q4(tok, lin.codes, lin.scales, y, 3)
    assert torch.equal(y, lin.dequant()[tok])


@cuda
@pytest.mark.parametrize("r", [384, 512, 640])
def test_mla_lat_attn(r):
    from hgrt.packed_kernels import ext
    torch.manual_seed(r)
    H, G, Tmax, ACH = 16, 4, 2048, 32
    kr = torch.randn(Tmax, 64, device="cuda").to(torch.bfloat16)
    c = torch.randn(Tmax, r, device="cuda").to(torch.bfloat16)
    ir = torch.rand(Tmax, G, device="cuda") + 0.5
    qr, ql = torch.randn(H, 64, device="cuda"), torch.randn(H, r, device="cuda") * 0.1
    nch = Tmax // ACH
    pm, pl, pacc, out = (torch.empty(nch * H, device="cuda"), torch.empty(nch * H, device="cuda"),
                         torch.empty(nch * H * r, device="cuda"), torch.empty(H, r, device="cuda"))
    grp = torch.arange(H, device="cuda") // (H // G)
    for pos in (0, 31, 32, 777, Tmax - 1):
        n = pos + 1
        ext().mla_lat_attn(qr, ql, kr, c, ir, torch.tensor([pos], device="cuda"), 0.0625, pm, pl, pacc, out)
        s = (qr @ kr[:n].float().T + ql @ c[:n].float().T) * ir[:n, grp].T * 0.0625
        ref = torch.softmax(s, -1) @ c[:n].float()
        assert ((out - ref).abs().max() / ref.abs().max()).item() < 1e-4, pos


@cuda
@pytest.mark.parametrize("r", [384, 640])
def test_fovea_codes_attn(r):
    """FoveaKV attention reading the codes vs PyTorch on the same rules: tiers (sinks + window exact, top-k by the coarse
    log-probability max over heads at 4 bits, the rest at 2 bits) over vectors decoded from the same codes."""
    import math
    from hgrt.foveakv_mla import FoveaCodec, nested_codebook, rht
    from hgrt.packed_kernels import ext
    torch.manual_seed(r)
    H, G, T, sinks, window, hot, blk = 16, 4, 1024, 4, 128, 0.1, 128
    dev = "cuda"
    xc = (torch.randn(T, r, device=dev) * 2 + 0.3).to(torch.bfloat16)
    xr = (torch.randn(T, 64, device=dev) * 3).to(torch.bfloat16)
    ir = (torch.rand(T, G, device=dev) + 0.5).half()
    cod_c = FoveaCodec(r, 128, blk, 8, True, 0, ("k", "v"))
    cod_r = FoveaCodec(64, 128, blk, 8, True, 0, ("k",))
    P = 300
    cod_c.calibrate(xc, P); cod_r.calibrate(xr, P)
    u8 = lambda *sh: torch.zeros(*sh, device=dev, dtype=torch.uint8)
    cc, gc, cr, gr = u8(T, r // 2), u8(T, r // 128), u8(T, 32), u8(T)
    Mc, Mr = torch.zeros(T // blk, r, device=dev).half(), torch.zeros(T // blk, 64, device=dev).half()
    s_all = torch.arange(0, T - window, device=dev)
    cod_c.encode_codes(lambda i: xc[i], s_all, cc, gc, Mc)
    cod_r.encode_codes(lambda i: xr[i], s_all, cr, gr, Mr)
    bnd16, lut16, lut4, Df, Dc = nested_codebook()
    ext().fovea_set_luts(lut16.to(dev, torch.float32), lut4.to(dev, torch.float32), bnd16.to(dev, torch.float32))
    R, R64 = rht(128, 0).to(dev).float().contiguous(), rht(64, 0).to(dev).float().contiguous()
    shr = [(1 - Df) ** -0.5, 1 / (1 - Df), (1 - Dc) ** -0.5, 1 / (1 - Dc), (1 - Df) ** -0.5, (1 - Dc) ** -0.5]
    RW = 256

    def decode(codes, gains, M, lo, span, hd, Rm, lut, sh, idx_fn):
        idx = torch.stack([codes & 15, codes >> 4], -1).view(T, -1).long()
        lev = lut.to(dev, torch.float32)[idx_fn(idx)].view(T, -1, hd)
        gq = torch.exp(lo + (gains.view(T, -1).float() + 0.5) / 256 * span) / math.sqrt(hd)
        Mt = M.float()[torch.arange(T, device=dev) // blk].view(T, -1, hd)
        return (Mt + (gq.unsqueeze(-1) * lev * sh) @ Rm).view(T, -1)
    dc = lambda pl, role: decode(cc, gc, Mc, cod_c.lo, cod_c.span, 128, R, lut16 if pl == "f" else lut4,
                                 cod_c.s[pl, role], (lambda i: i) if pl == "f" else (lambda i: i >> 2))
    dr = lambda pl: decode(cr, gr, Mr, cod_r.lo, cod_r.span, 64, R64, lut16 if pl == "f" else lut4,
                           cod_r.s[pl, "k"], (lambda i: i) if pl == "f" else (lambda i: i >> 2))
    Xc = {(pl, ro): dc(pl, ro) for pl in "fc" for ro in "kv"}
    Xr = {pl: dr(pl) for pl in "fc"}
    qr, ql = torch.randn(H, 64, device=dev), torch.randn(H, r, device=dev) * 0.1
    q_rr = (qr @ R64.T).contiguous()
    q_lr = (ql.view(H, -1, 128) @ R.T).reshape(H, r).contiguous()
    grp = torch.arange(H, device=dev) // (H // G)
    z = lambda n: torch.zeros(n, device=dev)
    sc, lse, lp, thr = z(H * T), z(H), z(T), z(1)
    pm, pl_, pax, par, out = z(T // 16 * H), z(T // 16 * H), z(T // 16 * H * r), z(T // 16 * H * r), torch.empty(H, r, device=dev)
    for t in (P, 600, T - 1):
        ring_c = torch.zeros(RW, r, device=dev, dtype=torch.bfloat16)
        ring_r = torch.zeros(RW, 64, device=dev, dtype=torch.bfloat16)
        idx = torch.arange(max(0, t - RW + 1), t + 1, device=dev)
        ring_c[idx % RW], ring_r[idx % RW] = xc[idx], xr[idx]
        tens = [cc, gc, cr, gr, Mc, Mr, cod_c.lo, cod_c.span, cod_r.lo, cod_r.span, xc[:sinks].contiguous(),
                xr[:sinks].contiguous(), ring_c, ring_r, ir]
        ext().fovea_codes_attn(qr, ql, q_rr, q_lr, torch.tensor([t], device=dev), 0.0625, tens, shr, [sinks, window, blk, 8],
                               hot, R, sc, lse, lp, thr, pm, pl_, pax, par, out)
        n = t + 1
        s = torch.arange(n, device=dev)
        A = (s < sinks) | (t - s < window)
        comp = ~A
        f = lambda kr, c: (qr @ kr[:n].float().T + ql @ c[:n].float().T) * ir[:n, grp].float().T * 0.0625
        sx, scf, scc_ = f(xr, xc), f(Xr["f"], Xc["f", "k"]), f(Xr["c"], Xc["c", "k"])
        lpr = (scc_ - torch.logsumexp(torch.where(comp, scc_, sx), -1, keepdim=True)).amax(0).masked_fill(~comp, float("-inf"))
        k = min(math.ceil(hot * (t + 1)), int(comp.sum()))
        hotm = (lpr >= lpr.topk(k).values[-1]) & comp if k > 0 else torch.zeros(n, dtype=torch.bool, device=dev)
        sel = (lp[:n] >= thr) & comp
        assert (sel != hotm).sum().item() <= 1, t              # one near-tie may round differently (rotated-domain dot)
        S = torch.where(A, sx, torch.where(sel, scf, scc_))
        V = torch.where(A.unsqueeze(1), xc[:n].float(), torch.where(sel.unsqueeze(1), Xc["f", "v"][:n], Xc["c", "v"][:n]))
        ref = torch.softmax(S, -1) @ V
        assert ((out - ref).abs().max() / ref.abs().max()).item() < 2e-4, t


def _rope_ref(x, cos, sin):
    h = x.shape[-1] // 2
    return torch.cat([x[..., :h], x[..., h:]], -1) * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


@cuda
@pytest.mark.parametrize("r", [384, 640])
def test_mla_prep_out(r):
    """fused MLA query / key preparation and gated output vs the PyTorch per-row ops; rows identical for any M."""
    from hgrt.packed_kernels import ext
    torch.manual_seed(1)
    dev, bf = "cuda", torch.bfloat16
    H, G, D, rd, nr, eps, Tm = 16, 4, 256, 64, 1, 1e-6, 17016
    nres, half = (G - nr) * rd, rd // 2
    nq = H * 2 * D
    y = torch.randn(4, nq + r + G * D + nr * rd, device=dev).to(bf)
    qn = torch.randn(D, device=dev) * 0.1
    qmap = (torch.eye(D, device=dev) + torch.randn(H, D, D, device=dev) * 0.05).contiguous()
    U = (torch.randn(H, G, half, device=dev) * 0.5 + 1).to(bf)
    Ures = (torch.randn(nres, r, device=dev) * 0.05).to(bf)
    Unope = (torch.randn(G, D - rd, r, device=dev) * 0.05).to(bf)
    WvT = (torch.randn(G, r, D, device=dev) * 0.05).to(bf)
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, rd, 2, device=dev).float() / rd))
    pos = torch.tensor([5, 6, 7, 8], device=dev) + 17000
    z = lambda *sh: torch.zeros(*sh, device=dev)
    E = z(0)

    def run(M, rows):
        qr, ql, olat = z(M, H, 64), z(M, H, r), torch.randn(M, H, r, device=dev)
        kr, c, ir = torch.zeros(Tm, 64, device=dev, dtype=bf), torch.zeros(Tm, r, device=dev, dtype=bf), z(Tm, G)
        yy = y[rows].contiguous()
        ext().mla_prep(yy, yy.stride(0), M, [pos[rows].contiguous(), inv_freq, qn, qmap, U, Ures, Unope, qr, ql, E, E, E, E,
                                             kr, c, ir], [H, G, D, rd, nr, r, 1, 1, 0, 0], eps)
        return qr, ql, kr, c, ir, yy

    qr4, ql4, kr4, c4, ir4, y4 = run(4, [0, 1, 2, 3])
    for m in range(4):                                   # PyTorch reference, per row
        yr = y[m]
        qg = yr[:nq].view(H, 2 * D)
        q = qg[:, :D]
        qf = q.float() * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + eps)
        q = (qf * (1.0 + qn)).to(bf)
        q = torch.einsum("hd,hde->he", q.float(), qmap).to(bf)
        ang = pos[m].float() * inv_freq
        cos, sin = torch.cat([ang, ang]).cos().to(bf), torch.cat([ang, ang]).sin().to(bf)
        qc = torch.cat([q[:, :half].unsqueeze(-2) * U, q[:, half:rd].unsqueeze(-2) * U], -1)
        q_rope = _rope_ref(qc[:, :nr], cos, sin).reshape(H, nr * rd).float()
        q_lat = qc[:, nr:].reshape(H, nres).float() @ Ures.float() \
            + torch.einsum("he,her->hr", q[:, rd:].float(), Unope.float()[torch.arange(H, device=dev) // (H // G)])
        assert (qr4[m] - q_rope).abs().max().item() <= 2 ** -7 * q_rope.abs().max().item()
        err = (ql4[m] - q_lat).abs() / q_lat.abs().max()     # fp32 order; a bf16 tie in q (q_map sum order) moves a head
        assert err.max().item() < 2e-3 and err.median().item() < 1e-6
        krms = yr[nq + r:nq + r + G * D]
        inv_r = torch.rsqrt(krms.float().view(G, D).pow(2).sum(-1) / D + eps)
        p = int(pos[m])
        assert torch.allclose(ir4[p], inv_r, rtol=1e-6)
        kro = _rope_ref(yr[nq + r + G * D:].view(nr, rd), cos, sin).reshape(-1)
        assert torch.equal(kr4[p], kro) and torch.equal(c4[p], yr[nq:nq + r])
    # bit identity across M: row 2 alone == row 2 of the 4-row launch
    qr1, ql1, *_ = run(1, [2])
    assert torch.equal(qr1[0], qr4[2]) and torch.equal(ql1[0], ql4[2])
    # output
    olat = torch.randn(4, H, r, device=dev)
    o4 = torch.zeros(4, H * D, device=dev, dtype=bf)
    ext().mla_out(olat, WvT, y4, y4.stride(0), 4, H, True, o4, o4.stride(0))
    grp = torch.arange(H, device=dev) // (H // G)
    for m in range(4):
        o = torch.einsum("hr,hrd->hd", olat[m], WvT.float()[grp]).to(bf).reshape(-1)
        gate = y[m][:nq].view(H, 2 * D)[:, D:].reshape(-1)
        ref = o * torch.sigmoid(gate)
        assert ((o4[m].float() - ref.float()).abs() > 2 ** -6 * ref.float().abs() + 1e-6).float().mean().item() < 1e-3
    o1 = torch.zeros(1, H * D, device=dev, dtype=bf)
    ext().mla_out(olat[2:3].contiguous(), WvT, y4[2:3].contiguous(), y4.stride(0), 1, H, True, o1, o1.stride(0))
    assert torch.equal(o1[0], o4[2])


@cuda
def test_fovea_encode():
    """the fused decode-time encoder reproduces FoveaCodec.encode_codes (codes, gains, block means)."""
    from hgrt.foveakv_mla import FoveaCodec, nested_codebook, rht
    from hgrt.packed_kernels import ext
    torch.manual_seed(2)
    dev, r, blk, window, T = "cuda", 512, 128, 128, 1024
    RW = 2 * blk + 4
    xc = (torch.randn(T, r, device=dev) * 2 + 0.3).to(torch.bfloat16)
    xr = (torch.randn(T, 64, device=dev) * 3).to(torch.bfloat16)
    cod_c = FoveaCodec(r, 128, blk, 8, True, 0, ("k", "v"))
    cod_r = FoveaCodec(64, 128, blk, 8, True, 0, ("k",))
    cod_c.calibrate(xc, 300); cod_r.calibrate(xr, 300)
    bnd16, lut16, lut4, *_ = nested_codebook()
    ext().fovea_set_luts(lut16.to(dev, torch.float32), lut4.to(dev, torch.float32), bnd16.to(dev, torch.float32))
    R, R64 = rht(128, 0).to(dev).float().contiguous(), rht(64, 0).to(dev).float().contiguous()
    u8 = lambda *sh: torch.zeros(*sh, device=dev, dtype=torch.uint8)
    hm = lambda *sh: torch.zeros(*sh, device=dev, dtype=torch.float16)
    ref = [u8(T, r // 2), u8(T, r // 128), hm(T // blk, r), u8(T, 32), u8(T), hm(T // blk, 64)]
    got = [u8(T, r // 2), u8(T, r // 128), hm(T // blk, r), u8(T, 32), u8(T), hm(T // blk, 64)]
    ring_c = torch.zeros(RW, r, device=dev, dtype=torch.bfloat16)
    ring_r = torch.zeros(RW, 64, device=dev, dtype=torch.bfloat16)
    for p0 in range(window, T - 4, 4):                       # 4 rows per launch, the ring holds up to p0 + 3
        idx = torch.arange(max(0, p0 + 4 - RW), p0 + 4, device=dev)
        ring_c[idx % RW], ring_r[idx % RW] = xc[idx], xr[idx]
        pos = torch.arange(p0, p0 + 4, device=dev)
        ext().fovea_encode([ring_c, R, cod_c.lo, cod_c.span, got[0], got[1], got[2]],
                           [ring_r, R64, cod_r.lo, cod_r.span, got[3], got[4], got[5]], pos, 4, [window, blk, 8])
    s = torch.arange(0, T - 4 - window, device=dev)
    cod_c.encode_codes(lambda i: xc[i], s, ref[0], ref[1], ref[2])
    cod_r.encode_codes(lambda i: xr[i], s, ref[3], ref[4], ref[5])
    n = len(s)
    for a, b in zip(got[:2] + got[3:5], ref[:2] + ref[3:5]):
        assert (a[:n] != b[:n]).float().mean().item() < 2e-3
    nb = n // blk
    for a, b in ((got[2], ref[2]), (got[5], ref[5])):
        assert ((a[:nb].float() - b[:nb].float()).abs() <= 1e-3 * b[:nb].float().abs() + 1e-4).all()


@cuda
@pytest.mark.skipif(not os.environ.get("HGRT_MERCURIUS05"), reason="set HGRT_MERCURIUS05 to a mercurius-packed-v1 dir")
def test_engine_graph_and_speculative_lossless():
    """eager == graph, and MTP speculative decoding (k = 1..3) emits exactly the plain greedy tokens."""
    from hgrt.mercurius_packed import PackedMercuriusEngine
    e = PackedMercuriusEngine(os.environ["HGRT_MERCURIUS05"], Tmax=512)
    prompt = [9707, 11, 1246, 525, 498, 30, 3555, 374, 279, 6722, 315, 9625, 30]
    eager, _ = e.generate(prompt, 48)
    e.capture(ks=(0, 1, 2, 3))
    plain, _ = e.generate(prompt, 48)
    assert plain == eager
    for k in (1, 2, 3):
        spec, st = e.generate(prompt, 48, k=k)
        assert spec == plain, k
        assert st["blocks"] < 48
