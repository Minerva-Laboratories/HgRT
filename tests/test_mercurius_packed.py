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
    _, lut16, lut4, Df, Dc = nested_codebook()
    ext().fovea_set_luts(lut16.to(dev, torch.float32), lut4.to(dev, torch.float32))
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
