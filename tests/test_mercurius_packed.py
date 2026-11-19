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
def test_packed_gemv(fmt):
    from hgrt.packed_kernels import PackedLin
    g = torch.Generator().manual_seed(0)
    N, K = 1000, 2560                                        # N not a multiple of the rows per block
    lin = PackedLin([(_q4 if fmt == "q4_0" else _q8)(N, K, g)], fmt)
    W = lin.dequant().float()
    x = torch.randn(4, K, generator=g).to("cuda", torch.bfloat16)
    ref = x.float() @ W.T
    for R in (1, 2, 4):
        lin.R = R
        full = lin(x, 4, torch.empty(4, N, device="cuda"))
        assert ((full - ref).abs().max() / ref.abs().max()).item() < 1e-5
        for M in (1, 2, 3):
            assert torch.equal(lin(x, M, torch.empty(M, N, device="cuda")), full[:M])


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
@pytest.mark.skipif(not os.environ.get("HGRT_MERCURIUS05"), reason="set HGRT_MERCURIUS05 to a mercurius-packed-v1 dir")
def test_engine_graph_matches_eager():
    from hgrt.mercurius_packed import PackedMercuriusEngine
    e = PackedMercuriusEngine(os.environ["HGRT_MERCURIUS05"], Tmax=256)
    prompt = [9707, 11, 1246, 525, 498, 30]
    eager = e.generate(prompt, 12)
    e.capture()
    assert e.generate(prompt, 12) == eager
