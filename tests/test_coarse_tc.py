"""The tensor-core coarse pass (csrc/coarse_tc.cu) against FoveaKV's fp32 pass: same partials, same logits."""
import pytest
import torch

from hgrt.dyn import BLK, D, G, coarse_pass, fg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cache(P, Hkv=4, block_mean=True, seed=0):
    torch.manual_seed(seed)
    c = fg.FoveaKV(Hkv, P + 512, "cuda", k_frac=0.1, block_mean=block_mean)
    k = (torch.randn(P, Hkv, D, device="cuda") * .8 + .3).bfloat16()
    v = (torch.randn(P, Hkv, D, device="cuda") + .2).bfloat16()
    c.prefill(k, v)
    q = torch.randn(Hkv * G, D, device="cuda") * 2
    qk = ((q * c.scale).view(Hkv, G, D) @ c.Rt).contiguous()
    return c, qk


def _run(c, qk, n, tpb, tc, pages=None, n_dev=None):
    nb = -(-n // tpb)
    part = c._partials(nb)
    logits = torch.full_like(c.logits, float("nan"))
    if tc:
        coarse_pass(c, qk, n, tpb, part, logits=logits, n_dev=n_dev, pages=pages)
    else:
        fg.run_pass(1, qk, c.Hkv, c.cap, 0, n, None, tpb, 0, part, logits=logits, n_dev=n_dev, pages=pages,
                    **c._gs_args(0, 0, c.k4, c.v4))
    o, m, l, a = fg.ext().combine_partials(*part, nb, True)
    return o, m, l, logits


@pytest.mark.parametrize("block_mean", [True, False])
@pytest.mark.parametrize("n_live", [4096, 4096 - 128 - 16 * 3])
def test_coarse_tc_matches_fp32_pass(block_mean, n_live):
    c, qk = _cache(4096, block_mean=block_mean)
    n_dev = torch.tensor([n_live], dtype=torch.int32, device="cuda")
    tpb = fg.tpb_for(4096, c.Hkv)
    ref, mr, lr, lgr = _run(c, qk, 4096, tpb, False, n_dev=n_dev)
    out, m, l, lg = _run(c, qk, 4096, tpb, True, n_dev=n_dev)
    assert torch.allclose(m, mr, atol=1e-4, rtol=0)
    assert ((l - lr).abs() / lr).max() < 1e-4
    rel = (out - ref).norm(dim=-1) / ref.norm(dim=-1)
    assert rel.max() < 1e-4, rel.max()
    lg, lgr = lg[..., :n_live], lgr[..., :n_live]
    assert (lg - lgr).abs().max() < 1e-4 * lgr.abs().max()


def test_coarse_tc_pages():
    c, qk = _cache(8192)
    pages = torch.stack([torch.randperm(8192 // BLK, device="cuda")[:12] for _ in range(c.Hkv)]).int().contiguous()
    n = 12 * BLK
    ref, *_ = _run(c, qk, n, 256, False, pages=pages)
    out, *_ = _run(c, qk, n, 256, True, pages=pages)
    assert ((out - ref).norm(dim=-1) / ref.norm(dim=-1)).max() < 1e-4


def test_delta_tc_matches_foveakv_delta():
    from hgrt.dyn import delta_pass
    c, qk = _cache(4096)
    nb = -(-4096 // 512)
    part = c._partials(nb)
    qoff = torch.zeros(c.Hkv, c.cap // BLK, G, device="cuda")
    coarse_pass(c, qk, 4096, 512, part, logits=c.logits, qoff=qoff)
    _, m1, l1, _ = fg.ext().combine_partials(*part, nb, False)
    m1 = m1.view(c.Hkv, G).contiguous()
    hot = torch.stack([torch.randperm(4096, device="cuda")[:410] for _ in range(c.Hkv)]).int()
    hot[:, -7:] = -1
    hot = hot.contiguous()
    nbd = -(-410 // 64)
    pr, pt = c._partials(nbd), c._partials(nbd)
    fg.ext().delta(qk, c.Hkv, c.cap, hot, 64, m1, pr[1], pr[2], c.khi, c.klo, c.vhi, c.vlo, c.kg8, c.vg8,
                   c.glut["k"], c.glut["v"], c.kmu, c.vmu, BLK, c.k4, c.k16, c.v4, c.v16)
    delta_pass(c, qk, hot, 64, m1, qoff, c.logits, pt)
    lr, lt = pr[1].view(nbd, -1).sum(0), pt[1].view(nbd, -1).sum(0)
    ar, at = pr[2].view(nbd, -1, D).sum(0), pt[2].view(nbd, -1, D).sum(0)
    assert ((lt - lr).abs() / l1).max() < 1e-5            # dl is a difference: compare to the normalizer it updates
    assert ((at - ar).norm(dim=-1) / ar.norm(dim=-1)).max() < 1e-4


def test_delta_on_tensor_cores_matches_foveakv_delta():
    from hgrt.dyn import delta_pass_tc
    c, qk = _cache(4096)
    nb = -(-4096 // 512)
    part = c._partials(nb)
    qoff = torch.zeros(c.Hkv, c.cap // BLK, G, device="cuda")
    coarse_pass(c, qk, 4096, 512, part, logits=c.logits, qoff=qoff)
    _, m1, l1, _ = fg.ext().combine_partials(*part, nb, False)
    m1 = m1.view(c.Hkv, G).contiguous()
    hot = torch.stack([torch.randperm(4096, device="cuda")[:410] for _ in range(c.Hkv)]).int()
    hot[:, -7:] = -1
    hot = hot.contiguous()
    for tpb in (64, 128, 256):
        nbd = -(-410 // tpb)
        pr, pt = c._partials(nbd), c._partials(nbd)
        fg.ext().delta(qk, c.Hkv, c.cap, hot, tpb, m1, pr[1], pr[2], c.khi, c.klo, c.vhi, c.vlo, c.kg8, c.vg8,
                       c.glut["k"], c.glut["v"], c.kmu, c.vmu, BLK, c.k4, c.k16, c.v4, c.v16)
        wsum = torch.zeros_like(qoff)
        delta_pass_tc(c, qk, hot, tpb, m1, qoff, c.logits, wsum, 4096 // BLK, pt)
        assert wsum.abs().max() == 0
        lr, lt = pr[1].view(nbd, -1).sum(0), pt[1].view(nbd, -1).sum(0)
        ar, at = pr[2].view(nbd, -1, D).sum(0), pt[2].view(nbd, -1, D).sum(0)
        assert ((lt - lr).abs() / l1).max() < 1e-5, tpb
        assert ((at - ar).norm(dim=-1) / ar.norm(dim=-1)).max() < 1e-4, tpb
