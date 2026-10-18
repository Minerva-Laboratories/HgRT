"""The stack cache: the PyTorch reference on CPU against attention over FoveaKV's own dequantized pages, and the fused
kernel path against the reference (marked gpu)."""
import pytest
import torch

from cairn.stack import BLK, D, StackCache, make_stack

H = 2


def _kv(T, dev="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)
    nt = -(-T // 512)
    mu = torch.randn(nt, 1, H, D, generator=g) * 0.8
    K = mu.expand(nt, 512, H, D).reshape(-1, H, D)[:T] + 0.6 * torch.randn(T, H, D, generator=g)
    return K.to(dev), torch.randn(T, H, D, generator=g).to(dev), (1.5 * torch.randn(H * 4, D, generator=g)).to(dev)


def _dense_over(c, q, K, V):
    qh = (q.float() * c.scale).view(H, 4, D)
    return torch.cat([torch.softmax(qh[h] @ K[:, h].T, -1) @ V[:, h] for h in range(H)])


def test_all_pages_open_equals_attention_over_the_4bit_cache():
    K, V, q = _kv(9 * BLK + 60)
    c = StackCache(H, len(K) + 256, device="cpu", rho=1.0)
    c.use_kernel = False
    c.prefill(K, V)
    Kd, Vd = c.dequant()
    kt, vt = c.tail()
    ref = _dense_over(c, q, torch.cat([Kd, kt.float()]), torch.cat([Vd, vt.float()]))
    assert c.n_comp == 8 * BLK and torch.allclose(c.attend(q), ref, atol=1e-5)


def test_selection_opens_sink_and_top_pages():
    K, V, q = _kv(41 * BLK)
    c = make_stack("stack10", H, len(K) + 256, device="cpu")
    c.use_kernel = False
    c.prefill(K, V)
    c.attend(q)
    sel = c.last_sel
    assert sel.shape == (H, c.n_open()) and bool((sel == 0).any(-1).all())          # page 0 always opens
    assert 0 < c.read_bits() < 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("name,T", [("stack10", 64 * BLK + 33), ("stack20-bound", 30 * BLK), ("stack5-mean", 100 * BLK)])
def test_kernel_matches_reference(name, T):
    K, V, q = _kv(T, dev="cuda")
    c = make_stack(name, H, T + 256)
    c.prefill(K, V)
    ref = c.attend_ref(q)
    out = c.attend_kernel(q)
    torch.cuda.synchronize()
    assert float((out - ref).norm() / ref.norm()) < 1e-4
    m = c.n_open()
    for h in range(H):
        assert set(c._sel[h, :m].tolist()) == set(c.last_sel[h].tolist())
