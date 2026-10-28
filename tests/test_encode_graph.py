"""Graph-replayed decode-time block encode (StackCache.pre_step) == the eager encode, bit for bit, over a long
generation (several blocks), and the attention outputs agree. CUDA only."""
import pytest
import torch

pytestmark = pytest.mark.gpu
cuda = pytest.importorskip("torch").cuda.is_available()


@pytest.mark.skipif(not cuda, reason="needs CUDA")
def test_graph_encode_matches_eager():
    from hgrt.stack import BLK, D, G, StackCache, fg
    torch.manual_seed(0)
    Hkv, P, steps = 2, 3 * BLK + 40, 5 * BLK + 17                      # five decode-time encodes
    k = (torch.randn(P + steps, Hkv, D, device="cuda") * 0.8).bfloat16()
    v = torch.randn(P + steps, Hkv, D, device="cuda").bfloat16()
    qs = torch.randn(steps, Hkv * G, D, device="cuda")
    caches = []
    for graph in (False, True):
        c = StackCache(Hkv, P + steps + 2 * BLK)
        c.encode_graph = graph
        c.prefill(k[:P], v[:P])
        c.graph_setup()
        caches.append(c)
    outs = [[], []]
    for i in range(steps):
        pos = torch.full((1,), P + i, dtype=torch.long, device="cuda")
        for j, c in enumerate(caches):
            c.pre_step()
            outs[j].append(c.graph_attend(qs[i], k[P + i][None], v[P + i][None], pos).clone())
    a, b = caches
    assert a.n_comp == b.n_comp and a.n_comp >= P + 4 * BLK - fg.W
    assert int(b.npg_dev) == b.n_comp // BLK and int(b.ncomp_dev) == b.n_comp
    for name in ("khi", "klo", "vhi", "vlo", "kg8", "vg8", "kmu", "vmu", "lamres", "kmx", "kmn", "kraw", "vraw"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert torch.allclose(a.U.float(), b.U.float(), atol=1e-3)          # same summarize() arithmetic, same inputs
    assert torch.equal(torch.stack(outs[0]), torch.stack(outs[1]))
