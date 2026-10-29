"""FoveaKV dynamic 2+2 in a CUDA graph (hgrt.dyn.DynGraph) == the eager reference FoveaKV.attend(): same hot sets
(exact top-k) and outputs within float tolerance, over a generation that crosses block encodes and a bucket change.
With the reference kernels the hot sets are identical; the tensor-core coarse pass sums in another order (logits
differ by ~1e-6), so a near-tie token at the top-k boundary can swap: every swap must be a tie under the graph's own
scores (gap < 1e-4), swaps stay below 1e-4 of the selections, and steps without a swap keep the 1e-4 output bound
(a swapped boundary token moves the output by the method's own 2-bit vs 4-bit difference for that token)."""
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("select,kernels", [("radix", "tc"), ("torch", "tc"), ("radix", "ref")])
def test_dyn_graph_matches_eager_reference(select, kernels):
    from hgrt import dyn
    fg = dyn.fg
    torch.manual_seed(0)
    Hkv, P, steps = 2, 3900, 700                              # crosses the 4096 bucket and several block encodes
    k = (torch.randn(P + steps, Hkv, fg.D, device="cuda") * 0.8).bfloat16()
    v = torch.randn(P + steps, Hkv, fg.D, device="cuda").bfloat16()
    k[1000:1040] += 2.0                                       # a hot region
    qs = torch.randn(steps, Hkv * fg.G, fg.D, device="cuda") * 1.5
    ref = fg.FoveaKV(Hkv, P + steps + 512, k_frac=0.1)
    ref.prefill(k[:P], v[:P])
    gc = dyn.DynGraph(Hkv, P + steps + 512, k_frac=0.1)
    gc.select, gc.kernels = select, kernels
    gc.prefill(k[:P], v[:P])
    gc.graph_setup()
    q_in, k_in, v_in = torch.empty_like(qs[0]), torch.empty(1, Hkv, fg.D, device="cuda"), torch.empty(1, Hkv, fg.D, device="cuda")
    pos = torch.zeros(1, dtype=torch.long, device="cuda")
    graph = None
    worst, set_mismatch, n_sel, buckets = 0.0, 0, 0, set()
    for i in range(steps):
        ref.append(k[P + i][None], v[P + i][None])
        o_ref = ref.attend(qs[i])
        gc.pre_step()
        q_in.copy_(qs[i]); k_in.copy_(k[P + i][None]); v_in.copy_(v[P + i][None]); pos.fill_(P + i)
        if graph is None or gc.recapture:
            gc.recapture = False
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gc.graph_attend(q_in, k_in, v_in, pos)
            buckets.add(gc.bucket)
        graph.replay()
        o = gc.out_g
        err = float((o - o_ref).abs().max() / o_ref.abs().max())
        hot_ref, swapped = ref.last_hot, 0
        if hot_ref is not None:
            h = gc.last_hot
            n = gc.n_comp
            sc = (gc.logits[:, :, :n] - gc.last_lse.view(Hkv, fg.G, 1)).amax(1)       # the graph's selection scores
            for hh in range(Hkv):
                a = set(hot_ref[hh].tolist()); b = set(x for x in h[hh].tolist() if x >= 0)
                if a != b:
                    only_ref, only_g = sorted(a - b), sorted(b - a)
                    gap = float(sc[hh, only_g].max() - sc[hh, only_ref].min())
                    assert 0 <= gap < 1e-4, (i, gap)
                swapped += len(a ^ b) // 2
                n_sel += len(a)
        set_mismatch += swapped
        if not swapped:
            worst = max(worst, err)
    assert gc.n_comp == ref.n_comp and len(buckets) >= 2
    assert set_mismatch <= (0 if kernels == "ref" else 1e-4 * n_sel), (set_mismatch, n_sel)
    assert worst < 1e-4, worst
