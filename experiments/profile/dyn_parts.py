"""Per-part GPU time of one DynGraph.graph_attend call (one layer) at a given compressed length (CUDA events)."""
import math, os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from hgrt.dyn import DynGraph, fg, G, D
HKV, P = 4, int(os.environ.get("P", 32768))
c = DynGraph(HKV, P + 2048, k_frac=0.1)
k = (torch.randn(P, HKV, D, device="cuda") * .8).bfloat16(); v = torch.randn(P, HKV, D, device="cuda").bfloat16()
c.prefill(k, v); c.graph_setup(); c.pre_step()
q = torch.randn(HKV * G, D, device="cuda"); pos = torch.full((1,), P, dtype=torch.long, device="cuda")
e = fg.ext(); b = c.bucket
qs = (q * c.scale).view(HKV, G, D).contiguous(); qk = (qs @ c.Rt).contiguous()
def t(f, n=30):
    for _ in range(3): f()
    torch.cuda.synchronize(); out = []
    for _ in range(n):
        s_, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s_.record(); f(); e_.record(); torch.cuda.synchronize(); out.append(s_.elapsed_time(e_))
    return round(float(np.median(out)), 3)
coarse = lambda: fg.run_pass(1, qk, HKV, c.cap, 0, b, None, c.tpb_g, 0, c.p1_g, logits=c.logits, n_dev=c.ncomp_dev, **c._gs_args(0, 0, c.k4, c.v4))
print("bucket", b, "k_cap", c.k_cap, "n_comp", c.n_comp)
print("coarse pass", t(coarse))
print("combine", t(lambda: e.combine_partials(*c.p1_g, c.nb_g, False)))
lse = torch.zeros(HKV * G, device="cuda")
def sel_f():
    s = (c.logits[:, :, :b] - lse.view(HKV, G, 1)).amax(1)
    return s.masked_fill(c.ar[:b] >= c.ncomp_dev.long(), -math.inf)
print("sel (amax+mask)", t(sel_f))
s = sel_f()
print("topk k_cap sorted", t(lambda: torch.topk(s, c.k_cap, dim=-1, sorted=True)))
print("topk k_cap unsorted", t(lambda: torch.topk(s, c.k_cap, dim=-1, sorted=False)))
kk = math.ceil(0.1 * P)
print("topk k_live", kk, t(lambda: torch.topk(s[:, :P], kk, dim=-1, sorted=False)))
hot = torch.topk(s, c.k_cap, dim=-1).indices.int().contiguous()
_, pl_d, pa_d = c.pd_g
print("delta", t(lambda: e.delta(qk, HKV, c.cap, hot, c.tpb_d, torch.zeros(HKV, G, device="cuda"), pl_d, pa_d, c.khi, c.klo, c.vhi, c.vlo, c.kg8, c.vg8, c.glut["k"], c.glut["v"], c.kmu, c.vmu, 128, c.k4, c.k16, c.v4, c.v16)))
print("whole graph_attend (eager)", t(lambda: c.graph_attend(q, k[:1], v[:1], pos)))
