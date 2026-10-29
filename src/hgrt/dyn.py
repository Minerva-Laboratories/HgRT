"""FoveaKV dynamic 2+2 decode inside a CUDA graph (the eager FoveaKV.attend() is ~30 launches and host-side sizes per
call, ~2.5 ms per layer on the Orin, independent of context: the long-response bottleneck).

Same method as foveakv_gpu.FoveaKV.attend (dyn<k>, block means, shrinkage): one coarse 2-bit pass over the compressed
tokens writes per-token logits and softmax partials; the hot set is the exact top-k tokens per KV head by the estimated
log-probability max over the GQA group of (logit - lse), lse over coarse + the exact bf16 window, k = ceil(k_frac * n);
a delta pass replaces those tokens' coarse contribution by the 4-bit one; the bf16 window is merged last.
What changes for the graph: every size is a capacity, live counts are read on the GPU.
  - the coarse pass and the window pass read their token counts from device scalars (ncomp_dev, pos);
  - selection = exact top-k by a fused score + 3-level radix select kernel (csrc/dyn_select.cu: the live k and n read
    on the GPU, hot[h, k:] = -1, which the delta kernel skips); HGRT_DYN_SELECT=torch uses torch.topk on a fixed
    k_cap instead (the reference for the test);
  - capacities are bucketed (powers of two of the compressed length): a step is captured per bucket, so the coarse
    grid and the top-k work track the live length instead of Tmax; `recapture` tells the caller when the bucket grows.
Decode-time block encodes reuse FoveaKV.encode_block_dev in a captured graph, as hgrt.stack does.
Improvements over the llama-mercurius split-attention version (histogram threshold, two coarse sweeps): exact top-k,
one coarse sweep, the exact bf16 window, FoveaKV's gain / block-mean / rotated codes."""
import math
import os

import torch

from .deps import foveakv_gpu

fg = foveakv_gpu()
BLK, D, G, W = fg.BLK, fg.D, fg.G, fg.W
MIN_BUCKET = 4096
_SEL = None


def sel_ext():
    """csrc/dyn_select.cu (fused score + exact radix top-k with the count on the GPU), JIT-built once."""
    global _SEL
    if _SEL is None:
        from torch.utils.cpp_extension import load
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        bd = os.path.join(os.environ.get("HGRT_BUILD_DIR", os.path.expanduser("~/.cache/hgrt")), "dyn_select")
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "dyn_select.cu")
        _SEL = load("hgrt_dyn_select", [src], build_directory=bd, verbose=False,
                    extra_cuda_cflags=["-O3", "-allow-unsupported-compiler"])
    return _SEL


_TC = None


def tc_ext():
    """csrc/dyn_kernels.cu (coarse pass on the tensor cores, delta pass, step glue), JIT-built once for this head size."""
    global _TC
    if _TC is None:
        from torch.utils.cpp_extension import load
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        bd = os.path.join(os.environ.get("HGRT_BUILD_DIR", os.path.expanduser("~/.cache/hgrt")), f"dyn_kernels_d{D}")
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "dyn_kernels.cu")
        _TC = load(f"hgrt_dyn_kernels_d{D}", [src], build_directory=bd, verbose=False,
                   extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler", f"-DTC_D={D}"])
    return _TC


def coarse_pass(c, qk, n, tpb, part, logits=None, n_dev=None, pages=None, t0=0, boff=0, qoff=None):
    """the 2-bit gain-shape pass of FoveaKV cache `c` (kfine = vfine = 0) on the tensor cores; same partials as
    fg.run_pass(1, ...). Needs symmetric level tables ({-b, -a, a, b}, true for FoveaKV's codebook). `qoff`
    (Hkv, n_mean_blocks, G) receives q . kmu per mean block (for delta_pass)."""
    tc_ext().coarse(qk, c.Hkv, c.cap, t0, n, tpb, boff, *part, c.khi, c.vhi, c.kg8, c.vg8, c.glut["k"], c.glut["v"],
                    c.kmu, c.vmu, _mu_block(c), c.k4, c.v4, logits, pages, n_dev, qoff)


def _mu_block(c):
    return BLK if c.block_mean else c.cap


def delta_pass_tc(c, qk, hot, tpb, mref, qoff, logits, wsum, n_live_blocks, part):
    """the delta on the tensor cores: same partials as delta_pass (the values' block-mean term is accumulated in
    `wsum` (zero on entry, re-zeroed) and added to the first partial). tpb: multiple of 64."""
    _, pl, pacc = part
    tc_ext().delta_tc(qk, c.Hkv, c.cap, hot, tpb, mref, qoff, logits, wsum, pl, pacc, c.khi, c.klo, c.vhi, c.vlo,
                      c.kg8, c.vg8, c.glut["k"], c.glut["v"], c.vmu, _mu_block(c), n_live_blocks, c.k16, c.v4, c.v16)


def delta_pass(c, qk, hot, tpb, mref, qoff, logits, part):
    """FoveaKV's dyn delta over the hot list (-1 = skip) -> (pl, pacc) partials, like fg.ext().delta; `qoff` and
    `logits` (the coarse logits of every token) come from the coarse_pass that produced mref."""
    _, pl, pacc = part
    tc_ext().delta(qk, c.Hkv, c.cap, hot, tpb, mref, qoff, logits, pl, pacc, c.khi, c.klo, c.vhi, c.vlo, c.kg8, c.vg8,
                   c.glut["k"], c.glut["v"], c.vmu, _mu_block(c), c.k16, c.v4, c.v16)


class DynGraph(fg.FoveaKV):
    def __init__(self, Hkv, Tmax, device="cuda", k_frac=0.1, **kw):
        super().__init__(Hkv, Tmax, device, k_frac=k_frac, **kw)
        self.name = f"FoveaKV dyn{k_frac * 100:g} (graph)"
        self.encode_graph = True
        self.select = os.environ.get("HGRT_DYN_SELECT", "radix")      # radix (fused kernel) | torch (topk reference)
        self.kernels = os.environ.get("HGRT_DYN_KERNELS", "tc")       # tc (csrc/coarse_tc.cu) | ref (foveakv kernels)

    # ------------------------------------------------------------------ graph mode (hgrt.engine protocol)
    def graph_setup(self, ns_win=4):
        i32 = lambda: torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.npg_dev, self.ncomp_dev = i32(), i32()
        self.ns_win = ns_win
        self.p2_g = self._partials(ns_win)
        self.out_g = torch.empty(self.Hkv * G, D, device=self.dev)
        self.ar = torch.arange(self.cap, device=self.dev)
        self.qoff_g = torch.zeros(self.Hkv, self.cap // _mu_block(self), G, device=self.dev)
        self.wsum_g = torch.zeros_like(self.qoff_g)
        HG = self.Hkv * G
        f = lambda *sh: torch.zeros(*sh, device=self.dev)
        self.qk_g, self.m1_g, self.l1_g, self.acc1_g = f(HG, D), f(HG), f(HG), f(HG, D)
        self.mw_g, self.lw_g, self.aw_g, self.lse_g = f(HG), f(HG), f(HG, D), f(HG)
        self.Rt_c, self.R_c = self.Rt.float().contiguous(), self.R.float().contiguous()
        self.bucket = None
        self.recapture = False
        self._sync_dev()
        self._set_bucket()

    def _sync_dev(self):
        self.npg_dev.fill_(self.n_comp // BLK)
        self.ncomp_dev.fill_(self.n_comp)

    def _bucket_for(self, n_comp):
        b = MIN_BUCKET
        while b < n_comp:
            b *= 2
        return min(b, self.cap)

    def _set_bucket(self):
        b = self._bucket_for(self.n_comp + BLK)          # room for the next block encode
        if b == self.bucket:
            return
        self.bucket = b
        self.k_cap = min(b, math.ceil(self.k_frac * (b + W + BLK)) + 1)
        self.tpb_g = fg.tpb_for(b, self.Hkv)
        self.nb_g = -(-b // self.tpb_g)
        self.p1_g = self._partials(self.nb_g)
        self.tpb_d = 64
        self.nbd_g = -(-self.k_cap // self.tpb_d)
        self.pd_g = self._partials(self.nbd_g)
        self.tpb_dt = int(os.environ.get("HGRT_DELTA_TPB", 128))         # hot entries per block, tensor-core delta
        self.nbdt_g = -(-self.k_cap // self.tpb_dt)
        self.pdt_g = self._partials(self.nbdt_g)
        z = lambda *sh, dt=torch.int32: torch.zeros(*sh, dtype=dt, device=self.dev)
        self._sel_buf = torch.empty(self.Hkv, b, device=self.dev)
        self._h1, self._h2, self._h3 = z(self.Hkv, 4096), z(self.Hkv, 4096), z(self.Hkv, 256)
        self._state = z(self.Hkv, 8)
        self._hot = torch.full((self.Hkv, self.k_cap), -1, dtype=torch.int32, device=self.dev)
        self.recapture = True

    def pre_step(self):
        """before a graph step that appends one token (as hgrt.stack.StackCache.pre_step)."""
        target = max(0, ((self.n + 1 - W) // BLK) * BLK)
        if target > self.n_comp:
            nb, rem = target - self.n_comp, self.n - target
            if nb == BLK and rem == W - 1 and self.encode_graph:
                if getattr(self, "_enc_graph", None) is None:
                    self._capture_encode(rem)
                self._enc_graph.replay()
                self.n_comp = target
            else:
                self._encode(self.n_comp, target, self.kraw[:nb], self.vraw[:nb])
                self.kraw[:rem] = self.kraw[nb:nb + rem].clone()
                self.vraw[:rem] = self.vraw[nb:nb + rem].clone()
                self.n_comp = target
                self._sync_dev()
            self._set_bucket()
        self.n += 1

    def _encode_block_dev(self, page):
        self.encode_block_dev(self.kraw[:BLK], self.vraw[:BLK], page)
        r = self._enc_rem
        self._shift_k.copy_(self.kraw[BLK:BLK + r]); self.kraw[:r].copy_(self._shift_k)
        self._shift_v.copy_(self.vraw[BLK:BLK + r]); self.vraw[:r].copy_(self._shift_v)
        self.npg_dev += 1
        self.ncomp_dev += BLK

    def _capture_encode(self, rem):
        self._enc_rem = rem
        self._shift_k = torch.empty(rem, self.Hkv, D, dtype=self.kraw.dtype, device=self.dev)
        self._shift_v = torch.empty_like(self._shift_k)
        self._enc_page = torch.zeros(1, dtype=torch.long, device=self.dev)
        saved = [t.clone() for t in (self.kraw, self.vraw, self.npg_dev, self.ncomp_dev)]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self._enc_page.copy_(self.npg_dev.long())
            self._encode_block_dev(self._enc_page)
        torch.cuda.current_stream().wait_stream(s)
        for t, v in zip((self.kraw, self.vraw, self.npg_dev, self.ncomp_dev), saved):
            t.copy_(v)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._enc_page.copy_(self.npg_dev.long())
            self._encode_block_dev(self._enc_page)
        self._enc_graph = g

    def graph_attend(self, q, k, v, pos):
        """inside a CUDA graph: append k, v (1, Hkv, D) at position pos (1,) long, attend q (Hq, D) -> (Hq, D) fp32."""
        if self.kernels == "tc":
            return self._attend_tc(q, k, v, pos)
        # reference path: FoveaKV's kernels + torch glue
        idx = pos - self.ncomp_dev.long()
        self.kraw.index_copy_(0, idx, k.to(torch.bfloat16))
        self.vraw.index_copy_(0, idx, v.to(torch.bfloat16))
        e = fg.ext()
        b = self.bucket
        qs = (q.float() * self.scale).view(self.Hkv, G, D).contiguous()
        qk = (qs @ self.Rt).contiguous()
        # coarse 2-bit pass over the compressed tokens (count on the GPU) -> per-token logits + partials
        fg.run_pass(1, qk, self.Hkv, self.cap, 0, b, None, self.tpb_g, 0, self.p1_g, logits=self.logits,
                    n_dev=self.ncomp_dev, **self._gs_args(0, 0, self.k4, self.v4))
        _, m1, l1, acc1 = e.combine_partials(*self.p1_g, self.nb_g, False)            # (Hkv*G,), (Hkv*G, D)
        # exact bf16 window partials (length pos + 1 - ncomp), original domain
        from .kernels import ops
        ops().dense_attn(q, self.kraw, self.vraw, pos, self.ncomp_dev, *self.p2_g, None, True, self.scale)
        pm2, pl2, pa2 = self.p2_g
        pm2, pl2, pa2 = pm2.view(self.ns_win, -1), pl2.view(self.ns_win, -1), pa2.view(self.ns_win, -1, D)
        mw = pm2.amax(0)
        fw = torch.where(pm2 > -math.inf, torch.exp(pm2 - mw), torch.zeros_like(pm2))
        lw, aw = (pl2 * fw).sum(0), (pa2 * fw[..., None]).sum(0)
        # selection: lse over coarse + window, estimated log-prob max over the group, exact top-k_cap, live k
        M = torch.maximum(m1, mw)
        lse = M + torch.log(l1 * torch.exp(m1 - M) + lw * torch.exp(mw - M))
        if self.select == "radix":
            sel_ext().select_topk(self.logits, lse.contiguous(), b, self.ncomp_dev, pos, float(self.k_frac),
                                  self._sel_buf, self._h1, self._h2, self._h3, self._state, self._hot)
            hot = self._hot
        else:                                                                          # reference: torch.topk
            sel = (self.logits[:, :, :b] - lse.view(self.Hkv, G, 1)).amax(1)           # (Hkv, b)
            sel = sel.masked_fill(self.ar[:b] >= self.ncomp_dev.long(), -math.inf)
            top = torch.topk(sel, self.k_cap, dim=-1, sorted=True).indices.int()
            k_live = torch.minimum(torch.ceil(self.k_frac * (pos.float() + 1)), self.ncomp_dev.float()).long()
            hot = torch.where(torch.arange(self.k_cap, device=self.dev)[None] < k_live, top, torch.full_like(top, -1))
        # delta pass: hot tokens' coarse -> 4-bit contribution, relative to m1
        _, pl_d, pa_d = self.pd_g
        e.delta(qk, self.Hkv, self.cap, hot.contiguous(), self.tpb_d, m1.view(self.Hkv, G).contiguous(), pl_d, pa_d,
                self.khi, self.klo, self.vhi, self.vlo, self.kg8, self.vg8, self.glut["k"], self.glut["v"],
                self.kmu, self.vmu, BLK, self.k4, self.k16, self.v4, self.v16)
        l1 = l1 + pl_d.view(self.nbd_g, -1).sum(0)
        acc1 = acc1 + pa_d.view(self.nbd_g, -1, D).sum(0)
        # merge with the window (coarse accumulators are in the rotated domain)
        acc1 = acc1 @ self.R
        fa, fb = torch.exp(m1 - M), torch.exp(mw - M)
        self.out_g.copy_((acc1 * fa[:, None] + aw * fb[:, None]) / (l1 * fa + lw * fb)[:, None])
        self.last_hot, self.last_lse = hot, lse
        return self.out_g


    def _attend_tc(self, q, k, v, pos):
        """graph_attend on csrc/dyn_kernels.cu: prologue -> coarse (tensor cores) -> window -> merge1 -> select ->
        delta (tensor cores) + its block-mean term -> merge2."""
        t, b = tc_ext(), self.bucket
        q = q.float().contiguous()
        t.prologue(q, k.float().reshape(self.Hkv, D).contiguous(), v.float().reshape(self.Hkv, D).contiguous(), pos,
                   self.ncomp_dev, self.kraw, self.vraw, self.Rt_c, float(self.scale), self.qk_g)
        qk = self.qk_g.view(self.Hkv, G, D)
        coarse_pass(self, qk, b, self.tpb_g, self.p1_g, logits=self.logits, n_dev=self.ncomp_dev, qoff=self.qoff_g)
        from .kernels import ops
        ops().dense_attn(q, self.kraw, self.vraw, pos, self.ncomp_dev, *self.p2_g, None, True, self.scale)
        t.merge1(*self.p1_g, self.nb_g, *self.p2_g, self.m1_g, self.l1_g, self.acc1_g, self.mw_g, self.lw_g,
                 self.aw_g, self.lse_g)
        sel_ext().select_topk(self.logits, self.lse_g, b, self.ncomp_dev, pos, float(self.k_frac),
                              self._sel_buf, self._h1, self._h2, self._h3, self._state, self._hot)
        m1 = self.m1_g.view(self.Hkv, G)
        delta_pass_tc(self, qk, self._hot, self.tpb_dt, m1, self.qoff_g, self.logits, self.wsum_g,
                      -(-b // _mu_block(self)), self.pdt_g)
        _, pl_d, pa_d = self.pdt_g
        t.merge2(pl_d, pa_d, self.nbdt_g, self.m1_g, self.l1_g, self.acc1_g, self.mw_g, self.lw_g, self.aw_g,
                 self.R_c, self.out_g)
        self.last_hot, self.last_lse = self._hot, self.lse_g
        return self.out_g


def make_dyn(name, Hkv, Tmax, device="cuda"):
    """dyn<k>-graph, e.g. dyn10-graph."""
    k = int(name[3:].split("-")[0]) / 100
    return DynGraph(Hkv, Tmax, device, k_frac=k)
