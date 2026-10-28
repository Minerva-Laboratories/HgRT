"""The decode stack: FoveaKV 2+2-bit gain-shape pages, Riffle page selection, bf16 recent window.

Storage per attention layer (FoveaKV `GainShape`, mode gs, 4 bits = 2-bit coarse + 2-bit residual planes, 8-bit gain,
128-token block means, all in FoveaKV's rotated domain) plus Riffle's per-page key summaries computed from the raw keys
when a page is encoded: covariance directions rotated into FoveaKV's domain (the mean is FoveaKV's own block mean),
channel max/min in the original domain (the Quest bound needs it). Per decode step and KV head:
  Riffle score (rotated query for mean / covariance, original query for the bound) -> Riffle top-m select
  -> FoveaKV pass over the selected pages (page-list mode, both planes) + FoveaKV bf16 window pass
  -> Riffle combine (unrotates the page partials with FoveaKV's R).
No host synchronization: m and the grid sizes follow from the token count, so the step is CUDA-graph capturable.
attend_ref() is the PyTorch reference (the opened pages dequantized at 4 bits + the exact window)."""
import math
import os

import torch

from .deps import foveakv_gpu

fg = foveakv_gpu()
from riffle import summarize                       # noqa: E402
from riffle.ext import ext as riffle_ext            # noqa: E402

BLK, D, G = fg.BLK, fg.D, fg.G
SIG = 0.5


class StackCache(fg.GainShape):
    def __init__(self, Hkv, Tmax, device="cuda", rho=0.1, cov=True, bound=True, sig=SIG):
        super().__init__(Hkv, Tmax, device, mode="gs", bits=4, block_mean=True, residual=True)
        self.rho, self.cov, self.bound, self.sig = rho, cov, bound, sig
        npg = self.cap // BLK
        f16 = lambda *s: torch.zeros(*s, dtype=torch.float16, device=device)
        self.U = f16(Hkv, npg, 2, D)                                            # rotated domain
        self.lamres = torch.zeros(Hkv, npg, 3, device=device)
        self.kmx, self.kmn = f16(Hkv, npg, D), f16(Hkv, npg, D)                 # original domain
        self._scores = torch.empty(Hkv, npg, device=device)
        self._sel = torch.zeros(Hkv, npg, dtype=torch.int32, device=device)
        self.use_kernel = True
        self.encode_graph = os.environ.get("CAIRN_ENCODE_GRAPH", "1") == "1"   # decode-time block encode from a graph
        tag = "+".join(t for t, on in (("cov", cov), ("bound", bound)) if on) or "mean"
        self.name = f"stack rho={rho:g} {tag}"
        self.last_sel = None

    def _encode(self, a, b, kx, vx):
        super()._encode(a, b, kx, vx)
        s = summarize(kx)                                                       # (Hkv, pages, ...) from raw keys
        p0, p1 = a // BLK, b // BLK
        self.U[:, p0:p1] = (s["U"] @ self.Rt).half()
        self.lamres[:, p0:p1] = s["lamres"]
        self.kmx[:, p0:p1], self.kmn[:, p0:p1] = s["mx"].half(), s["mn"].half()

    def _encode_block_dev(self, page):
        """the decode-time encode (one block: the oldest BLK tail tokens, kraw[:BLK]) at block index page ((1,) long on
        the GPU), every write device-indexed: FoveaKV codes + Riffle summaries, then the tail shifted down one block
        and the device counts advanced. Graph-captured by pre_step (same arithmetic as _encode + _encode_ready)."""
        kx, vx = self.kraw[:BLK], self.vraw[:BLK]
        self.encode_block_dev(kx, vx, page)
        s = summarize(kx)
        self.U.index_copy_(1, page, (s["U"] @ self.Rt).half())
        self.lamres.index_copy_(1, page, s["lamres"])
        self.kmx.index_copy_(1, page, s["mx"].half())
        self.kmn.index_copy_(1, page, s["mn"].half())
        r = self._enc_rem
        self._shift_k.copy_(self.kraw[BLK:BLK + r]); self.kraw[:r].copy_(self._shift_k)
        self._shift_v.copy_(self.vraw[BLK:BLK + r]); self.vraw[:r].copy_(self._shift_v)
        self.npg_dev += 1
        self.ncomp_dev += BLK

    def n_open(self):
        npg = self.n_comp // BLK
        return min(npg, max(1, math.ceil(self.rho * npg)))

    def page_scores(self, q):
        from riffle import page_scores
        npg = self.n_comp // BLK
        qs = (q.float() * self.scale).view(self.Hkv, G, D)
        return page_scores(qs @ self.Rt, self.kmu[:, :npg], self.U[:, :npg], self.lamres[:, :npg], self.kmx[:, :npg],
                           self.kmn[:, :npg], self.cov, self.bound, self.sig, q_b=qs)

    def attend_kernel(self, q):
        re = riffle_ext(D)
        npg, m = self.n_comp // BLK, self.n_open()
        qs = (q.float() * self.scale).view(self.Hkv, G, D).contiguous()
        qr = (qs @ self.Rt).contiguous()
        re.score(qr, qs, self.kmu, self.U, self.lamres, self.kmx, self.kmn, npg, int(self.cov), int(self.bound),
                 self.sig, self._scores, None)
        re.select(self._scores, npg, m, self._sel, None, 0.0, None)
        n_tok = m * BLK
        tpb = fg.tpb_for(n_tok, self.Hkv)
        nb1 = -(-n_tok // tpb)
        p1 = self._partials(nb1)
        fg.run_pass(1, qr, self.Hkv, self.cap, 0, n_tok, None, tpb, 0, p1, pages=self._sel,
                    **self._gs_args(1, 1, self.k16, self.v16))
        kw, vw = self._window()
        nw = kw.shape[1]
        nb2 = -(-nw // 64)
        p2 = self._partials(max(nb2, 1))
        if nw:
            fg.run_pass(0, qs, self.Hkv, nw, 0, nw, None, 64, 0, p2, kb=kw, vb=vw)
        out = torch.empty(self.Hkv * G, D, device=q.device)
        re.combine(*p1, nb1, *p2, nb2, self.R, out)
        return out

    # ---------------------------------------------------------------------------- graph mode (cairn.engine)
    # Every launch has a fixed grid (sized for the capacity) and reads the live counts from the GPU: compressed pages
    # (npg_dev), compressed tokens (ncomp_dev), selected tokens (ntok_dev, written by the select kernel), window length
    # (pos + 1 - ncomp). Block encoding (every 128 tokens) runs eagerly in pre_step(), before the step's graph.
    def graph_setup(self, ns_win=4):
        npg_cap = self.cap // BLK
        self.m_max = min(npg_cap, max(1, math.ceil(self.rho * npg_cap)))
        self.tpb_g = fg.tpb_for(self.m_max * BLK, self.Hkv)
        self.nb1_g = -(-self.m_max * BLK // self.tpb_g)
        self.p1_g, self.p2_g, self.ns_win = self._partials(self.nb1_g), self._partials(ns_win), ns_win
        i32 = lambda: torch.zeros(1, dtype=torch.int32, device=self.dev)
        self.npg_dev, self.ncomp_dev, self.ntok_dev = i32(), i32(), i32()
        self.out_g = torch.empty(self.Hkv * G, D, device=self.dev)
        self.gs_g = self._gs_args(1, 1, self.k16, self.v16)
        self._sync_dev()

    def _sync_dev(self):
        self.npg_dev.fill_(self.n_comp // BLK)
        self.ncomp_dev.fill_(self.n_comp)

    def pre_step(self):
        """before a graph step that appends one token: encode the blocks that this append makes old enough. In decode
        that is exactly one block with a fixed tail length, replayed from a captured CUDA graph (CAIRN_ENCODE_GRAPH=0:
        eager, as before); anything else (several blocks at once) runs eagerly."""
        target = max(0, ((self.n + 1 - fg.W) // BLK) * BLK)
        if target > self.n_comp:
            nb, rem = target - self.n_comp, self.n - target
            if nb == BLK and rem == fg.W - 1 and self.encode_graph and hasattr(self, "npg_dev"):
                if getattr(self, "_enc_graph", None) is None:
                    self._capture_encode(rem)
                self._enc_graph.replay()
            else:
                self._encode(self.n_comp, target, self.kraw[:nb], self.vraw[:nb])
                self.kraw[:rem] = self.kraw[nb:nb + rem].clone()
                self.vraw[:rem] = self.vraw[nb:nb + rem].clone()
                self.n_comp = target
                self._sync_dev()
                self.n += 1
                return
            self.n_comp = target                                                # host bookkeeping, no sync
        self.n += 1

    def _capture_encode(self, rem):
        """capture _encode_block_dev at the live page index npg_dev (warm-up on a throwaway copy of the counters:
        the capture itself must not advance them)."""
        self._enc_rem = rem
        self._shift_k = torch.empty(rem, self.Hkv, D, dtype=self.kraw.dtype, device=self.dev)
        self._shift_v = torch.empty_like(self._shift_k)
        self._enc_page = torch.zeros(1, dtype=torch.long, device=self.dev)
        state = [t.clone() for t in self._enc_state()]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                                              # warm-up (cuBLAS / allocator init)
            self._enc_page.copy_(self.npg_dev.long())
            self._encode_block_dev(self._enc_page)
        torch.cuda.current_stream().wait_stream(s)
        for t, v in zip(self._enc_state(), state):                              # undo the warm-up's writes
            t.copy_(v)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._enc_page.copy_(self.npg_dev.long())
            self._encode_block_dev(self._enc_page)
        self._enc_graph = g

    def _enc_state(self):
        """what the warm-up must not leave changed: the tail and the counts. (Its page writes are the very values the
        first replay writes at the same page, from the same tail.)"""
        return [self.kraw, self.vraw, self.npg_dev, self.ncomp_dev]

    def graph_attend(self, q, k, v, pos):
        """inside a CUDA graph: append k, v (1, Hkv, D) at position pos (1,) long, attend q (Hq, D) -> (Hq, D) fp32."""
        idx = pos - self.ncomp_dev.long()
        self.kraw.index_copy_(0, idx, k.to(torch.bfloat16))
        self.vraw.index_copy_(0, idx, v.to(torch.bfloat16))
        re = riffle_ext(D)
        npg_cap = self.cap // BLK
        qs = (q.float() * self.scale).view(self.Hkv, G, D)
        qr = (qs @ self.Rt).contiguous()
        re.score(qr, qs.contiguous(), self.kmu, self.U, self.lamres, self.kmx, self.kmn, npg_cap, int(self.cov),
                 int(self.bound), self.sig, self._scores, self.npg_dev)
        re.select(self._scores, npg_cap, self.m_max, self._sel, self.npg_dev, float(self.rho), self.ntok_dev)
        fg.run_pass(1, qr, self.Hkv, self.cap, 0, self.m_max * BLK, None, self.tpb_g, 0, self.p1_g, pages=self._sel,
                    n_dev=self.ntok_dev, **self.gs_g)
        from .kernels import ops
        ops().dense_attn(q, self.kraw, self.vraw, pos, self.ncomp_dev, *self.p2_g, None, True, self.scale)
        re.combine(*self.p1_g, self.nb1_g, *self.p2_g, self.ns_win, self.R, self.out_g)
        return self.out_g

    def attend_ref(self, q):
        sel = torch.topk(self.page_scores(q), self.n_open(), dim=-1).indices
        self.last_sel = sel
        K, V = self.dequant()                                                   # (n_comp, Hkv, D), original domain
        kt, vt = self.tail()
        qh = (q.float() * self.scale).view(self.Hkv, G, D)
        out = []
        for h in range(self.Hkv):
            tok = (sel[h][:, None] * BLK + torch.arange(BLK, device=q.device)[None]).reshape(-1)
            Kh = torch.cat([K[tok, h], kt[:, h].float()])
            Vh = torch.cat([V[tok, h], vt[:, h].float()])
            out.append(torch.softmax(qh[h] @ Kh.T, -1) @ Vh)
        return torch.cat(out)

    def attend(self, q):
        if self.n_comp == 0:
            return super().attend(q)
        return self.attend_kernel(q) if self.use_kernel else self.attend_ref(q)

    def read_bits(self):
        """bits read per cached token per step: opened pages at 4 bits + gain + mean, summaries, window bf16."""
        nw, npg = self.n - self.n_comp, self.n_comp // BLK
        if npg == 0:
            return 16.0
        page_bits = BLK * D * (4 + 8 / D) + 16 * D
        summ = 16 * D * ((2 if self.cov else 0) + (2 if self.bound else 0)) + 32 * 3
        per_kv_head = self.n_open() * 2 * page_bits + npg * summ
        return (per_kv_head / (2 * D) + 16 * nw) / self.n


def make_stack(name, Hkv, Tmax, device="cuda"):
    """stack<rho%>[-mean|-bound|-cov], e.g. stack10 (covariance + bound), stack20, stack10-bound (bound only)."""
    base, _, var = name.partition("-")
    rho = int(base[5:]) / 100
    cov, bound = {"": (True, True), "mean": (False, False), "bound": (False, True), "cov": (True, False)}[var]
    c = StackCache(Hkv, Tmax, device, rho=rho, cov=cov, bound=bound)
    c.use_kernel = os.environ.get("CAIRN_STACK_KERNEL", "1") == "1"
    return c
