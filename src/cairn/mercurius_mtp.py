"""Self-speculative decoding for the HgRT Mercurius engine with Qwen3.5-4B's stock MTP module (unchanged weights).

MTP (one token row):  e = emb(x_{t+1}), h = the trunk's NORMED final hidden at t
    z = fc([RMSNorm_e(e) ; RMSNorm_h(h)]) -> one Qwen3.5 gated full-attention decoder layer (own KV cache, RoPE at t)
      -> mtp.norm -> tied LM head (the NF4 embedding table) => draft for x_{t+2}; recursive: (its normed output, its draft).
Its weights (bf16 in the checkpoint) are quantized once to NF4 blocks of 64 (they only steer drafting, so their rounding
can cost acceptance, never correctness); its cache is bf16 head-major, read by the split-K kernel of cairn's ops.

One speculation cycle, all on the GPU and captured as ONE CUDA graph (no host synchronization):
  1. MTP rows for positions P0 .. P0 + k (the previous cycle's verified rows: true hidden + true next token): they
     write the correct MTP cache entries for every accepted position, and row n (the last accepted) gives draft 1;
  2. MTP steps 2..k (one row each, recursive);
  3. trunk verification of [x_P, d_1..d_k] at positions P .. P + k in ONE M = k + 1 pass (GDN states untouched);
  4. greedy acceptance n = #leading d_j == argmax_{j-1}; emit argmax_0..argmax_n; commit the GDN states for the n + 1
     accepted positions (recomputed exactly from the saved per-step inputs); the MLA cache keeps the accepted rows
     (rejected rows are overwritten later); P += n + 1.
The emitted tokens equal plain greedy decoding by construction: the verification pass computes every row with the
arithmetic of the single-token step (tests/test_mercurius.py), and acceptance compares against those argmaxes."""
import glob
import json
import os

import torch
import torch.nn.functional as F

from .kernels import ops as cairn_ops
from .mercurius_engine import DEV, FF, HD, HID, NH, NF4Lin, RD
from .mercurius_kernels import ext

QWEN4B = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/")
NKV = 4


def quant_nf4(w, code, block=64):
    """bf16/fp32 (N, K) -> (packed (N, K/2) uint8 low nibble first, absmax (N, K/64) fp32): nearest NF4 level."""
    N, K = w.shape
    g = w.float().view(N, K // block, block)
    a = g.abs().amax(-1).clamp_min(1e-12)
    x = g / a[..., None]
    mid = (code[1:] + code[:-1]) / 2
    q = torch.bucketize(x.contiguous(), mid.to(x.device)).to(torch.uint8).view(N, K)
    return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous(), a.float()


def quant_rows_kv(x, bits):
    """(..., 256) fp32 -> codes, fp16 scales as kvq_store_row: int8 per row (absmax / 127), or int4 per 32 values
    (absmax / 7, q + 8, low nibble first)."""
    if bits == 8:
        s = (x.abs().amax(-1, keepdim=True) / 127).half()
        inv = torch.where(s.float() > 0, 1 / s.float(), torch.zeros_like(s.float()))
        q = torch.round(x * inv).clamp(-127, 127).to(torch.int8)
        return q.view(torch.uint8), s
    g = x.view(*x.shape[:-1], -1, 32)
    s = (g.abs().amax(-1, keepdim=True) / 7).half()
    inv = torch.where(s.float() > 0, 1 / s.float(), torch.zeros_like(s.float()))
    q = (torch.round(g * inv) + 8).clamp(0, 15).to(torch.uint8).view(*x.shape)
    return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous(), s.squeeze(-1)


def load_mtp_tensors():
    root = glob.glob(QWEN4B)[0]
    from safetensors import safe_open
    idx = json.load(open(root + "model.safetensors.index.json"))["weight_map"]
    sd = {}
    for f in sorted({v for k, v in idx.items() if k.startswith("mtp.")}):
        with safe_open(root + f, "pt") as s:
            for k in s.keys():
                if k.startswith("mtp."):
                    sd[k[4:].replace("layers.0.", "")] = s.get_tensor(k)
    return sd


class MTPHead:
    @torch.no_grad()
    def __init__(self, eng, Tmax=None, draft_vocab=None, NSa=64, kv="bf16"):
        """kv: the MTP cache form: "bf16" (dense), "int8" (per kv-head-token row, fp16 scale) or "int4" (fp16 scale per
        32 values)."""
        self.E = eng
        sd = load_mtp_tensors()
        code = eng.emb_code.cpu()
        q = lambda *names: NF4Lin([quant_nf4(sd[n].to(DEV), code.to(DEV)) for n in names])
        self.fc = q("fc.weight")
        self.qkv = q("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight")
        self.o = q("self_attn.o_proj.weight")
        self.gu = q("mlp.gate_proj.weight", "mlp.up_proj.weight")
        self.down = q("mlp.down_proj.weight")
        g1 = lambda n: (1.0 + sd[n].float()).to(DEV)
        self.pre_e, self.pre_h = g1("pre_fc_norm_embedding.weight"), g1("pre_fc_norm_hidden.weight")
        self.ln1, self.ln2 = g1("input_layernorm.weight"), g1("post_attention_layernorm.weight")
        self.qn, self.kn = g1("self_attn.q_norm.weight"), g1("self_attn.k_norm.weight")
        self.norm = g1("norm.weight")
        self.Tmax = Tmax or eng.Tmax
        M = eng.maxM
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=DEV)
        self.kv = kv
        if kv == "bf16":
            self.kb = z(NKV, self.Tmax, HD, dt=torch.bfloat16)
            self.vb = z(NKV, self.Tmax, HD, dt=torch.bfloat16)
        else:
            self.bits = 8 if kv == "int8" else 4
            rb, nsc = (HD, 1) if self.bits == 8 else (HD // 2, HD // 32)
            self.kc, self.vc = z(NKV, self.Tmax, rb, dt=torch.uint8), z(NKV, self.Tmax, rb, dt=torch.uint8)
            self.ks, self.vs = z(NKV, self.Tmax, nsc, dt=torch.float16), z(NKV, self.Tmax, nsc, dt=torch.float16)
        self.cat = z(M, 2 * HID, dt=torch.bfloat16)
        self.ebuf = z(M, HID, dt=torch.bfloat16)
        self.zres = z(M, HID, dt=torch.bfloat16)
        self.x = z(M, HID, dt=torch.bfloat16)
        self.yqkv = z(M, NH * 2 * HD + 2 * NKV * HD, dt=torch.bfloat16)
        self.qbuf = z(NH, HD)
        self.gbuf = z(NH * HD, dt=torch.bfloat16)
        self.obuf = z(NH, HD)
        self.ya = z(M, NH * HD, dt=torch.bfloat16)
        self.gub = z(M, 2 * FF, dt=torch.bfloat16)
        self.out = z(1, HID, dt=torch.bfloat16)                        # normed output hidden (mtp.norm)
        self.qrows = z(M, NH, HD)
        self.grows = z(M, NH * HD, dt=torch.bfloat16)
        self.zsel = z(1, HID, dt=torch.bfloat16)
        self.parts = (z(NSa, NH), z(NSa, NH), z(NSa, NH, HD))
        self.xq5 = (z(M * 2 * HID, dt=torch.int8), z(M * 2 * HID // 32))
        # draft head: the full tied head, or the rows of a vocabulary subset (drafting only; verification is exact)
        if draft_vocab is None:
            self.head, self.vmap, self.Vd = eng.head, None, eng.V
        else:
            ids = draft_vocab.to(DEV).long()
            self.vmap = ids
            self.head = NF4Lin.__new__(NF4Lin)
            self.head.codes = eng.head.codes[ids].contiguous()
            self.head.scales = eng.head.scales[ids].contiguous()
            self.head.N, self.head.K = ids.numel(), HID
            self.head.mode, self.head.ks = eng.head.mode, getattr(eng.head, "ks", 1)
            self.Vd = ids.numel()
        self.dlogits = z(1, self.head.N)
        self.dtok = z(1, dt=torch.long)

    # ---------------------------------------------------------------------------------------------- forward
    def rows(self, hid, toks, pos0, M, row=None):
        """M MTP rows: hid (M, HID) bf16 normed trunk (or MTP) hidden, toks (M,) the next tokens, at positions
        pos0 .. pos0 + M - 1 (pos0: 1-element long GPU tensor). Every row writes its K/V; only row `row` (GPU index,
        default the last) is carried through attention, MLP and mtp.norm -> self.out[0] (its normed output)."""
        X, O, E = ext(), cairn_ops(), self.E
        E.embed(toks, self.ebuf, M)
        X.rms_rows(self.ebuf, HID, self.pre_e, self.cat, 2 * HID, None, None, M, HID, 1e-6)
        X.rms_rows(hid, HID, self.pre_h, self.cat[:, HID:], 2 * HID, None, None, M, HID, 1e-6)
        xq5, xd5 = self.xq5
        X.quant_rows(self.cat, 2 * HID, xq5, xd5, M, 2 * HID)
        self.fc(xq5, xd5, M, self.zres, HID)
        xq, xd = E.xq(HID)
        X.rms_rows(self.zres, HID, self.ln1, None, 0, xq, xd, M, HID, 1e-6)
        self.qkv(xq, xd, M, self.yqkv, self.yqkv.shape[1])
        if self.kv == "bf16":                                         # every row's K/V into the cache
            X.mtp_attn_pre(self.yqkv, self.yqkv.shape[1], self.qn, self.kn, pos0, E.inv_freq, self.qrows, self.kb,
                           self.vb, self.grows, 1e-6, M)
        else:
            X.mtp_attn_pre_q(self.yqkv, self.yqkv.shape[1], self.qn, self.kn, pos0, E.inv_freq, self.qrows, self.kc,
                             self.ks, self.vc, self.vs, self.grows, 1e-6, M, self.bits)
        if M > 1:                                                     # the carried row
            r = row if row is not None else torch.full((1,), M - 1, dtype=torch.long, device=DEV)
            p = pos0 + r
            self.qbuf.copy_(self.qrows.index_select(0, r)[0])
            self.gbuf.copy_(self.grows.index_select(0, r)[0])
            self.zsel.copy_(self.zres.index_select(0, r))
            z = self.zsel
        else:
            p = pos0
            self.qbuf.copy_(self.qrows[0])
            self.gbuf.copy_(self.grows[0])
            z = self.zres[0:1]
        if self.kv == "bf16":
            O.dense_attn(self.qbuf, self.kb, self.vb, p, None, *self.parts, self.obuf, False, HD ** -0.5)
        else:
            X.kvq_attn(self.qbuf, self.kc, self.ks, self.vc, self.vs, p, *self.parts, self.obuf, HD ** -0.5, self.bits)
        O.attn_post(self.obuf.reshape(-1), self.gbuf, self.ya[0])
        xq2, xd2 = E.xq(NH * HD)
        X.quant_rows(self.ya, NH * HD, xq2, xd2, 1, NH * HD)
        self.o(xq2, xd2, 1, z, HID, z, HID)
        X.rms_rows(z, HID, self.ln2, None, 0, xq, xd, 1, HID, 1e-6)
        self.gu(xq, xd, 1, self.gub, 2 * FF)
        xq3, xd3 = E.xq(FF)
        X.silu_mul_q(self.gub, 2 * FF, None, xq3, xd3, 1, FF)
        self.down(xq3, xd3, 1, z, HID, z, HID)
        X.rms_rows(z, HID, self.norm, self.out, HID, None, None, 1, HID, 1e-6)

    def cache_tensors(self):
        return (self.kb, self.vb) if self.kv == "bf16" else (self.kc, self.ks, self.vc, self.vs)

    def cache_tensors_clone(self):
        return tuple(t.clone() for t in self.cache_tensors())

    def draft(self):
        """greedy draft from self.out[0] -> self.dtok"""
        X, E = ext(), self.E
        xq, xd = E.xq(HID)
        X.quant_rows(self.out, HID, xq, xd, 1, HID)
        self.head(xq, xd, 1, self.dlogits, self.head.N)
        X.argmax_rows(self.dlogits, self.head.N, self.Vd, self.dtok, 1)
        if self.vmap is not None:
            self.dtok.copy_(self.vmap.index_select(0, self.dtok))

    # ---------------------------------------------------------------------------------------------- prompt cache
    @torch.no_grad()
    def prefill(self, snap, chunk=2048):
        """MTP K/V for prompt positions 0 .. T-2 from the trunk's normed hidden states (snapshot) and the next tokens,
        in PyTorch with the same (NF4-rounded) weights."""
        E = self.E
        ids = snap["ids"].to(DEV)
        hidden = snap["hidden"].to(DEV)
        T = ids.shape[0]
        lv = E.levels.to(DEV).float()
        Wfc = self.fc.dequant(lv).to(torch.bfloat16)
        Wqkv = self.qkv.dequant(lv)
        Wk = Wqkv[NH * 2 * HD:NH * 2 * HD + NKV * HD].to(torch.bfloat16)
        Wv = Wqkv[NH * 2 * HD + NKV * HD:].to(torch.bfloat16)
        rms = lambda x, w: (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w).to(torch.bfloat16)
        for a in range(0, T - 1, chunk):
            b = min(T - 1, a + chunk)
            e = E.emb_rows(ids[a + 1:b + 1])
            zz = torch.cat([rms(e, self.pre_e), rms(hidden[a:b], self.pre_h)], -1) @ Wfc.T
            x = rms(zz, self.ln1)
            k = (x @ Wk.T).view(b - a, NKV, HD)
            v = (x @ Wv.T).view(b - a, NKV, HD)
            k = rms(k, self.kn).float()
            pos = torch.arange(a, b, device=DEV).float()
            ang = pos[:, None] * E.inv_freq[None]
            cos = torch.cat([ang.cos(), ang.cos()], -1).to(torch.bfloat16).float()[:, None]
            sin = torch.cat([ang.sin(), ang.sin()], -1).to(torch.bfloat16).float()[:, None]
            r = k[..., :RD]
            rh = torch.cat([-r[..., RD // 2:], r[..., :RD // 2]], -1)
            k = torch.cat([r * cos + rh * sin, k[..., RD:]], -1).to(torch.bfloat16)
            if self.kv == "bf16":
                self.kb[:, a:b] = k.transpose(0, 1)
                self.vb[:, a:b] = v.transpose(0, 1)
            else:
                for x, codes, sc in ((k, self.kc, self.ks), (v, self.vc, self.vs)):
                    c, sv = quant_rows_kv(x.transpose(0, 1).float(), self.bits)
                    codes[:, a:b] = c
                    sc[:, a:b] = sv


class SpecDecoder:
    """Self-speculative greedy decoding on a MercuriusEngine + MTPHead.
    k: drafts per cycle (fixed), or adaptive=True: k in ks chosen every `group` cycles by maximizing expected tokens per
    cycle (geometric acceptance model, recent cycles) over the measured cycle time of each k.
    fused=True: the GDN-2 commit of a cycle is applied at the start of the next cycle's verification (one state pass)."""

    def __init__(self, eng, mtp, k=2, adaptive=False, ks=(1, 2, 3), fused=True, group=8):
        self.E, self.H, self.k, self.adaptive, self.ks, self.fused, self.group = eng, mtp, k, adaptive, ks, fused, group
        kmax = max(ks) if adaptive else k
        assert 1 <= kmax <= eng.maxM - 1
        self.M1 = eng.maxM if adaptive else k + 1             # MTP step-1 rows (all k must fit when k varies)
        z = lambda *s, dt=torch.long: torch.zeros(*s, dtype=dt, device=DEV)
        self.hbuf = z(eng.maxM, HID, dt=torch.bfloat16)      # MTP step-1 inputs: trunk normed hidden rows
        self.abuf = z(eng.maxM)                              # and their next tokens
        self.P0 = z(1)                                       # position of row 0
        self.nrow = z(1)                                     # last valid row (= accepted drafts of the last cycle)
        self.drafts = z(kmax)
        self.hrec = z(1, HID, dt=torch.bfloat16)
        self.out = z(1 << 16)
        self.count = z(1)
        self.ncycles = z(1)
        self.acc_hist = z(kmax + 1)
        self.graphs = {}
        self.parity = 0
        self.k_log = []

    @torch.no_grad()
    def start(self, snap):
        """engine already loaded with snap; prepares the MTP cache and the first cycle's step-1 inputs."""
        E, H = self.E, self.H
        H.prefill(snap)
        T = int(snap["ids"].shape[0])
        self.hbuf.zero_()
        self.hbuf[0].copy_(snap["hidden"][T - 1].to(DEV))
        self.abuf.zero_()
        self.abuf[0] = E.tok_in[0]
        self.P0.fill_(T - 1)
        self.nrow.zero_()
        self.out.zero_()
        self.out[0] = E.tok_in[0]                            # the first token (prefill's greedy pick), as plain decode
        self.count.fill_(1)
        self.ncycles.zero_()
        self.acc_hist.zero_()
        E.nacc_prev.fill_(-1)
        self.parity = 0
        self.k_log = []

    def _cycle(self, k, parity):
        E, H = self.E, self.H
        M = k + 1
        # 1. MTP rows for the previous cycle's verified positions (K/V of all); draft 1 from the last accepted row
        H.rows(self.hbuf[:self.M1], self.abuf[:self.M1], self.P0, self.M1, row=self.nrow)
        H.draft()
        self.drafts[0:1].copy_(H.dtok)
        pj = self.P0 + self.nrow                             # position of that row
        for j in range(1, k):                                # 2. recursive drafts (its own output + draft)
            self.hrec.copy_(H.out)
            H.rows(self.hrec, self.drafts[j - 1:j], pj + j, 1)
            H.draft()
            self.drafts[j:j + 1].copy_(H.dtok)
        # 3. verification: [x_P, d_1..d_k] at positions P .. P + k (fused: the previous commit is applied first)
        E.tok_in[1:M].copy_(self.drafts[:k])
        E._block(M, 3 if self.fused else 1, parity)
        # 4. acceptance, emission, commit
        match = (self.drafts[:k] == E.tok_out[:k]).long()
        n = match.cumprod(0).sum().view(1)
        idx = self.count + torch.arange(M, device=DEV)
        self.out.index_copy_(0, idx, E.tok_out[:M])         # rows beyond n are overwritten by the next cycle
        self.count.add_(n + 1)
        self.ncycles.add_(1)
        self.acc_hist.index_add_(0, n, torch.ones_like(n))
        if self.fused:
            E.nacc_prev.copy_(n.to(torch.int32))             # applied by the next cycle (or finalize)
        else:
            E.nacc.copy_(n.to(torch.int32))
            for i in E.lin_ids:
                E._gdn_commit(i)
        self.P0.copy_(E.pos)
        self.nrow.copy_(n)
        self.hbuf[:M].copy_(E.x[:M])
        self.abuf[:M].copy_(E.tok_out[:M])
        E.pos.add_(n + 1)
        E.tok_in[0:1].copy_(E.tok_out.index_select(0, n))

    def _state(self):
        E = self.E
        return (E.save_state(), self.hbuf.clone(), self.abuf.clone(), self.P0.clone(), self.nrow.clone(),
                self.count.clone(), self.ncycles.clone(), self.acc_hist.clone(), self.out.clone(),
                self.H.cache_tensors_clone(), {i: E.layers[i]["cache"].clone() for i in E.attn_ids})

    def _restore(self, saved):
        E = self.E
        st, hb, ab, p0, nr, cnt, nc, ah, out, hc, caches = saved
        E.load_state(st)
        self.hbuf.copy_(hb); self.abuf.copy_(ab); self.P0.copy_(p0); self.nrow.copy_(nr)
        self.count.copy_(cnt); self.ncycles.copy_(nc); self.acc_hist.copy_(ah); self.out.copy_(out)
        for dst, src in zip(self.H.cache_tensors(), hc):
            dst.copy_(src)
        for i in E.attn_ids:
            E.layers[i]["cache"].copy_(caches[i])

    def _graph(self, k, parity):
        key = (k, parity if self.fused else 0)
        if key not in self.graphs:
            saved = self._state()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._cycle(k, parity)
            torch.cuda.current_stream().wait_stream(s)
            self._restore(saved)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._cycle(k, parity)
            self._restore(saved)
            self.graphs[key] = g
        return self.graphs[key]

    @torch.no_grad()
    def warm(self, ks=None):
        """capture every graph that run() may need (state restored)."""
        for k in (ks or (self.ks if self.adaptive else (self.k,))):
            for p in ((0, 1) if self.fused else (0,)):
                self._graph(k, p)

    @torch.no_grad()
    def run(self, cycles, k=None, graph=True):
        k = k or self.k
        for _ in range(cycles):
            if graph:
                self._graph(k, self.parity).replay()
            else:
                self._cycle(k, self.parity)
            self.parity ^= 1

    @torch.no_grad()
    def finalize(self):
        """apply the pending GDN-2 commit (fused mode) so the engine state is the plain-decoding state."""
        if self.fused:
            self.E.finalize_pending(self.parity ^ 1)

    @torch.no_grad()
    def generate(self, n, graph=True, chunk=None):
        """>= n tokens (host sync every `group` cycles); returns (tokens[:n], cycles, accepted histogram)"""
        import time
        chunk = chunk or self.group
        if not self.adaptive:
            while int(self.count) < n:
                self.run(chunk, self.k, graph)
            self.finalize()
            return self.out[:n].clone(), int(self.ncycles), self.acc_hist.clone()
        cost, hist_prev = {}, torch.zeros_like(self.acc_hist).cpu()
        acc_n, acc_fail = 1.0, 1.0                           # prior: per-draft acceptance 0.5
        while True:
            if len(cost) < len(self.ks):                     # calibration: one group per k
                k = self.ks[len(cost)]
            else:
                a = acc_n / (acc_n + acc_fail)
                k = max(self.ks, key=lambda kk: (1 - a ** (kk + 1)) / (1 - a + 1e-9) / cost[kk])
            torch.cuda.synchronize()
            t0 = time.time()
            self.run(chunk, k, graph)
            cnt = int(self.count)                            # syncs
            dt = (time.time() - t0) / chunk
            cost[k] = dt if k not in cost else 0.7 * cost[k] + 0.3 * dt
            h = self.acc_hist.cpu()
            d = h - hist_prev
            hist_prev = h
            sn = float((d * torch.arange(len(d))).sum())
            fails = float(d[:k].sum())                       # cycles that stopped before using all k drafts
            acc_n, acc_fail = 0.8 * acc_n + sn, 0.8 * acc_fail + fails
            self.k_log.append(k)
            if cnt >= n:
                break
        self.finalize()
        return self.out[:n].clone(), int(self.ncycles), self.acc_hist.clone()
