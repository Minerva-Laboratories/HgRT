"""Single-sequence decode engine for Qwen3.5 (Gated DeltaNet + gated full attention), built for speed on one GPU.

Per token: the work between two full-attention layers runs as one captured CUDA graph (embedding, DeltaNet layers, MLPs,
norms, the next attention layer's projections and RoPE, and at the end the LM head and the greedy pick), so Python and
launch overhead vanish; attention-cache calls run eagerly between the graphs (8 per token on the 9B), because their
kernels' grids follow the cache length, except for caches with graph_attend (device-side counts, e.g. hgrt.stack),
which run inside one graph for the whole token (cache_graph). Weights: "w4" (int4 groups of 32 with llama.cpp Q4_0
rounding, GEMVs on int8-quantized activations like llama.cpp's Q4_0 x Q8 path; LM head int8 per row) or "bf16" (cuBLAS; for checking).
Positions and the current token live on the GPU, so a generation loop never synchronizes with the host.

Caches: any object with prefill(k, v) [(T, Hkv, D)], append(k, v) [(1, Hkv, D)], attend(q) [(Hq, D) -> (Hq, D) fp32];
"dense" here is the bf16 baseline stored head-major and read by FoveaKV's split-K bf16 kernel (no per-step copies)."""
import math

import torch
import torch.nn.functional as F

from .deps import foveakv_gpu
from .kernels import W4Linear, W8Linear, ext as gemv_ext, ops

DEV = "cuda"


# ------------------------------------------------------------------------------------------------- caches
class DenseHM:
    """bf16 K/V, head-major (Hkv, Tmax, D), exact attention with the split-K bf16 kernel."""
    name = "bf16 dense (head-major kernel)"

    def __init__(self, Hkv, Tmax, device=DEV):
        fg = foveakv_gpu()
        self.fg, self.Hkv, self.Tmax, self.D, self.G = fg, Hkv, Tmax, fg.D, fg.G
        self.kb = torch.zeros(Hkv, Tmax, self.D, dtype=torch.bfloat16, device=device)
        self.vb = torch.zeros_like(self.kb)
        self.n, self.scale = 0, self.D ** -0.5

    def prefill(self, k, v):
        T = k.shape[0]
        self.kb[:, :T] = k.transpose(0, 1).to(torch.bfloat16)
        self.vb[:, :T] = v.transpose(0, 1).to(torch.bfloat16)
        self.n = T

    def append(self, k, v):
        self.kb[:, self.n] = k[0].to(torch.bfloat16)
        self.vb[:, self.n] = v[0].to(torch.bfloat16)
        self.n += 1

    def attend(self, q):
        fg, n = self.fg, self.n
        tpb = fg.tpb_for(n, self.Hkv)
        nb = -(-n // tpb)
        part = (torch.empty(nb, self.Hkv, self.G, device=q.device), torch.empty(nb, self.Hkv, self.G, device=q.device),
                torch.empty(nb, self.Hkv, self.G, self.D, device=q.device))
        qk = (q.float() * self.scale).view(self.Hkv, self.G, self.D).contiguous()
        fg.run_pass(0, qk, self.Hkv, self.Tmax, 0, n, None, tpb, 0, part, kb=self.kb, vb=self.vb)
        return fg.ext().combine_partials(*part, nb, True)[0]

    def read_bits(self):
        return 16.0


def make_cache(name, Hkv, Tmax):
    if name == "dense":
        return DenseHM(Hkv, Tmax)
    from .caches import make_cache as mk
    return mk(name, Hkv, Tmax)


# ------------------------------------------------------------------------------------------------- weights
def _rows_for(N, K):
    """GEMV variant (rows per warp R, groups in flight U, encoded R*10+U) measured fastest on Orin for the shape:
    ~144-147 GB/s against a 153 GB/s read ceiling at MODE_50W."""
    if K >= 8192 or N >= 20000:
        return 82
    if N >= 10240:
        return 8
    return 4


class _Lin:
    """one linear (or several concatenated along N) in the chosen format; x (K,) bf16 -> (N,) bf16 [+ res]."""

    def __init__(self, ws, fmt, xq, pk=None):
        self.fmt, self.xq = fmt, xq
        if pk is not None and fmt == "w4":               # pre-quantized codes (e.g. AWQ), concatenated along N
            self.q = W4Linear.__new__(W4Linear)
            self.q.codes = torch.cat([c for c, _ in pk]) if len(pk) > 1 else pk[0][0]
            self.q.scales = torch.cat([s for _, s in pk]) if len(pk) > 1 else pk[0][1]
            self.N, self.K = self.q.codes.shape[0], self.q.codes.shape[1] * 32
            self.rows = _rows_for(self.N, self.K)
            return
        w = torch.cat([w.detach() for w in ws], 0) if len(ws) > 1 else ws[0].detach()
        self.N, self.K = w.shape
        if fmt == "w4":
            self.q = W4Linear(w.to(DEV))
            self.rows = _rows_for(self.N, self.K)
        else:
            self.w = w.to(DEV, torch.bfloat16).contiguous()

    def __call__(self, x, res=None, prequant=False):
        """prequant: the producer already wrote x's int8 groups into the shared buffers (fused quantization)."""
        if self.fmt == "w4":
            xq, xd = self.xq(self.K)
            if not prequant:
                gemv_ext().quant_x(x, xq, xd)
            y = torch.empty(self.N, dtype=torch.bfloat16, device=DEV)
            gemv_ext().w4q(self.q.codes, self.q.scales, xq, xd, res, y, self.rows)
            return y
        y = self.w @ x
        return y if res is None else y + res


class Engine:
    @torch.no_grad()
    def __init__(self, hf_model, weights="w4", lm_head="w8", Tmax=262144 + 1024, fused=True, packed=None):
        """packed: {hf parameter name: (codes, scales)} pre-quantized 4-bit weights (hgrt.awq.load_awq) used as stored."""
        m = hf_model
        self.fused = fused
        names = {id(p): n for n, p in m.named_parameters()}
        packed = packed or {}

        def lin(ws):
            pk = [packed.get(names[id(w)]) for w in ws]
            return _Lin(ws, weights, self.xq_buf, pk if all(x is not None for x in pk) else None)
        cfg = m.config
        self.cfg, self.weights = cfg, weights
        self.types = cfg.layer_types
        self.attn_ids = [i for i, t in enumerate(self.types) if t == "full_attention"]
        self.L, self.H = cfg.num_hidden_layers, cfg.hidden_size
        self.hd, self.Hq, self.Hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        self.eps = cfg.rms_norm_eps
        self.Tmax = Tmax
        self._xq = {}
        mm = m.model
        self.embed = mm.embed_tokens.weight
        f32 = lambda t: t.detach().float().to(DEV)
        self.norm_f = 1 + f32(mm.norm.weight)
        self.layers = []
        for i, layer in enumerate(mm.layers):
            d = dict(ln1=1 + f32(layer.input_layernorm.weight), ln2=1 + f32(layer.post_attention_layernorm.weight))
            mlp = layer.mlp
            d["gu"] = lin([mlp.gate_proj.weight, mlp.up_proj.weight])
            d["down"] = lin([mlp.down_proj.weight])
            if self.types[i] == "full_attention":
                a = layer.self_attn
                d["qkv"] = lin([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight])
                d["o"] = lin([a.o_proj.weight])
                d["qn"], d["kn"] = 1 + f32(a.q_norm.weight), 1 + f32(a.k_norm.weight)
            else:
                la = layer.linear_attn
                d["qkvz"] = lin([la.in_proj_qkv.weight, la.in_proj_z.weight])
                d["ba"] = torch.cat([la.in_proj_b.weight, la.in_proj_a.weight]).detach().to(DEV, torch.bfloat16)
                d["out"] = lin([la.out_proj.weight])
                d["conv_w"] = la.conv1d.weight.detach().squeeze(1).float().to(DEV)          # (conv_dim, 4)
                d["negA"] = -f32(la.A_log).exp()
                d["dt_bias"] = f32(la.dt_bias)
                d["gn"] = f32(la.norm.weight)
                self.nk, self.nv = la.num_k_heads, la.num_v_heads
                self.dk, self.dv = la.head_k_dim, la.head_v_dim
                self.key_dim, self.value_dim, self.conv_dim = la.key_dim, la.value_dim, la.conv_dim
            self.layers.append(d)
        lw = m.lm_head.weight.detach()
        self.lm = W8Linear(lw.to(DEV)) if lm_head == "w8" else None
        self.lm4 = _Lin([lw], "w4", self.xq_buf) if lm_head == "w4" else None
        self.lm_bf16 = lw.to(DEV, torch.bfloat16) if lm_head == "bf16" else None
        # rotary (partial, text positions: the interleaved mRoPE sections all see the same position)
        rp = cfg.rope_parameters
        self.rot = int(self.hd * rp.get("partial_rotary_factor", 1.0))
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (torch.arange(0, self.rot, 2, dtype=torch.float32, device=DEV) / self.rot))
        # device-side state
        self.pos = torch.zeros(1, dtype=torch.long, device=DEV)
        self.tok = torch.zeros(1, dtype=torch.long, device=DEV)
        self.h = torch.zeros(self.H, dtype=torch.bfloat16, device=DEV)
        self.qbuf = torch.zeros(self.Hq, self.hd, dtype=torch.float32, device=DEV)
        self.kbuf = torch.zeros(1, self.Hkv, self.hd, dtype=torch.float32, device=DEV)
        self.vbuf = torch.zeros_like(self.kbuf)
        self.gbuf = torch.zeros(self.Hq * self.hd, dtype=torch.bfloat16, device=DEV)
        self.obuf = torch.zeros(self.Hq, self.hd, dtype=torch.float32, device=DEV)
        self.logits = torch.zeros(cfg.vocab_size, dtype=torch.float32, device=DEV)
        self.conv = {i: torch.zeros(self.conv_dim, 4, dtype=torch.float32, device=DEV) for i in range(self.L) if i not in self.attn_ids}
        self.rec = {i: torch.zeros(self.nv, self.dk, self.dv, dtype=torch.float32, device=DEV) for i in self.conv}
        self.caches = None
        self.graphs = None
        self.dense = None                  # inline bf16 K/V {layer: (kb, vb)} (Hkv, Tmax, D) when the cache is "dense"
        self.graph_dense = None
        self.mode = None
        self.cache_graph = True            # caches with graph_attend run inside the token's graph

    def xq_buf(self, K):
        if K not in self._xq:
            self._xq[K] = (torch.empty(K, dtype=torch.int8, device=DEV), torch.empty(K // 32, 2, device=DEV))
        return self._xq[K]

    # ---------------------------------------------------------------------------------------------- pieces
    def _rms(self, x, w, quant=False):
        """quant: also write x's int8 groups for the next 4-bit GEMV (one kernel instead of two)."""
        if self.fused and x.dim() == 1:
            y = torch.empty_like(x)
            if quant and self.weights == "w4":
                ops().rms_q(x, w, y, *self.xq_buf(x.shape[0]), self.eps)
            else:
                ops().rms(x, w, y, self.eps)
            return y
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w).to(torch.bfloat16)

    def _rope(self, x):                                                       # x (heads, hd) float
        ang = self.pos.float() * self.inv_freq                                 # (rot/2,)
        cos = torch.cat([ang.cos(), ang.cos()]).to(torch.bfloat16).float()
        sin = torch.cat([ang.sin(), ang.sin()]).to(torch.bfloat16).float()
        r = x[:, :self.rot]
        half = self.rot // 2
        rh = torch.cat([-r[:, half:], r[:, :half]], -1)
        return torch.cat([r * cos + rh * sin, x[:, self.rot:]], -1)

    def _mlp(self, d):
        pq = self.fused and self.weights == "w4"
        gu = d["gu"](self._rms(self.h, d["ln2"], quant=True), prequant=pq)
        n = gu.shape[0] // 2
        if self.fused:
            act = torch.empty(n, dtype=torch.bfloat16, device=DEV)
            if pq:
                ops().silu_mul_q(gu, act, *self.xq_buf(n))
                self.h.copy_(d["down"](act, res=self.h, prequant=True))
                return
            ops().silu_mul(gu, act)
        else:
            act = (F.silu(gu[:n].float()) * gu[n:].float()).to(torch.bfloat16)
        self.h.copy_(d["down"](act, res=self.h))

    def _deltanet(self, i, d):
        pq = self.fused and self.weights == "w4"
        x = self._rms(self.h, d["ln1"], quant=True)
        qkvz = d["qkvz"](x, prequant=pq)
        ba = d["ba"] @ x
        b, a = ba[:self.nv], ba[self.nv:]
        mixed, z = qkvz[:self.conv_dim], qkvz[self.conv_dim:]
        if self.fused:
            mf = torch.empty(self.conv_dim, dtype=torch.float32, device=DEV)
            ops().conv(mixed, self.conv[i], d["conv_w"], mf)
            o = torch.empty(self.value_dim, dtype=torch.bfloat16, device=DEV)
            xq = self.xq_buf(self.value_dim) if pq else (None, None)
            ops().delta(mf, z, ba, d["negA"], d["dt_bias"], d["gn"], self.rec[i], o, self.nk, self.nv, self.eps, *xq)
            self.h.copy_(d["out"](o, res=self.h, prequant=pq))
            return
        cs = self.conv[i]
        cs.copy_(torch.cat([cs[:, 1:], mixed.float()[:, None]], 1))
        mixed = F.silu((cs * d["conv_w"]).sum(-1)).to(torch.bfloat16).float()
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim])
        q = q.view(self.nk, self.dk)
        k = k.view(self.nk, self.dk)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(self.dk)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        rep = self.nv // self.nk
        q, k = q.repeat_interleave(rep, 0), k.repeat_interleave(rep, 0)
        v = v.view(self.nv, self.dv)
        beta = torch.sigmoid(b.float())
        g = d["negA"] * F.softplus(a.float() + d["dt_bias"])
        S = self.rec[i]
        S.mul_(g.exp()[:, None, None])
        kv = torch.einsum("hkv,hk->hv", S, k)
        delta = (v - kv) * beta[:, None]
        S.add_(k[:, :, None] * delta[:, None, :])
        o = torch.einsum("hkv,hk->hv", S, q).to(torch.bfloat16).float()
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.eps)
        o = (d["gn"] * o.to(torch.bfloat16).float()).to(torch.bfloat16).float() * F.silu(z.view(self.nv, self.dv).float())
        self.h.copy_(d["out"](o.to(torch.bfloat16).reshape(-1), res=self.h))

    def _attn_pre(self, d):
        pq = self.fused and self.weights == "w4"
        x = self._rms(self.h, d["ln1"], quant=True)
        y = d["qkv"](x, prequant=pq)
        if self.fused:
            ops().attn_pre(y, d["qn"], d["kn"], self.pos, self.inv_freq, self.qbuf, self.kbuf[0], self.vbuf[0], self.gbuf,
                           self.Hq, self.Hkv, self.rot, self.eps)
            return
        nq = self.Hq * self.hd * 2
        qg = y[:nq].view(self.Hq, 2 * self.hd)
        q, gate = qg[:, :self.hd], qg[:, self.hd:]
        k = y[nq:nq + self.Hkv * self.hd].view(self.Hkv, self.hd)
        v = y[nq + self.Hkv * self.hd:].view(self.Hkv, self.hd)
        q = self._rms(q, d["qn"]).float()
        k = self._rms(k, d["kn"]).float()
        self.qbuf.copy_(self._rope(q).to(torch.bfloat16).float())
        self.kbuf.copy_(self._rope(k).to(torch.bfloat16).float()[None])
        self.vbuf.copy_(v.float()[None])
        self.gbuf.copy_(gate.reshape(-1))

    def _attn_post(self, d):
        if self.fused:
            o = torch.empty(self.Hq * self.hd, dtype=torch.bfloat16, device=DEV)
            if self.weights == "w4":
                ops().attn_post_q(self.obuf.reshape(-1), self.gbuf, o, *self.xq_buf(o.shape[0]))
                self.h.copy_(d["o"](o, res=self.h, prequant=True))
                return
            ops().attn_post(self.obuf.reshape(-1), self.gbuf, o)
            self.h.copy_(d["o"](o, res=self.h))
            return
        o = (self.obuf.to(torch.bfloat16).reshape(-1).float() * torch.sigmoid(self.gbuf.float())).to(torch.bfloat16)
        self.h.copy_(d["o"](o, res=self.h))

    def _head(self):
        x = self._rms(self.h, self.norm_f)
        if self.lm is not None:
            self.lm(x, out=self.logits)
        elif self.lm4 is not None:
            self.logits.copy_(self.lm4(x).float())
        else:
            self.logits.copy_((self.lm_bf16 @ x).float())

    # ---------------------------------------------------------------------------------------------- segments
    def _segments(self):
        """[fn] alternating with attention calls: seg 0 = embed .. first attention's projections; seg j = attention j-1's
        output projection + MLP .. next attention's projections; last = .. LM head, greedy token, position + 1."""
        A = self.attn_ids
        segs = []

        def run_range(lo, hi):                      # layers [lo, hi): full layers (non-attention)
            for i in range(lo, hi):
                self._deltanet(i, self.layers[i])
                self._mlp(self.layers[i])

        def seg0():
            self.h.copy_(torch.index_select(self.embed, 0, self.tok)[0])
            run_range(0, A[0])
            self._attn_pre(self.layers[A[0]])
        segs.append(seg0)
        for j in range(len(A)):
            a = A[j]
            nxt = A[j + 1] if j + 1 < len(A) else None

            def seg(a=a, nxt=nxt):
                self._attn_post(self.layers[a])
                self._mlp(self.layers[a])
                run_range(a + 1, nxt if nxt is not None else self.L)
                if nxt is not None:
                    self._attn_pre(self.layers[nxt])
                else:
                    self._head()
                    self.tok.copy_(self.logits.argmax()[None])
                    self.pos.add_(1)
            segs.append(seg)
        return segs

    @torch.no_grad()
    def _attn_inline(self, i):
        kb, vb = self.dense[i]
        kb.index_copy_(1, self.pos, self.kbuf.transpose(0, 1).to(torch.bfloat16))
        vb.index_copy_(1, self.pos, self.vbuf.transpose(0, 1).to(torch.bfloat16))
        ops().dense_attn(self.qbuf, kb, vb, self.pos, None, *self._dparts, self.obuf, False, self.hd ** -0.5)

    def _token_cache(self):
        segs = self._segments()
        segs[0]()
        for j, i in enumerate(self.attn_ids):
            self.obuf.copy_(self.caches[i].graph_attend(self.qbuf, self.kbuf, self.vbuf, self.pos))
            segs[j + 1]()

    def _token_dense(self):
        segs = self._segments()
        segs[0]()
        for j, i in enumerate(self.attn_ids):
            self._attn_inline(i)
            segs[j + 1]()

    def _capture_one(self, fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        saved = self._save_state()
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        self._load_state(saved)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            fn()
        self._load_state(saved)
        return g

    @torch.no_grad()
    def capture(self):
        segs = self._segments()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        saved = self._save_state()
        with torch.cuda.stream(s):                               # warm-up (Triton/cuBLAS init) outside capture
            for f in segs:
                f()
        torch.cuda.current_stream().wait_stream(s)
        self._load_state(saved)
        self._pool = pool = torch.cuda.graph_pool_handle()
        self.graphs = []
        for f in segs:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                f()
            self.graphs.append(g)
        self._load_state(saved)

    def _save_state(self):
        return ([t.clone() for t in (self.pos, self.tok, self.h)], {i: t.clone() for i, t in self.conv.items()},
                {i: t.clone() for i, t in self.rec.items()})

    def _load_state(self, st):
        (p, t, h), conv, rec = st
        self.pos.copy_(p); self.tok.copy_(t); self.h.copy_(h)
        for i in conv:
            self.conv[i].copy_(conv[i]); self.rec[i].copy_(rec[i])

    # ---------------------------------------------------------------------------------------------- API
    @torch.no_grad()
    def load(self, snap, cache_name):
        """state from a prefill snapshot (sparsepage.qwen / hgrt.pipeline): K/V of the attention layers, DeltaNet states,
        first-token logits; the first decode input is the greedy pick from those logits."""
        T = snap["T"]
        self.caches = {}
        self.mode = cache_name
        if cache_name == "dense":
            assert T + 1 < self.Tmax, "Tmax"
            if self.dense is None:
                z = lambda: torch.zeros(self.Hkv, self.Tmax, self.hd, dtype=torch.bfloat16, device=DEV)
                self.dense = {i: (z(), z()) for i in self.attn_ids}
                NS, HG = 64, self.Hq
                self._dparts = (torch.empty(NS, HG, device=DEV), torch.empty(NS, HG, device=DEV),
                                torch.empty(NS, HG, self.hd, device=DEV))
            for i, (k, v) in snap["kv"].items():
                self.dense[i][0][:, :T] = k.transpose(0, 1).to(torch.bfloat16)
                self.dense[i][1][:, :T] = v.transpose(0, 1).to(torch.bfloat16)
        for i, (k, v) in (snap["kv"].items() if cache_name != "dense" else ()):
            c = make_cache(cache_name, self.Hkv, max(self.Tmax, T + 1024))
            c.prefill(k, v)
            if hasattr(c, "graph_setup"):
                c.graph_setup()
            self.caches[i] = c
        self.graph_cache = None
        lin = snap["lin"]
        for i in self.conv:
            cs = lin.layers[i].conv_states[0][0].float()                       # (conv_dim, 4)
            self.conv[i].copy_(cs[:, -4:])
            self.rec[i].copy_(lin.layers[i].recurrent_states[0][0].float())
        self.pos.fill_(T)
        self.tok.copy_(snap["last_logits"].argmax()[None])
        self.logits.copy_(snap["last_logits"].float())

    @torch.no_grad()
    def step(self, token=None):
        """one token: feeds `token` (a 1-element long tensor on the GPU, teacher forcing) or the current greedy pick;
        afterwards self.logits are the next-token logits and self.tok the greedy pick. No host synchronization."""
        if token is not None:
            self.tok.copy_(token)
        if self.mode == "dense":
            if self.graphs is None:                      # eager
                self._token_dense()
                return
            if self.graph_dense is None:
                self.graph_dense = self._capture_one(self._token_dense)
            self.graph_dense.replay()
            return
        if self.cache_graph and self.caches and all(hasattr(c, "graph_attend") for c in self.caches.values()) and self.graphs is not None:
            for c in self.caches.values():
                c.pre_step()
            if self.graph_cache is None:
                self.graph_cache = self._capture_one(self._token_cache)
            self.graph_cache.replay()
            return
        segs = None if self.graphs is not None else self._segments()
        for j, i in enumerate([None] + self.attn_ids):
            if i is not None:
                c = self.caches[i]
                c.append(self.kbuf, self.vbuf)
                self.obuf.copy_(c.attend(self.qbuf))
            if self.graphs is not None:
                self.graphs[j].replay()
            else:
                segs[j]()

    @torch.no_grad()
    def generate(self, n):
        out = torch.empty(n, dtype=torch.long, device=DEV)
        for t in range(n):
            out[t] = self.tok[0]
            self.step()
        return out
