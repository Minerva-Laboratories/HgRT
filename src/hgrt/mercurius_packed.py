"""HgRT decode engine for mercurius-packed-v1 checkpoints (Mercurius 0.5: 24 GDN-2 + 8 CARE-latent MLA layers, Q4_0 /
Q8_0 weights, e.g. Minerva-Laboratories/Mercurius-0.5-4B-Preview).

The checkpoint is read as exported (model.safetensors + config.json "mercurius" section, no transformers model): the
packed projections run on csrc/packed_ops.cu as stored (Q4_0 codes / fp16 group scales, Q8_0 for the GDN-2 gates),
every weight used as bf16((q - 8) s), the bf16 tensor the pipeline reconstructs and the QAT forward trained with.

Per token, the arithmetic follows the pipeline's modules (Mercurius-Pipeline src/mercurius/arch) with their fp32 / bf16
casts:
  * ScaleNorm (scalar gain) / zero-centred RMSNorm / gated RMSNorm;
  * GDN-2: conv over the rolling input state, channel-wise decay g = -exp(A_log) softplus(a + dt_bias), erase gate
    b = sigmoid(be), write gate w = sigmoid(bw), l2-normed q / k, fp32 recurrent state
    S <- S diag(exp g); S <- S + k (w v - (b k) S); o = (q / sqrt(Dk)) S;
  * MLA in absorbed form over the per-token cache the model was trained for (k_rope after RoPE | latent c | 1 / rms
    per KV group): score = scale / rms_g * (q_rope . k_rope + q_lat . c) with q_lat = up_k^T [q_res | q_nope_g], and
    out_h = up_v_g (sum_t p_t c_t): the values of K = [k_rope | up_k c] / rms and V = up_v c without building them.
    The attention is one split-KV kernel over all heads reading the device-side length (csrc/packed_ops.cu).

One token is one captured CUDA graph (the position, the input token and the greedy pick live on the GPU; the position
advances inside the graph), so a generation loop never synchronizes with the host.
"""
import json
import os

import torch
import torch.nn.functional as F

from .packed_kernels import PackedLin, ext

DEV = "cuda"
BF16 = torch.bfloat16
ACH = 32                       # tokens per attention chunk (csrc/packed_ops.cu)


def load_packed(path):
    """config, mercurius spec, {packed base: {field: tensor}}, {dense name: tensor} of a mercurius-packed-v1 export."""
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(path, "config.json")))
    spec = cfg["mercurius"]
    if spec.get("export_format") != "mercurius-packed-v1":
        raise ValueError(f"{path}: not a mercurius-packed-v1 export ({spec.get('export_format')})")
    quant = spec["quant"]
    packed, dense = {}, {}
    with safe_open(os.path.join(path, "model.safetensors"), framework="pt", device="cpu") as f:
        for k in f.keys():
            base, _, field = k.rpartition(".")
            if base in quant:
                packed.setdefault(base, {})[field] = f.get_tensor(k)
            else:
                dense[k] = f.get_tensor(k)
    return cfg, spec, packed, dense


class PackedMercuriusEngine:
    @torch.no_grad()
    def __init__(self, path, Tmax=32768):
        cfg, spec, P, T = load_packed(path)
        self.cfg, self.spec = cfg, spec
        q = spec["quant"]
        lin = lambda *names: PackedLin([P[n] for n in names], q[names[0]]["format"])
        f32 = lambda k: T[k].to(DEV, torch.float32)
        self.H, self.L, self.eps, self.V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["rms_norm_eps"], cfg["vocab_size"]
        self.Tmax = Tmax = (Tmax + ACH - 1) // ACH * ACH
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            s = spec["layers"][str(i)]
            sn = tuple(s.get("scalenorm", ()))
            d = {"ln1": f32(p + "input_layernorm.weight"), "ln2": f32(p + "post_attention_layernorm.weight"),
                 "sc1": "input_layernorm" in sn, "sc2": "post_attention_layernorm" in sn,
                 "gu": lin(p + "mlp.gate_proj", p + "mlp.up_proj"), "down": lin(p + "mlp.down_proj")}
            if "linear_attn" in s:
                g = s["linear_attn"]
                la = p + "linear_attn."
                d.update(kind="gdn2", Hh=g["num_heads"], Hq=g["num_qk_heads"], Dk=g["head_k_dim"], Dv=g["head_v_dim"],
                         gn_eps=g["eps"])
                d["qkv"], d["z"], d["out"] = lin(la + "in_proj_qkv"), lin(la + "in_proj_z"), lin(la + "out_proj")
                d["gates"] = lin(la + "in_proj_a", la + "in_proj_be", la + "in_proj_bw")
                d["conv_w"] = T[la + "conv1d"].to(DEV, BF16)        # the fp32 master, used in bf16 like the module does
                d["A_log"], d["dt_bias"], d["gn"] = f32(la + "A_log"), f32(la + "dt_bias"), f32(la + "norm.weight")
            else:
                a = s["self_attn"]
                sa = p + "self_attn."
                assert a["kind"] == "mla" and a["k_rms_rank"] is None, a
                H, G, D, rd, nr, r = a["num_heads"], a["num_kv_groups"], a["head_dim"], a["rotary_dim"], a["n_rope"], a["rank"]
                assert H == 16 and rd * nr == 64 and r in (384, 512, 640), "mla_lat_attn: 16 heads, rope 64, rank 384/512/640"
                d.update(kind="mla", Hh=H, G=G, D=D, rd=rd, nr=nr, r=r, nres=(G - nr) * rd, gated=a["gated"],
                         k_eps=a["qk_eps"], qn_zc=a["zero_centered"])
                d["qdk"], d["o"] = lin(sa + "q_proj", sa + "down", sa + "k_rms", sa + "k_rope"), lin(sa + "o_proj")
                d["qn"] = f32(sa + "q_norm.weight")
                d["q_map"] = f32(sa + "q_map") if a["per_head_q"] else None
                d["U"] = T[sa + "rorope"].to(DEV, BF16)                       # (H, G, rd/2)
                up_k = lin(sa + "up_k").dequant().float()                     # (nres + G (D - rd), r)
                grp = torch.arange(H, device=DEV) // (H // G)
                d["grp"] = grp
                d["Ures"] = up_k[:d["nres"]].contiguous()                     # (nres, r): shared by every group
                d["Unope_h"] = up_k[d["nres"]:].view(G, D - rd, r)[grp].contiguous()           # per head (H, D-rd, r)
                d["Wv_h"] = lin(sa + "up_v").dequant().float().view(G, D, r)[grp].contiguous()  # per head (H, D, r)
                d["kr"] = torch.zeros(Tmax, nr * rd, device=DEV, dtype=BF16)                # cache: k_rope after RoPE
                d["c"] = torch.zeros(Tmax, r, device=DEV, dtype=BF16)                       #        latent
                d["ir"] = torch.zeros(Tmax, G, device=DEV)                                  #        1 / rms per group
            self.layers.append(d)
        self.norm_f = f32("model.norm.weight")
        assert cfg.get("tie_word_embeddings", False)
        self.emb = self.head = lin("model.embed_tokens")
        rp = cfg["rope_parameters"]
        self.rd = next(d for d in self.layers if d["kind"] == "mla")["rd"]
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (torch.arange(0, self.rd, 2, dtype=torch.int64, device=DEV).float() / self.rd))
        self._alloc()
        self.graph = None

    def _alloc(self):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=DEV)
        self.pos = z(1, dt=torch.long)                    # position of the token being fed
        self.tok = z(1, dt=torch.long)                    # the token being fed
        self.logits = z(1, self.V)
        nch = self.Tmax // ACH
        rmax = max(d["r"] for d in self.layers if d["kind"] == "mla")
        self.pm, self.pl, self.pacc = z(nch * 16), z(nch * 16), z(nch * 16 * rmax)
        self.olat = {d["r"]: z(16, d["r"]) for d in self.layers if d["kind"] == "mla"}
        for d in self.layers:
            if d["kind"] == "gdn2":
                d["conv"] = z(d["qkv"].N, d["conv_w"].shape[1], dt=BF16)
                d["S"] = z(d["Hh"], d["Dk"], d["Dv"])
        self.reset()

    def reset(self):
        self.pos.zero_()
        for d in self.layers:
            if d["kind"] == "gdn2":
                d["conv"].zero_()
                d["S"].zero_()

    # ------------------------------------------------------------------------------------------------ pieces
    def _norm(self, x, w, scalar):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * w).to(BF16) if scalar else (h * (1.0 + w)).to(BF16)

    @staticmethod
    def _rope(x, cos, sin):
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1, x2], -1) * cos + torch.cat([-x2, x1], -1) * sin

    def _gdn2(self, d, x):
        Hh, Hq, Dk, Dv = d["Hh"], d["Hq"], d["Dk"], d["Dv"]
        qkv = d["qkv"](x, 1, torch.empty(1, d["qkv"].N, device=DEV, dtype=BF16))
        st = d["conv"]                                                       # (C, K) raw inputs
        full = torch.cat([st[:, 1:], qkv[0].unsqueeze(-1)], -1)
        st.copy_(full)
        mixed = F.silu((full * d["conv_w"]).sum(-1))                          # (C,) bf16
        kd = Hq * Dk
        q, k, v = mixed[:kd].view(Hq, Dk), mixed[kd:2 * kd].view(Hq, Dk), mixed[2 * kd:].view(Hh, Dv)
        if Hq != Hh:
            q, k = q.repeat_interleave(Hh // Hq, 0), k.repeat_interleave(Hh // Hq, 0)
        gates = d["gates"](x, 1, torch.empty(1, d["gates"].N, device=DEV, dtype=BF16))
        a, be, bw = gates[0].split([Hh * Dk, Hh * Dk, Hh * Dv])
        g = -d["A_log"].exp() * F.softplus(a.view(Hh, Dk).float() + d["dt_bias"])
        b = be.view(Hh, Dk).sigmoid().float()
        w = bw.view(Hh, Dv).sigmoid().float()
        qf = q.float() * torch.rsqrt(q.float().pow(2).sum(-1, keepdim=True) + 1e-6)
        kf = k.float() * torch.rsqrt(k.float().pow(2).sum(-1, keepdim=True) + 1e-6)
        S = d["S"]
        S.mul_(g.exp().unsqueeze(-1))
        erase = torch.einsum("hk,hkv->hv", b * kf, S)
        S.add_(kf.unsqueeze(-1) * (w * v.float() - erase).unsqueeze(-2))
        o = torch.einsum("hk,hkv->hv", qf * Dk ** -0.5, S)                    # (Hh, Dv) fp32
        z = d["z"](x, 1, torch.empty(1, d["z"].N, device=DEV, dtype=BF16))
        ob = o.to(BF16)
        hn = ob.float() * torch.rsqrt(ob.float().pow(2).mean(-1, keepdim=True) + d["gn_eps"])
        hn = d["gn"] * hn.to(BF16)
        return (hn * F.silu(z.view(Hh, Dv).float())).to(BF16).reshape(1, -1)

    def _mla(self, d, x, cos, sin):
        Hh, G, D, rd, nr, r, nres = d["Hh"], d["G"], d["D"], d["rd"], d["nr"], d["r"], d["nres"]
        half = rd // 2
        y = d["qdk"](x, 1, torch.empty(1, d["qdk"].N, device=DEV, dtype=BF16))
        nq = Hh * D * (2 if d["gated"] else 1)
        qg, c, krms, kr = y[0].split([nq, r, G * D, nr * rd])
        qg = qg.view(Hh, -1)
        q, gate = (qg[:, :D], qg[:, D:].reshape(-1)) if d["gated"] else (qg, None)
        qn = q.float() * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + d["k_eps"])
        q = (qn * (1.0 + d["qn"]) if d["qn_zc"] else qn * d["qn"]).to(BF16)
        if d["q_map"] is not None:
            q = torch.einsum("hd,hde->he", q.float(), d["q_map"]).to(BF16)
        inv_r = torch.rsqrt(krms.float().view(G, D).pow(2).sum(-1) / D + d["k_eps"])        # (G,)
        k_rope = self._rope(kr.view(nr, rd), cos, sin).reshape(1, -1)
        U = d["U"]
        qc = torch.cat([q[:, :half].unsqueeze(-2) * U, q[:, half:rd].unsqueeze(-2) * U], -1)  # (H, G, rd)
        q_rope = self._rope(qc[:, :nr], cos, sin).reshape(Hh, nr * rd).float()
        q_lat = qc[:, nr:].reshape(Hh, nres).float() @ d["Ures"] \
            + torch.einsum("he,her->hr", q[:, rd:].float(), d["Unope_h"])                       # (H, r)
        d["kr"].index_copy_(0, self.pos, k_rope)
        d["c"].index_copy_(0, self.pos, c.view(1, -1))
        d["ir"].index_copy_(0, self.pos, inv_r.view(1, -1))
        olat = self.olat[r]
        ext().mla_lat_attn(q_rope.contiguous(), q_lat.contiguous(), d["kr"], d["c"], d["ir"], self.pos, D ** -0.5,
                           self.pm, self.pl, self.pacc, olat)
        out = torch.einsum("hr,hdr->hd", olat, d["Wv_h"]).to(BF16).reshape(1, -1)
        if gate is not None:
            out = out * torch.sigmoid(gate)
        return out

    def _step_body(self):
        """the token in self.tok at position self.pos -> self.logits; then pos += 1 and tok = argmax (greedy)."""
        h = torch.empty(1, self.H, device=DEV, dtype=BF16)
        ext().emb_q4(self.tok, self.emb.codes, self.emb.scales, h, 1)
        ang = self.pos.float() * self.inv_freq
        cos, sin = torch.cat([ang, ang]).cos().to(BF16), torch.cat([ang, ang]).sin().to(BF16)
        for d in self.layers:
            x = self._norm(h, d["ln1"], d["sc1"])
            if d["kind"] == "gdn2":
                h = d["out"](self._gdn2(d, x), 1, torch.empty_like(h), res=h, ldr=self.H)
            else:
                h = d["o"](self._mla(d, x, cos, sin), 1, torch.empty_like(h), res=h, ldr=self.H)
            x = self._norm(h, d["ln2"], d["sc2"])
            gu = d["gu"](x, 1, torch.empty(1, d["gu"].N, device=DEV, dtype=BF16))
            gte, up = gu[0].split(d["gu"].N // 2)
            h = d["down"]((F.silu(gte) * up).reshape(1, -1), 1, torch.empty_like(h), res=h, ldr=self.H)
        self.head(self._norm(h, self.norm_f, False), 1, self.logits)
        self.pos.add_(1)
        self.tok.copy_(self.logits[0].argmax().view(1))

    # ------------------------------------------------------------------------------------------------ API
    @torch.no_grad()
    def capture(self):
        """record one decode step as a CUDA graph (state is restored afterwards)."""
        saved = (self.pos.clone(), self.tok.clone(),
                 [(d["conv"].clone(), d["S"].clone()) for d in self.layers if d["kind"] == "gdn2"])
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._step_body()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step_body()
        self.pos.copy_(saved[0]); self.tok.copy_(saved[1])
        for d, (cv, S) in zip([d for d in self.layers if d["kind"] == "gdn2"], saved[2]):
            d["conv"].copy_(cv); d["S"].copy_(S)

    @torch.no_grad()
    def step(self, token=None):
        """feed one token (or, with token=None, the previous greedy pick) -> logits (V,) fp32 (a view, overwritten
        by the next step)."""
        if token is not None:
            self.tok.fill_(token)
        if self.graph is not None:
            self.graph.replay()
        else:
            self._step_body()
        return self.logits[0]

    @torch.no_grad()
    def generate(self, prompt, n, eos=None):
        """greedy: feed the prompt token by token, then n tokens; returns the generated ids."""
        self.reset()
        for t in prompt:
            self.step(t)
        out = torch.empty(n, dtype=torch.long, device=DEV)
        for i in range(n):
            out[i] = self.tok[0]
            self.step()
        ids = out.tolist()
        if eos is not None and eos in ids:
            ids = ids[:ids.index(eos) + 1]
        return ids
