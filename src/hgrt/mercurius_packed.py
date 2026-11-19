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

One token (M = 1), or one speculative verification block of M = 1 + k tokens, is one captured CUDA graph. The rows of a
block share every weight read (packed GEMVs over M rows, bit-identical per row); everything with a reduction runs per
row with exactly the single-token shapes, and GDN-2 writes each row's state to its own slot. So a verification pass
produces, for every row, the logits plain decoding would, and accepting the longest draft prefix that matches the
greedy picks is lossless: speculative and plain greedy decoding emit the same tokens.

Drafts come from the checkpoint's MTP head (mtp.*: fc over [norm(emb(x_{t+1})); norm(h_t)], one gated GQA layer with
its own KV cache, the family MLP, norm, tied head restricted to the first `draft_vocab` ids), filled at every accepted
position and run recursively for k drafts.
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
    def __init__(self, path, Tmax=32768, maxM=4, draft_vocab=32768, kv=None):
        cfg, spec, P, T = load_packed(path)
        self.cfg, self.spec = cfg, spec
        kvc = spec.get("kv_cache") or {"format": "none"}
        self.kv = kv or ("foveakv" if kvc.get("format") == "foveakv" else "exact")   # default: as trained
        if self.kv == "foveakv" and kvc.get("format") != "foveakv":
            raise ValueError("kv='foveakv' needs a checkpoint trained with it (mercurius.kv_cache)")
        self.kvc = kvc
        q = spec["quant"]
        lin = lambda *names: PackedLin([P[n] for n in names], q[names[0]]["format"],
                                       R=4 if q[names[0]]["format"] == "q4_0" else 2)      # rows per warp (bench)
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
                assert d["Hq"] == d["Hh"] and d["Dk"] == d["Dv"] == 128, "gdn2_step: expanded q/k heads, Dk = Dv = 128"
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
                if self.kv == "foveakv":
                    from .foveakv_mla import FoveaCodec
                    k = self.kvc
                    d["cod_c"] = FoveaCodec(r, k["head_dim"], k["block"], k["gain_bits"], k["shrink"], k["seed"], ("k", "v"))
                    d["cod_r"] = FoveaCodec(nr * rd, k["head_dim"], k["block"], k["gain_bits"], k["shrink"], k["seed"], ("k",))
                    zb = lambda w: torch.zeros(Tmax, w, device=DEV, dtype=BF16)
                    d["tc"] = {("f", "k"): zb(r), ("c", "k"): zb(r), ("f", "v"): zb(r), ("c", "v"): zb(r)}
                    d["tr"] = {("f", "k"): zb(nr * rd), ("c", "k"): zb(nr * rd)}
                    d["tiers"] = [d["kr"], d["c"], d["tr"]["f", "k"], d["tc"]["f", "k"], d["tc"]["f", "v"],
                                  d["tr"]["c", "k"], d["tc"]["c", "k"], d["tc"]["c", "v"]]
            self.layers.append(d)
        self.norm_f = f32("model.norm.weight")
        assert cfg.get("tie_word_embeddings", False)
        self.emb = self.head = lin("model.embed_tokens")
        rp = cfg["rope_parameters"]
        self.rd = next(d for d in self.layers if d["kind"] == "mla")["rd"]
        self.inv_freq = 1.0 / (rp["rope_theta"] ** (torch.arange(0, self.rd, 2, dtype=torch.int64, device=DEV).float() / self.rd))
        self.maxM = maxM
        self.mtp = None
        if "mtp.fc" in q:
            m = {"fc": lin("mtp.fc"), "pre_e": f32("mtp.pre_fc_norm_embedding.weight"),
                 "pre_h": f32("mtp.pre_fc_norm_hidden.weight"), "ln1": f32("mtp.layer.input_layernorm.weight"),
                 "ln2": f32("mtp.layer.post_attention_layernorm.weight"), "norm": f32("mtp.norm.weight"),
                 "qkv": lin("mtp.layer.self_attn.q_proj", "mtp.layer.self_attn.k_proj", "mtp.layer.self_attn.v_proj"),
                 "o": lin("mtp.layer.self_attn.o_proj"), "qn": f32("mtp.layer.self_attn.q_norm.weight"),
                 "kn": f32("mtp.layer.self_attn.k_norm.weight"),
                 "gu": lin("mtp.layer.mlp.gate_proj", "mtp.layer.mlp.up_proj"), "down": lin("mtp.layer.mlp.down_proj")}
            m["Hh"], m["G"], m["D"] = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
            m["K"] = torch.zeros(self.Tmax, m["G"], m["D"], device=DEV, dtype=BF16)
            m["Vc"] = torch.zeros(self.Tmax, m["G"], m["D"], device=DEV, dtype=BF16)
            dv = min(draft_vocab or self.V, self.V)
            m["head"] = PackedLin.__new__(PackedLin)                 # the tied head's first dv rows (a view, no copy)
            m["head"].__dict__.update(self.head.__dict__)
            m["head"].codes, m["head"].scales, m["head"].N = self.head.codes[:dv], self.head.scales[:dv], dv
            self.mtp = m
        self._alloc()
        self.graphs = {}

    def _alloc(self):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=DEV)
        M = self.maxM
        self.pos = z(1, dt=torch.long)                    # position of the first token of the next block
        self.toks = z(M, dt=torch.long)                   # tokens fed at pos .. pos + M - 1
        self.posM = z(M, dt=torch.long)                   # their positions
        self.pick = z(M, dt=torch.long)                   # greedy pick after each of them
        self.logits = z(M, self.V)
        self.hn = z(M, self.H, dt=BF16)                   # normed final hidden per row (the MTP head's input)
        nch = self.Tmax // ACH
        rmax = max(d["r"] for d in self.layers if d["kind"] == "mla")
        self.pm, self.pl, self.pacc = z(2 * nch * 16), z(2 * nch * 16), z(2 * nch * 16 * max(rmax, 256))  # 16-token chunks
        self.fovea_on = False                             # tiered reads: after the prompt (calibrate_fovea)
        if self.kv == "foveakv":
            self.f_sc, self.f_lse = z(16 * self.Tmax), z(16)
            self.f_lp, self.f_thr = z(self.Tmax), z(1)
        self.olat = {d["r"]: z(16, d["r"]) for d in self.layers if d["kind"] == "mla"}
        for d in self.layers:
            if d["kind"] == "gdn2":
                d["conv"] = z(d["qkv"].N, d["conv_w"].shape[1], dt=BF16)
                d["S"] = z(d["Hh"], d["Dk"], d["Dv"])
                d["conv_slots"] = z(M, d["qkv"].N, d["conv_w"].shape[1], dt=BF16)
                d["S_slots"] = z(M, d["Hh"], d["Dk"], d["Dv"])
                d["b_qkv"], d["b_gates"] = z(M, d["qkv"].N, dt=BF16), z(M, d["gates"].N, dt=BF16)
                d["b_z"], d["b_o"] = z(M, d["z"].N, dt=BF16), z(M, d["Hh"] * d["Dv"], dt=BF16)
            else:
                d["b_qdk"] = z(M, d["qdk"].N, dt=BF16)
                d["b_o"] = z(M, d["Hh"] * d["D"], dt=BF16)
        self.b_h = z(M, self.H, dt=BF16)
        self.b_x = z(M, self.H, dt=BF16)
        FFmax = max(d["gu"].N for d in self.layers)
        self.b_gu, self.b_act = z(M, FFmax, dt=BF16), z(M, FFmax // 2, dt=BF16)
        if self.mtp is not None:
            m = self.mtp
            m["b_in"], m["b_z"] = z(M, 2 * self.H, dt=BF16), z(M, self.H, dt=BF16)
            m["b_qkv"], m["b_o"] = z(M, m["qkv"].N, dt=BF16), z(M, m["Hh"] * m["D"], dt=BF16)
            m["b_gu"], m["b_act"] = z(M, m["gu"].N, dt=BF16), z(M, m["gu"].N // 2, dt=BF16)
            m["b_out"], m["logits"] = z(M, self.H, dt=BF16), z(1, m["head"].N)
            m["att"] = z(m["Hh"], m["D"])
            m["pos"] = z(1, dt=torch.long)
        self.reset()

    def reset(self):
        self.pos.zero_()
        self.fovea_on = False
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

    def _cos_sin(self, pos1):
        ang = pos1.float() * self.inv_freq
        return torch.cat([ang, ang]).cos().to(BF16), torch.cat([ang, ang]).sin().to(BF16)

    def _gdn2(self, d, x, M, slots):
        qkv = d["qkv"](x, M, d["b_qkv"])
        gates = d["gates"](x, M, d["b_gates"])
        z = d["z"](x, M, d["b_z"])
        for i in range(M):
            if slots:                                            # row i: state of row i - 1 -> slot i
                ci, si = (d["conv"], d["S"]) if i == 0 else (d["conv_slots"][i - 1], d["S_slots"][i - 1])
                co, so = d["conv_slots"][i], d["S_slots"][i]
            else:
                ci = co = d["conv"]
                si = so = d["S"]
            ext().gdn2_step(qkv[i], ci, co, d["conv_w"], gates[i], d["A_log"], d["dt_bias"], si, so, z[i], d["gn"],
                            d["gn_eps"], d["b_o"][i], d["Hh"], d["Hq"])
        return d["b_o"]

    def _mla_row(self, d, y, pos1, cos, sin, out):
        """one row: y = this token's fused [q | gate | c | k_rms | k_rope] projection -> out (H D) bf16."""
        Hh, G, D, rd, nr, r, nres = d["Hh"], d["G"], d["D"], d["rd"], d["nr"], d["r"], d["nres"]
        half = rd // 2
        nq = Hh * D * (2 if d["gated"] else 1)
        qg, c, krms, kr = y.split([nq, r, G * D, nr * rd])
        qg = qg.view(Hh, -1)
        q, gate = (qg[:, :D], qg[:, D:].reshape(-1)) if d["gated"] else (qg, None)
        qn = q.float() * torch.rsqrt(q.float().pow(2).mean(-1, keepdim=True) + d["k_eps"])
        q = (qn * (1.0 + d["qn"]) if d["qn_zc"] else qn * d["qn"]).to(BF16)
        if d["q_map"] is not None:
            q = torch.einsum("hd,hde->he", q.float(), d["q_map"]).to(BF16)
        inv_r = torch.rsqrt(krms.float().view(G, D).pow(2).sum(-1) / D + d["k_eps"])        # (G,)
        if self.kv == "foveakv":
            inv_r = inv_r.half().float()                                                        # the cache stores it in fp16
        k_rope = self._rope(kr.view(nr, rd), cos, sin).reshape(1, -1)
        U = d["U"]
        qc = torch.cat([q[:, :half].unsqueeze(-2) * U, q[:, half:rd].unsqueeze(-2) * U], -1)  # (H, G, rd)
        q_rope = self._rope(qc[:, :nr], cos, sin).reshape(Hh, nr * rd).float()
        q_lat = qc[:, nr:].reshape(Hh, nres).float() @ d["Ures"] \
            + torch.einsum("he,her->hr", q[:, rd:].float(), d["Unope_h"])                       # (H, r)
        d["kr"].index_copy_(0, pos1, k_rope)
        d["c"].index_copy_(0, pos1, c.view(1, -1))
        d["ir"].index_copy_(0, pos1, inv_r.view(1, -1))
        olat = self.olat[r]
        if self.fovea_on:
            k = self.kvc
            s_enc = (pos1 - k["window"]).clamp_min(0)                  # the token leaving the exact window now
            d["cod_c"].encode(d["c"], s_enc, d["tc"])
            d["cod_r"].encode(d["kr"], s_enc, d["tr"])
            ext().fovea_attn(q_rope.contiguous(), q_lat.contiguous(), pos1, D ** -0.5, k["sinks"], k["window"],
                             k["hot_frac"], d["ir"], d["tiers"], self.f_sc, self.f_lse, self.f_lp, self.f_thr,
                             self.pm, self.pl, self.pacc, olat)
        else:
            ext().mla_lat_attn(q_rope.contiguous(), q_lat.contiguous(), d["kr"], d["c"], d["ir"], pos1, D ** -0.5,
                               self.pm, self.pl, self.pacc, olat)
        o = torch.einsum("hr,hdr->hd", olat, d["Wv_h"]).to(BF16).reshape(-1)
        out.copy_(o * torch.sigmoid(gate) if gate is not None else o)

    def _rows(self, M, slots):
        """the M tokens in self.toks at positions self.posM -> self.logits[:M], self.hn[:M], self.pick[:M]."""
        h = self.b_h[:M]
        ext().emb_q4(self.toks, self.emb.codes, self.emb.scales, h, M)
        cs = [self._cos_sin(self.posM[i:i + 1]) for i in range(M)]
        x = self.b_x[:M]
        for d in self.layers:
            for i in range(M):
                x[i:i + 1].copy_(self._norm(h[i:i + 1], d["ln1"], d["sc1"]))
            if d["kind"] == "gdn2":
                o = self._gdn2(d, x, M, slots)
                d["out"](o, M, h, res=h, ldr=self.H)
            else:
                y = d["qdk"](x, M, d["b_qdk"])
                for i in range(M):
                    self._mla_row(d, y[i], self.posM[i:i + 1], *cs[i], d["b_o"][i])
                d["o"](d["b_o"], M, h, res=h, ldr=self.H)
            for i in range(M):
                x[i:i + 1].copy_(self._norm(h[i:i + 1], d["ln2"], d["sc2"]))
            F2 = d["gu"].N // 2
            gu = d["gu"](x, M, self.b_gu[:M, :2 * F2], ldy=self.b_gu.shape[1])
            act = self.b_act[:M, :F2]
            act.copy_(F.silu(gu[:, :F2]) * gu[:, F2:2 * F2])
            d["down"](act, M, h, res=h, ldr=self.H, ldx=self.b_act.shape[1])
        for i in range(M):
            self.hn[i:i + 1].copy_(self._norm(h[i:i + 1], self.norm_f, False))
        self.head(self.hn, M, self.logits)
        for i in range(M):
            self.pick[i:i + 1].copy_(self.logits[i].argmax().view(1))

    # ------------------------------------------------------------------------------------------------ MTP head
    def _rms(self, x, w):
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return (h * (1.0 + w)).to(BF16)

    def _mtp(self, M, h, toks, pos0):
        """MTP head over M rows: (normed hidden h[i], token toks[i]) at positions pos0 + i -> m["b_out"][:M] (normed
        output, the next recursion's hidden); writes the MTP KV cache; draft logits of the LAST row in m["logits"]."""
        m = self.mtp
        e = torch.empty(M, self.H, device=DEV, dtype=BF16)
        ext().emb_q4(toks, self.emb.codes, self.emb.scales, e, M)
        inp = m["b_in"][:M]
        for i in range(M):
            inp[i:i + 1, :self.H].copy_(self._rms(e[i:i + 1], m["pre_e"]))
            inp[i:i + 1, self.H:].copy_(self._rms(h[i:i + 1], m["pre_h"]))
        z = m["fc"](inp, M, m["b_z"][:M])
        Hh, G, D = m["Hh"], m["G"], m["D"]
        x = torch.empty(M, self.H, device=DEV, dtype=BF16)
        for i in range(M):
            x[i:i + 1].copy_(self._rms(z[i:i + 1], m["ln1"]))
        qkv = m["qkv"](x, M, m["b_qkv"][:M])
        for i in range(M):
            p1 = pos0 + i
            m["pos"].fill_(p1)
            cos, sin = self._cos_sin(m["pos"])
            qg, k, v = qkv[i].split([Hh * 2 * D, G * D, G * D])
            qg = qg.view(Hh, 2 * D)
            q, gate = qg[:, :D], qg[:, D:].reshape(-1)
            q = self._rms(q, m["qn"])
            k = self._rms(k.view(G, D), m["kn"])
            rd = self.rd
            q = torch.cat([self._rope(q[:, :rd], cos, sin), q[:, rd:]], -1)
            k = torch.cat([self._rope(k[:, :rd], cos, sin), k[:, rd:]], -1)
            m["K"][p1].copy_(k)
            m["Vc"][p1].copy_(v.view(G, D))
            ext().gqa_attn(q.float().contiguous(), m["K"], m["Vc"], m["pos"], D ** -0.5, self.pm, self.pl, self.pacc, m["att"])
            m["b_o"][i].copy_((m["att"].to(BF16).reshape(-1) * torch.sigmoid(gate)))
        z = m["o"](m["b_o"][:M], M, z, res=z, ldr=self.H)
        for i in range(M):
            x[i:i + 1].copy_(self._rms(z[i:i + 1], m["ln2"]))
        F2 = m["gu"].N // 2
        gu = m["gu"](x, M, m["b_gu"][:M])
        act = m["b_act"][:M]
        act.copy_(F.silu(gu[:, :F2]) * gu[:, F2:])
        z = m["down"](act, M, z, res=z, ldr=self.H)
        out = m["b_out"][:M]
        for i in range(M):
            out[i:i + 1].copy_(self._rms(z[i:i + 1], m["norm"]))
        m["head"](out[M - 1:M], 1, m["logits"])
        return int(m["logits"][0].argmax())

    # ------------------------------------------------------------------------------------------------ API
    @torch.no_grad()
    def calibrate_fovea(self):
        """after the prompt (positions 0 .. P-1, attended exactly): calibrate every layer's gain range on the prompt and
        encode the prompt tokens the first decode query already reads quantized; later queries read tiers."""
        if self.kv != "foveakv":
            return
        P, k = int(self.pos), self.kvc
        for d in self.layers:
            if d["kind"] != "mla":
                continue
            d["cod_c"].calibrate(d["c"], P)
            d["cod_r"].calibrate(d["kr"], P)
            if P - k["window"] >= 0:
                s = torch.arange(0, P - k["window"] + 1, device=DEV)
                d["cod_c"].encode(d["c"], s, d["tc"])
                d["cod_r"].encode(d["kr"], s, d["tr"])
        self.fovea_on = True

    def _graph(self, M, slots):
        key = (M, slots, self.fovea_on)
        if key not in self.graphs:
            snap = self._snapshot()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._rows(M, slots)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._rows(M, slots)
            self._restore(snap)
            self.graphs[key] = g
        return self.graphs[key]

    def _snapshot(self):
        return [(d["conv"].clone(), d["S"].clone()) for d in self.layers if d["kind"] == "gdn2"]

    def _restore(self, snap):
        for d, (cv, S) in zip([d for d in self.layers if d["kind"] == "gdn2"], snap):
            d["conv"].copy_(cv); d["S"].copy_(S)

    @torch.no_grad()
    def capture(self, ks=(0,)):
        """record the decode step (k = 0) and the verification blocks for k drafts as CUDA graphs (for the prompt's
        exact attention and, with FoveaKV, the tiered decode)."""
        modes = (False, True) if self.kv == "foveakv" else (False,)
        saved = self.fovea_on
        for on in modes:
            self.fovea_on = on
            for k in ks:
                self._graph(k + 1, k > 0)
        self.fovea_on = saved

    @torch.no_grad()
    def _run(self, toks, use_graph=True, slots=False):
        M = len(toks)
        self.toks[:M].copy_(torch.tensor(toks, device=DEV))
        self.posM[:M].copy_(self.pos + torch.arange(M, device=DEV))
        if use_graph and (M, slots, self.fovea_on) in self.graphs:
            self.graphs[(M, slots, self.fovea_on)].replay()
        else:
            self._rows(M, slots)

    @torch.no_grad()
    def step(self, token):
        """feed one token at self.pos -> logits (V,) fp32 (overwritten by the next call)."""
        self._run([token])
        self.pos.add_(1)
        return self.logits[0]

    @torch.no_grad()
    def generate(self, prompt, n, eos=None, k=0):
        """greedy decoding: the prompt, then n tokens. k > 0: speculative with k MTP drafts per block (lossless: the
        same tokens as k = 0). Returns (ids, stats)."""
        self.reset()
        mtp = k > 0 and self.mtp is not None
        out, blocks, accepted = [], 0, 0
        for i, t in enumerate(prompt):
            self._run([t])
            if mtp and i + 1 < len(prompt):                      # MTP cache entry i: (h_i, x_{i+1})
                self._mtp(1, self.hn[:1].clone(), torch.tensor([prompt[i + 1]], device=DEV), i)
            self.pos.add_(1)
        x = int(self.pick[0])
        self.calibrate_fovea()
        if not mtp:
            while len(out) < n:
                out.append(x)
                if eos is not None and x == eos:
                    break
                self._run([x])
                self.pos.add_(1)
                x = int(self.pick[0])
            return out[:n], {}
        drafts = self._draft(self.hn[:1].clone(), x, len(prompt) - 1, k)   # MTP at the last prompt position
        out.append(x)                                                # the first generated token
        while len(out) < n and not (eos is not None and x == eos):
            block = [x] + drafts
            self._run(block, slots=True)
            y = self.pick[:len(block)].tolist()
            nacc = 0
            while nacc < len(drafts) and drafts[nacc] == y[nacc]:
                nacc += 1
            emit = y[:nacc + 1]
            blocks += 1
            accepted += nacc
            done = False
            for t in emit:
                if len(out) >= n:
                    break
                out.append(t)
                if eos is not None and t == eos:
                    done = True
                    break
            for d in self.layers:                                     # commit the state after row nacc
                if d["kind"] == "gdn2":
                    d["conv"].copy_(d["conv_slots"][nacc])
                    d["S"].copy_(d["S_slots"][nacc])
            hrows = self.hn[:nacc + 1].clone()
            base = int(self.pos)
            self.pos.add_(nacc + 1)
            if done or len(out) >= n:
                break
            x = emit[-1]
            drafts = self._draft(hrows, x, base, k, fill_toks=emit)
        return out[:n], {"blocks": blocks, "accepted": accepted, "drafts": blocks * k,
                         "acceptance": accepted / max(blocks * k, 1), "tokens_per_block": (len(out) - 1) / max(blocks, 1)}

    def _draft(self, hrows, x, base, k, fill_toks=None):
        """fill the MTP cache at positions base .. base + R - 1 with (hrows[i], token after row i) and draft k tokens."""
        R = hrows.shape[0]
        nxt = fill_toks if fill_toks is not None else [x]
        d1 = self._mtp(R, hrows, torch.tensor(nxt, device=DEV), base)
        drafts = [d1]
        hcur = self.mtp["b_out"][R - 1:R].clone()
        for j in range(1, k):
            d = self._mtp(1, hcur, torch.tensor([drafts[-1]], device=DEV), base + R - 1 + j)
            drafts.append(d)
            hcur = self.mtp["b_out"][:1].clone()
        return drafts
