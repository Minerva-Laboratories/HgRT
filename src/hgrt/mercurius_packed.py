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

from .spec_policy import now


def _vcost_fn(vab):
    """V(M, T) from the calibrated (A_M, B_M), without a reference to the engine (no cycle: a deleted engine frees its
    GPU memory at once)."""
    return lambda M, T: vab[M][0] + vab[M][1] * T
from .packed_kernels import PackedLin, ext

DEV = "cuda"
BF16 = torch.bfloat16
GDN_PREFILL = os.environ.get("HGRT_GDN_PREFILL", "chunk")   # "step": prefill GDN-2 token by token
ACH = 32                       # tokens per attention chunk (csrc/packed_ops.cu)
NS = 64                        # split blocks per row of the decode attentions (fixed: rows identical for any M)


def _fovea_splits():
    """FoveaKV decode attention splits per row: the largest multiple of the SM count <= NS (one block per SM, so whole
    waves; 64 on 20 SMs leaves a last wave of 4). Fixed per GPU, never per M: rows stay identical for any M."""
    sms = torch.cuda.get_device_properties(DEV).multi_processor_count
    return max(sms, NS // sms * sms) if sms <= NS else NS


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
    def __init__(self, path, Tmax=32768, maxM=8, draft_vocab=32768, kv=None):
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
                d["U"] = T[sa + "rorope"].to(DEV, BF16).contiguous()          # (H, G, rd/2)
                up_k = lin(sa + "up_k").dequant()                             # (nres + G (D - rd), r) bf16 (exact values)
                d["Ures"] = up_k[:d["nres"]].contiguous()                     # (nres, r): shared by every group
                d["Unope"] = up_k[d["nres"]:].view(G, D - rd, r).contiguous()  # per group (G, D-rd, r)
                d["WvT"] = lin(sa + "up_v").dequant().view(G, D, r).transpose(1, 2).contiguous()   # (G, r, D)
                Te = Tmax if self.kv == "exact" else 0       # FoveaKV: the exact cache holds the prompt only (prefill)
                d["kr"] = torch.zeros(Te, nr * rd, device=DEV, dtype=BF16)                  # cache: k_rope after RoPE
                d["c"] = torch.zeros(Te, r, device=DEV, dtype=BF16)                         #        latent
                d["ir"] = torch.zeros(Te, G, device=DEV)                                    #        1 / rms per group
                if self.kv == "foveakv":
                    from .foveakv_mla import FoveaCodec
                    k = self.kvc
                    d["cod_c"] = FoveaCodec(r, k["head_dim"], k["block"], k["gain_bits"], k["shrink"], k["seed"], ("k", "v"))
                    d["cod_r"] = FoveaCodec(nr * rd, k["head_dim"], k["block"], k["gain_bits"], k["shrink"], k["seed"], ("k",))
                    u8 = lambda *sh: torch.zeros(*sh, device=DEV, dtype=torch.uint8)
                    nb, RW = -(-Tmax // k["block"]), 2 * k["block"] + maxM   # blocks (the last partial too); ring: the encoder's block span + M rows ahead
                    d["f_cc"], d["f_gc"] = u8(Tmax, r // 2), u8(Tmax, r // k["head_dim"])          # codes, per token
                    d["f_cr"], d["f_gr"] = u8(Tmax, nr * rd // 2), u8(Tmax)
                    d["f_Mc"] = torch.zeros(nb, r, device=DEV, dtype=torch.float16)               # block means
                    d["f_Mr"] = torch.zeros(nb, nr * rd, device=DEV, dtype=torch.float16)
                    d["f_ir"] = torch.zeros(Tmax, G, device=DEV, dtype=torch.float16)
                    d["f_sink_c"] = torch.zeros(k["sinks"], r, device=DEV, dtype=BF16)            # exact: sinks, ring
                    d["f_sink_r"] = torch.zeros(k["sinks"], nr * rd, device=DEV, dtype=BF16)
                    d["f_ring_c"] = torch.zeros(RW, r, device=DEV, dtype=BF16)
                    d["f_ring_r"] = torch.zeros(RW, nr * rd, device=DEV, dtype=BF16)
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
        rmax = max(d["r"] for d in self.layers if d["kind"] == "mla")
        self.pm, self.pl, self.pacc = z(M * NS * 16), z(M * NS * 16), z(M * NS * 16 * max(rmax, 256))   # split partials
        self.fovea_on = False                             # tiered reads: after the prompt (calibrate_fovea)
        self._E = z(0)                                    # "absent" for optional kernel arguments
        if self.kv == "foveakv":
            from .foveakv_mla import nested_codebook, rht
            self.f_sc, self.f_lse = z(M * 16 * self.Tmax), z(M * 16)      # per row: scores, logsumexp per head,
            self.f_lp, self.f_thr = z(M * self.Tmax), z(M)                # ranking key, hot threshold
            self.f_pax, self.f_par = z(M * NS * 16 * rmax), z(M * NS * 16 * rmax)
            self.nsf = _fovea_splits()                     # <= NS: fits the split buffers
            k = self.kvc
            bnd16, lut16, lut4, Df, Dc = nested_codebook()
            ext().fovea_set_luts(lut16.to(DEV, torch.float32), lut4.to(DEV, torch.float32), bnd16.to(DEV, torch.float32))
            self.f_R = rht(k["head_dim"], k["seed"]).to(DEV).float().contiguous()
            self.f_R64 = rht(64, k["seed"]).to(DEV).float().contiguous()
            sk = lambda D, role: ((1 - D) ** -0.5 if role == "k" else 1.0 / (1 - D)) if k["shrink"] else 1.0
            self.f_shr = [sk(Df, "k"), sk(Df, "v"), sk(Dc, "k"), sk(Dc, "v"), sk(Df, "k"), sk(Dc, "k")]
            self.Pcap = 0
        self.b_qr, self.b_qrr = z(M, 16, 64), z(M, 16, 64)               # per row: q_rope (and FoveaKV-rotated)
        self.b_ql, self.b_qlr = z(M * 16 * rmax), z(M * 16 * rmax)       #          q_lat, viewed (M, 16, r)
        self.b_olat = z(M * 16 * rmax)                                   #          attention output (latent)
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
            m["att"] = z(M, m["Hh"], m["D"])
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
    def _norm(self, x, w, scalar, out=None):
        """RMSNorm of every row of x (M, H) in one kernel (ScaleNorm if scalar, else zero-centred)."""
        out = torch.empty_like(x) if out is None else out
        ext().rms_rows(x, x.stride(0), w, scalar, self.eps, out, out.stride(0), x.shape[0], x.shape[1])
        return out

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
        if slots:                                                # row i: state of row i - 1 -> slot i
            co, so = d["conv_slots"], d["S_slots"]
            cst, sst = co[0].numel(), so[0].numel()
        else:                                                    # plain decoding: one row, in place
            assert M == 1
            co, so, cst, sst = d["conv"], d["S"], 0, 0
        ext().gdn2_step(qkv, d["conv"], co, cst, d["conv_w"], gates, d["A_log"], d["dt_bias"], d["S"], so, sst, z, d["gn"],
                        d["gn_eps"], d["b_o"], d["Hh"], d["Hq"], M)
        return d["b_o"]

    def _mla(self, d, x, M):
        """the MLA layer over M rows: fused projections, query / key preparation and cache writes for all rows (one
        launch), attention for all rows (row i sees positions <= its own), then the gated output for all rows."""
        y = d["qdk"](x, M, d["b_qdk"])
        r, Hh = d["r"], d["Hh"]
        fv = self.fovea_on
        ql = self.b_ql[:M * Hh * r].view(M, Hh, r)
        qlr = self.b_qlr[:M * Hh * r].view(M, Hh, r)
        olat = self.b_olat[:M * Hh * r].view(M, Hh, r)
        E = self._E
        cache = (d["f_ring_r"], d["f_ring_c"], d["f_ir"]) if fv else (d["kr"], d["c"], d["ir"])
        ext().mla_prep(y, y.stride(0), M,
                       [self.posM, self.inv_freq, d["qn"], E if d["q_map"] is None else d["q_map"], d["U"], d["Ures"],
                        d["Unope"], self.b_qr, ql, self.b_qrr if fv else E, qlr if fv else E,
                        self.f_R if fv else E, self.f_R64 if fv else E, *cache],
                       [Hh, d["G"], d["D"], d["rd"], d["nr"], r, int(d["gated"]), int(d["qn_zc"]), int(fv),
                        int(self.kv == "foveakv")], d["k_eps"])
        if fv:
            k = self.kvc
            ext().fovea_encode([d["f_ring_c"], self.f_R, d["cod_c"].lo, d["cod_c"].span, d["f_cc"], d["f_gc"], d["f_Mc"]],
                               [d["f_ring_r"], self.f_R64, d["cod_r"].lo, d["cod_r"].span, d["f_cr"], d["f_gr"], d["f_Mr"]],
                               self.posM, M, [k["window"], k["block"], k["gain_bits"]])
        if fv:                                         # every row at once (row i reads positions <= its own)
            self._fovea_attend(d, self.b_qr[:M], ql, self.b_qrr[:M], qlr, M, olat)
        else:
            ext().mla_lat_attn(self.b_qr[:M], ql, d["kr"], d["c"], d["ir"], self.posM, M, NS, d["D"] ** -0.5,
                               self.pm, self.pl, self.pacc, olat)
        ext().mla_out(olat, d["WvT"], y, y.stride(0), M, Hh, bool(d["gated"]), d["b_o"], d["b_o"].stride(0))
        return d["b_o"]

    def _rows(self, M, slots):
        """the M tokens in self.toks at positions self.posM -> self.logits[:M], self.hn[:M], self.pick[:M]."""
        h = self.b_h[:M]
        ext().emb_q4(self.toks, self.emb.codes, self.emb.scales, h, M)
        x = self.b_x[:M]
        for d in self.layers:
            self._norm(h, d["ln1"], d["sc1"], x)
            if d["kind"] == "gdn2":
                o = self._gdn2(d, x, M, slots)
                d["out"](o, M, h, res=h, ldr=self.H)
            else:
                d["o"](self._mla(d, x, M), M, h, res=h, ldr=self.H)
            self._norm(h, d["ln2"], d["sc2"], x)
            F2 = d["gu"].N // 2
            gu = d["gu"](x, M, self.b_gu[:M, :2 * F2], ldy=self.b_gu.shape[1])
            act = self.b_act[:M, :F2]
            act.copy_(F.silu(gu[:, :F2]) * gu[:, F2:2 * F2])
            d["down"](act, M, h, res=h, ldr=self.H, ldx=self.b_act.shape[1])
        self._norm(h, self.norm_f, False, self.hn[:M])
        self.head(self.hn, M, self.logits)
        torch.argmax(self.logits[:M], -1, out=self.pick[:M])

    # ------------------------------------------------------------------------------------------------ MTP head
    def _rms(self, x, w, out=None):
        """zero-centred RMSNorm over the last dim of x (rows, n), one kernel."""
        x2 = x.reshape(-1, x.shape[-1])
        o = torch.empty_like(x2) if out is None else out
        ext().rms_rows(x2, x2.stride(0), w, False, self.eps, o, o.stride(0), x2.shape[0], x2.shape[1])
        return o.view_as(x) if out is None else o

    def _mtp(self, M, h, toks, pos0, fill_only=False, with_p=False):
        """MTP head over M rows: (normed hidden h[i], token toks[i]) at positions pos0 + i -> m["b_out"][:M] (normed
        output, the next recursion's hidden); writes the MTP KV cache; draft logits of the LAST row in m["logits"]."""
        m = self.mtp
        B = lambda key: m[key][:M] if M <= self.maxM else torch.empty(M, *m[key].shape[1:], device=DEV, dtype=BF16)
        e = torch.empty(M, self.H, device=DEV, dtype=BF16)
        ext().emb_q4(toks, self.emb.codes, self.emb.scales, e, M)
        inp = B("b_in")
        self._rms(e, m["pre_e"], inp[:, :self.H])
        self._rms(h, m["pre_h"], inp[:, self.H:])
        z = m["fc"](inp, M, B("b_z"))
        Hh, G, D = m["Hh"], m["G"], m["D"]
        x = torch.empty(M, self.H, device=DEV, dtype=BF16)
        self._rms(z, m["ln1"], x)
        qkv = m["qkv"](x, M, B("b_qkv"))
        pos = torch.arange(pos0, pos0 + M, device=DEV)
        qg, k, v = qkv.split([Hh * 2 * D, G * D, G * D], -1)
        qg = qg.reshape(M, Hh, 2 * D)
        rd = self.rd
        ang = pos[:, None].float() * self.inv_freq
        cos, sin = (f(torch.cat([ang, ang], -1)).to(BF16)[:, None] for f in (torch.cos, torch.sin))
        k = self._rms(k.reshape(M * G, D), m["kn"]).view(M, G, D)
        k = torch.cat([self._rope(k[..., :rd], cos, sin), k[..., rd:]], -1)
        m["K"].index_copy_(0, pos, k)
        m["Vc"].index_copy_(0, pos, v.reshape(M, G, D))
        if fill_only:                              # prefill: only the cache entries are needed
            return None
        q = self._rms(qg[..., :D].reshape(M * Hh, D), m["qn"]).view(M, Hh, D)
        q = torch.cat([self._rope(q[..., :rd], cos, sin), q[..., rd:]], -1)
        att = m["att"][:M]
        ext().gqa_attn(q.float().contiguous(), m["K"], m["Vc"], pos, M, NS, D ** -0.5, self.pm, self.pl, self.pacc, att)
        m["b_o"][:M].copy_(att.to(BF16).reshape(M, -1) * torch.sigmoid(qg[..., D:].reshape(M, -1)))
        z = m["o"](m["b_o"][:M], M, z, res=z, ldr=self.H)
        self._rms(z, m["ln2"], x)
        F2 = m["gu"].N // 2
        gu = m["gu"](x, M, m["b_gu"][:M])
        act = m["b_act"][:M]
        act.copy_(F.silu(gu[:, :F2]) * gu[:, F2:])
        z = m["down"](act, M, z, res=z, ldr=self.H)
        out = m["b_out"][:M]
        self._rms(z, m["norm"], out)
        m["head"](out[M - 1:M], 1, m["logits"])
        if with_p:                                 # -> (draft, its top-1 probability over the draft vocabulary)
            lg = m["logits"][0]
            mx, i = lg.max(0)
            d, p = torch.stack([i.float(), 1.0 / torch.exp(lg - mx).sum()]).tolist()
            return int(d), p
        return int(m["logits"][0].argmax())

    # ------------------------------------------------------------------------------------------------ prefill
    def _pf_alloc(self, C):
        if getattr(self, "pf", None) is not None and self.pf["C"] >= C:
            return
        self.pf = None
        z = lambda *s, dt=BF16: torch.zeros(*s, dtype=dt, device=DEV)
        mla = [d for d in self.layers if d["kind"] == "mla"]
        gdn = [d for d in self.layers if d["kind"] == "gdn2"]
        rmax = max(d["r"] for d in mla)
        n1 = max([d["qkv"].N for d in gdn] + [d["qdk"].N for d in mla])
        no = max([d["Hh"] * d["Dv"] for d in gdn] + [d["Hh"] * d["D"] for d in mla])
        F2 = max(d["gu"].N for d in self.layers) // 2
        gw = max(ext().gdn2_prefill_work(C, d["Hh"]) for d in gdn)  # chunked GDN-2 scan scratch (fp32)
        self.pf = {"C": C, "gw": torch.empty(gw, dtype=torch.float32, device=DEV), "toks": z(C, dt=torch.long), "pos": z(C, dt=torch.long), "h": z(C, self.H), "x": z(C, self.H),
                   "hn": z(C, self.H), "y1": z(C * n1), "g": z(C * max(d["gates"].N for d in gdn)),
                   "z": z(C * max(d["z"].N for d in gdn)), "o": z(C * no), "gu": z(C * 2 * F2), "act": z(C * F2),
                   "qr": z(C, 16, 64, dt=torch.float32), "ql": z(C * 16 * rmax, dt=torch.float32),
                   "olat": z(C * 16 * rmax, dt=torch.float32)}

    def _pf_rows(self, M):
        """prefill M prompt tokens (self.pf toks / pos) through every layer: GEMMs over the rows (bf16 weights, fp32
        accumulation), GDN-2 as a chunked scan over the rows (csrc gdn2_prefill), MLA cache written for every row, causal
        attention as fp32 GEMMs (_pf_attn). -> self.pf["hn"][:M]."""
        pf, E = self.pf, self._E
        V = lambda key, n, dt=BF16: pf[key][:M * n].view(M, n)
        h, x = pf["h"][:M], pf["x"][:M]
        ext().emb_q4(pf["toks"], self.emb.codes, self.emb.scales, h, M)
        for d in self.layers:
            self._norm(h, d["ln1"], d["sc1"], x)
            if d["kind"] == "gdn2":
                qkv = d["qkv"](x, M, V("y1", d["qkv"].N))
                gates = d["gates"](x, M, V("g", d["gates"].N))
                zz = d["z"](x, M, V("z", d["z"].N))
                o = V("o", d["Hh"] * d["Dv"])
                if GDN_PREFILL == "step":                             # token by token (the decode kernel), for checks
                    ext().gdn2_step(qkv, d["conv"], d["conv"], 0, d["conv_w"], gates, d["A_log"], d["dt_bias"], d["S"],
                                    d["S"], 0, zz, d["gn"], d["gn_eps"], o, d["Hh"], d["Hq"], M)
                else:                                                 # chunked scan over the rows
                    ext().gdn2_prefill(qkv, d["conv"], d["conv_w"], gates, d["A_log"], d["dt_bias"], d["S"], zz, d["gn"],
                                       d["gn_eps"], o, pf["gw"], d["Hh"], M)
                d["out"](o, M, h, res=h, ldr=self.H)
            else:
                r, Hh = d["r"], d["Hh"]
                y = d["qdk"](x, M, V("y1", d["qdk"].N))
                ql = pf["ql"][:M * Hh * r].view(M, Hh, r)
                olat = pf["olat"][:M * Hh * r].view(M, Hh, r)
                ext().mla_prep(y, y.stride(0), M,
                               [pf["pos"], self.inv_freq, d["qn"], E if d["q_map"] is None else d["q_map"], d["U"], d["Ures"],
                                d["Unope"], pf["qr"], ql, E, E, E, E, d["kr"], d["c"], d["ir"]],
                               [Hh, d["G"], d["D"], d["rd"], d["nr"], r, int(d["gated"]), int(d["qn_zc"]), 0,
                                int(self.kv == "foveakv")], d["k_eps"])
                self._pf_attn(d, pf["qr"][:M], ql, M, int(pf["pos"][0]), olat)
                o = V("o", Hh * d["D"])
                self._pf_out(d, olat, y, M, o)
                d["o"](o, M, h, res=h, ldr=self.H)
            self._norm(h, d["ln2"], d["sc2"], x)
            F2 = d["gu"].N // 2
            gu = d["gu"](x, M, V("gu", 2 * F2))
            act = V("act", F2)
            torch.mul(F.silu(gu[:, :F2]), gu[:, F2:], out=act)
            d["down"](act, M, h, res=h, ldr=self.H)
        self._norm(h, self.norm_f, False, pf["hn"][:M])

    def _pf_attn(self, d, qr, ql, M, c0, olat):
        """causal MLA attention of the chunk's M rows (positions c0 .. c0 + M - 1) over the cache: score =
        scale / rms_g(t) (q_rope . k_rope + q_lat . c), out = sum_t p_t c_t. One fused bf16 attention (fp32 softmax and
        accumulation) per KV group, the per-key 1 / rms folded into the group's keys."""
        Hh, G, T = d["Hh"], d["G"], c0 + M
        hg = Hh // G
        sc = (d["ir"][:T].float() * d["D"] ** -0.5).t()                       # (G, T)
        K = (torch.cat([d["kr"][:T], d["c"][:T]], -1).float()[None] * sc[:, :, None]).to(BF16)   # (G, T, 64 + r)
        V = d["c"][:T]
        q = torch.cat([qr, ql], -1).view(M, G, hg, -1).permute(1, 2, 0, 3).to(BF16)               # (G, hg, M, 64 + r)
        mask = torch.arange(T, device=DEV)[None] <= torch.arange(c0, c0 + M, device=DEV)[:, None]
        o = F.scaled_dot_product_attention(q, K[:, None].expand(G, hg, T, -1), V[None, None].expand(G, hg, T, -1),
                                           attn_mask=mask, scale=1.0)                             # (G, hg, M, r)
        olat.copy_(o.permute(2, 0, 1, 3).reshape(M, Hh, -1))

    def _pf_out(self, d, olat, y, M, o):
        """mla_out for many rows: per KV group one fp32 GEMM (olat @ WvT[g]) -> bf16, times the bf16 sigmoid gate."""
        Hh, G = d["Hh"], d["G"]
        hg, D = Hh // G, d["WvT"].shape[2]
        a = torch.matmul(olat.view(M, G, hg, -1).transpose(0, 1).reshape(G, M * hg, -1), d["WvT"].float())
        v = a.view(G, M, hg, D).transpose(0, 1).reshape(M, Hh, D).to(BF16)
        if d["gated"]:
            gt = y[:, :Hh * 2 * D].view(M, Hh, 2, D)[:, :, 1].float()
            v = (v.float() * torch.sigmoid(gt).to(BF16).float()).to(BF16)
        o.view(M, Hh, D).copy_(v)

    @torch.no_grad()
    def prefill(self, prompt, mtp=False, chunk=1024):
        """the prompt at positions 0 .. P-1 in chunks of `chunk` tokens (the caches, the GDN-2 state and, with mtp,
        the MTP cache entries (h_i, x_{i+1})); -> logits / pick / hn of the last token, self.pos = P."""
        P = len(prompt)
        self.ensure_prefill(P)                                   # FoveaKV: the exact prompt cache may have been released
        self._pf_alloc(min(chunk, P))
        pf = self.pf
        ids = torch.tensor(prompt, device=DEV)
        for c0 in range(0, P, pf["C"]):
            M = min(pf["C"], P - c0)
            pf["toks"][:M].copy_(ids[c0:c0 + M])
            torch.arange(c0, c0 + M, device=DEV, out=pf["pos"][:M])
            self._pf_rows(M)
            R = M if c0 + M < P else M - 1                       # MTP cache entry i needs x_{i+1}
            if mtp and self.mtp is not None and R > 0:
                self._mtp(R, pf["hn"][:R], ids[c0 + 1:c0 + 1 + R], c0, fill_only=True)
        self.hn[0].copy_(pf["hn"][M - 1])
        self.head(self.hn, 1, self.logits)
        torch.argmax(self.logits[:1], -1, out=self.pick[:1])
        self.pos.fill_(P)

    def free_prefill(self):
        """drop the prefill buffers (and the GEMM workspace)."""
        from . import packed_kernels
        self.pf, packed_kernels._WS = None, None
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------------------------------ API
    def _fovea_attend(self, d, q_rope, q_lat, q_rr, q_lr, M, olat):
        k = self.kvc
        t = [d["f_cc"], d["f_gc"], d["f_cr"], d["f_gr"], d["f_Mc"], d["f_Mr"], d["cod_c"].lo, d["cod_c"].span,
             d["cod_r"].lo, d["cod_r"].span, d["f_sink_c"], d["f_sink_r"], d["f_ring_c"], d["f_ring_r"], d["f_ir"]]
        ext().fovea_codes_attn(q_rope, q_lat, q_rr, q_lr, self.posM, M, self.nsf, d["D"] ** -0.5, t, self.f_shr,
                               [k["sinks"], k["window"], k["block"], k["gain_bits"]], k["hot_frac"], self.f_R, self.f_sc,
                               self.f_lse, self.f_lp, self.f_thr, self.pm, self.pl, self.f_pax, self.f_par, olat)

    @torch.no_grad()
    def ensure_prefill(self, P):
        """FoveaKV: the exact cache holds the prompt only; (re)allocate it for a P-token prompt."""
        if self.kv != "foveakv" or P <= self.Pcap:
            return
        cap = (P + 1023) // 1024 * 1024
        for d in self.layers:
            if d["kind"] == "mla":
                d["kr"] = torch.zeros(cap, d["nr"] * d["rd"], device=DEV, dtype=BF16)
                d["c"] = torch.zeros(cap, d["r"], device=DEV, dtype=BF16)
                d["ir"] = torch.zeros(cap, d["G"], device=DEV)
        self.Pcap = cap
        self.graphs = {key: g for key, g in self.graphs.items() if key[2]}     # prompt-mode graphs held the old buffers

    @torch.no_grad()
    def release_prefill(self):
        """FoveaKV: free the exact prompt cache (decode needs only the codes, the sinks and the ring)."""
        if self.kv != "foveakv":
            return
        for d in self.layers:
            if d["kind"] == "mla":
                d["kr"], d["c"], d["ir"] = (torch.zeros(0, x.shape[1], device=DEV, dtype=x.dtype) for x in (d["kr"], d["c"], d["ir"]))
        self.Pcap = 0
        self.graphs = {key: g for key, g in self.graphs.items() if key[2]}
        torch.cuda.empty_cache()

    @torch.no_grad()
    def calibrate_fovea(self):
        """after the prompt (positions 0 .. P-1, attended exactly): calibrate every layer's gain range on the prompt,
        move the sinks and the last RW prompt tokens to the exact ring, encode the prompt tokens the first decode
        query reads quantized; later queries read codes."""
        if self.kv != "foveakv":
            return
        P, k = int(self.pos), self.kvc
        for d in self.layers:
            if d["kind"] != "mla":
                continue
            d["cod_c"].calibrate(d["c"], P)
            d["cod_r"].calibrate(d["kr"], P)
            ns = min(k["sinks"], P)
            d["f_sink_c"][:ns].copy_(d["c"][:ns]); d["f_sink_r"][:ns].copy_(d["kr"][:ns])
            RW = d["f_ring_c"].shape[0]
            lo = max(0, P - RW)
            idx = torch.arange(lo, P, device=DEV)
            d["f_ring_c"].index_copy_(0, idx % RW, d["c"][lo:P]); d["f_ring_r"].index_copy_(0, idx % RW, d["kr"][lo:P])
            d["f_ir"][:P].copy_(d["ir"][:P].half())
            if P - k["window"] >= 0:
                for s0 in range(0, P - k["window"] + 1, 256):        # chunks: each token gathers its block's rows
                    s = torch.arange(s0, min(s0 + 256, P - k["window"] + 1), device=DEV)
                    d["cod_c"].encode_codes(lambda i, x=d["c"]: x[i], s, d["f_cc"], d["f_gc"], d["f_Mc"])
                    d["cod_r"].encode_codes(lambda i, x=d["kr"]: x[i], s, d["f_cr"], d["f_gr"], d["f_Mr"])
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
    def capture(self, ks=None):
        """record the decode step (k = 0) and the verification blocks for k drafts as CUDA graphs (for the prompt's
        exact attention and, with FoveaKV, the tiered decode). Default: every k from 0 to maxM - 1 (what k="auto" uses)."""
        if ks is None:
            ks = tuple(range(self.maxM)) if self.mtp is not None else (0,)
        assert max(ks) + 1 <= self.maxM, f"k = {max(ks)} drafts needs maxM >= {max(ks) + 1}"
        self._capture_ks = tuple(ks)
        modes = (False, True) if self.kv == "foveakv" else (False,)
        saved = self.fovea_on
        for on in modes:
            if not on and self.kv == "foveakv" and self.Pcap == 0:
                continue                                  # prompt mode: captured once the prompt cache exists
            self.fovea_on = on
            for k in ks:
                self._graph(k + 1, k > 0)
        self.fovea_on = saved
        if self.mtp is not None and max(ks) > 0:                  # k="auto": verification costs measured now, so the
            from .spec_policy import DepthPolicy                  # first generate() does not pay for it
            self.calibrate_spec(max(ks))
            self._policy = DepthPolicy(max(ks), _vcost_fn(self._vab))

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
    def generate(self, prompt, n, eos=None, k="auto", prefill=True, keep_prefill=False):
        """greedy decoding of n tokens after prompt -> (tokens, stats). k: drafts per speculative block; "auto" (default)
        chooses 0 .. the largest captured k per block (spec_policy.DepthPolicy). Every k emits the plain greedy tokens."""
        if k == "auto":
            return self._generate_auto(prompt, n, eos, prefill, keep_prefill)
        return self._generate(prompt, n, eos, k, prefill, keep_prefill)

    def _generate_auto(self, prompt, n, eos=None, prefill=True, keep_prefill=False):
        """speculative decoding with the depth chosen per block by spec_policy.DepthPolicy (1 .. the largest captured k
        drafts; 0 = a plain step). Lossless like any fixed k."""
        from .spec_policy import DepthPolicy
        kmax = max(getattr(self, "_capture_ks", (0,)))
        if self.mtp is None or kmax == 0:
            return self._generate(prompt, n, eos, 0, prefill, keep_prefill)
        if getattr(self, "_policy", None) is None or self._policy.kmax != kmax:
            self.calibrate_spec(kmax)
            self._policy = DepthPolicy(kmax, _vcost_fn(self._vab))
        pol = self._policy
        pol.rate = None                                              # the rate depends on the context: per call
        hist0 = list(pol.hist)                                       # acceptance statistics carry over between calls
        self.reset()
        self.ensure_prefill(len(prompt))
        if (1, False, False) not in self.graphs and self.kv == "exact":
            self._graph(1, False)                                    # as in _generate
        if self.kv == "foveakv" and len(prompt) < self.kvc["sinks"]:
            raise ValueError(f"FoveaKV needs a prompt of at least {self.kvc['sinks']} tokens (its sinks)")
        if prefill:
            self.prefill(prompt, mtp=True)
        else:                                                        # token-by-token prompt: a fixed depth
            return self._generate(prompt, n, eos, min(kmax, 3), prefill, keep_prefill)
        x = int(self.pick[0])
        self.calibrate_fovea()
        if not keep_prefill:
            self.free_prefill()
            self.release_prefill()
        out, blocks, accepted, drafted = [x], 0, 0, 0
        hrows, base, fill = self.hn[:1].clone(), len(prompt) - 1, None
        while len(out) < n and not (eos is not None and x == eos):
            pol.begin(int(self.pos))
            drafts, probs, nd, td = self._draft_auto(hrows, x, base, pol, fill)
            block = [x] + drafts
            t0 = now()
            self._run(block, slots=bool(drafts))
            y = self.pick[:len(block)].tolist()
            tv = now() - t0
            nacc = 0
            while nacc < len(drafts) and drafts[nacc] == y[nacc]:
                nacc += 1
            emit = y[:nacc + 1]
            blocks += 1; accepted += nacc; drafted += len(drafts)
            pol.end(len(drafts), nacc, probs, td + tv, td, nd)
            done = False
            for t in emit:
                if len(out) >= n:
                    break
                out.append(t)
                if eos is not None and t == eos:
                    done = True
                    break
            if drafts:
                for d in self.layers:                                 # commit the state after row nacc
                    if d["kind"] == "gdn2":
                        d["conv"].copy_(d["conv_slots"][nacc])
                        d["S"].copy_(d["S_slots"][nacc])
            hrows = self.hn[:nacc + 1].clone()
            base = int(self.pos)
            self.pos.add_(nacc + 1)
            if done or len(out) >= n:
                break
            x, fill = emit[-1], emit
        return out[:n], {"blocks": blocks, "accepted": accepted, "drafts": drafted,
                         "acceptance": accepted / max(drafted, 1), "tokens_per_block": (len(out) - 1) / max(blocks, 1),
                         "depths": [a - b for a, b in zip(pol.hist, hist0)]}   # blocks by drafts verified, this call

    def _generate(self, prompt, n, eos=None, k=0, prefill=True, keep_prefill=False):
        """greedy decoding: the prompt, then n tokens. k > 0: speculative with k MTP drafts per block (lossless: the
        same tokens as k = 0). After the prompt, the prefill buffers and GEMM workspace and (FoveaKV) the exact prompt
        cache are freed unless keep_prefill: decode then holds only the codes, sinks and ring. Returns (ids, stats)."""
        self.reset()
        if self.kv == "foveakv" and len(prompt) < self.kvc["sinks"]:
            raise ValueError(f"FoveaKV needs a prompt of at least {self.kvc['sinks']} tokens (its sinks)")
        self.ensure_prefill(len(prompt))
        if getattr(self, "_capture_ks", None) and (1, False, False) not in self.graphs and (not prefill or self.kv == "exact"):
            self._graph(1, False)                         # prompt-mode step graph for the (re)allocated prompt cache
        mtp = k > 0 and self.mtp is not None
        out, blocks, accepted = [], 0, 0
        if prefill:
            self.prefill(prompt, mtp=mtp)
        else:                                                    # token by token (the decode kernels)
            for i, t in enumerate(prompt):
                self._run([t])
                if mtp and i + 1 < len(prompt):                  # MTP cache entry i: (h_i, x_{i+1})
                    self._mtp(1, self.hn[:1].clone(), torch.tensor([prompt[i + 1]], device=DEV), i)
                self.pos.add_(1)
        x = int(self.pick[0])
        self.calibrate_fovea()
        if not keep_prefill:
            self.free_prefill()
            self.release_prefill()
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

    def _draft_auto(self, hrows, x, base, pol, fill_toks=None):
        """as _draft, the depth chosen by the policy draft by draft: -> (drafts, their probabilities, drafted, seconds)"""
        t0 = now()
        R = hrows.shape[0]
        nxt = fill_toks if fill_toks is not None else [x]
        d, p = self._mtp(R, hrows, torch.tensor(nxt, device=DEV), base, with_p=True)
        drafts, probs, nd, S = [], [], 1, 1.0
        keep, S1 = pol.keep(0, p, S)
        if keep:
            drafts.append(d); probs.append(p); S = S1
            hcur = self.mtp["b_out"][R - 1:R].clone()
            while pol.want_more(len(drafts), S, probs[-1]):
                d, p = self._mtp(1, hcur, torch.tensor([drafts[-1]], device=DEV), base + R - 1 + len(drafts), with_p=True)
                nd += 1
                keep, S1 = pol.keep(len(drafts), p, S)
                if not keep:
                    break
                drafts.append(d); probs.append(p); S = S1
                hcur = self.mtp["b_out"][:1].clone()
        return drafts, probs, nd, now() - t0

    def calibrate_spec(self, kmax, reps=5):
        """verification cost V(M, T) = A_M + B_M T for M = 1 .. kmax + 1 rows, from the captured decode graphs replayed
        at a short and a long context (the state is reset afterwards)."""
        saved = self.fovea_on
        self.fovea_on = self.kv == "foveakv"
        Ts = (64, self.Tmax - self.maxM - 8)
        cost = {}
        for M in range(1, kmax + 2):
            g = self._graph(M, M > 1)
            for T in Ts:
                self.posM[:M].copy_(torch.arange(T, T + M, device=DEV))
                for _ in range(2):
                    g.replay()
                torch.cuda.synchronize(); t0 = now()
                for _ in range(reps):
                    g.replay()
                torch.cuda.synchronize(); cost[(M, T)] = (now() - t0) / reps
        self.fovea_on = saved
        self.reset()
        self._vab = {M: (cost[(M, Ts[0])], (cost[(M, Ts[1])] - cost[(M, Ts[0])]) / (Ts[1] - Ts[0])) for M in range(1, kmax + 2)}
        return self._vab

    def vcost(self, M, T):
        a, b = self._vab[M]
        return a + b * T

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
