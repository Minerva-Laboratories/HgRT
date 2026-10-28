import os
"""Native prefill for the HgRT Mercurius engine (no transformers model, no snapshot).

Layer-major over the whole prompt, token-chunked:
  * NF4 linears: the layer's weights dequantized once to bf16 with the exact NF4 levels (bf16(L * absmax), like
    bitsandbytes' dequantize_4bit), then cuBLAS bf16 tensor-core GEMMs per token chunk;
  * GDN-2: factored gates (tiled bases + VeRA + decay LoRA) as GEMMs, causal conv carried across chunks, fla's
    chunk_gdn2 with the recurrent state carried across chunks (the kernel the deployed model uses), gated RMSNorm;
  * MLA: the chunk's TurboQuant codes written into the engine's cache by the decode encode kernel; the chunk's keys are
    decoded back from those codes (so prefill attends to exactly what decode will read) into per-group GQA keys
    in the expanded form, [rope(R0^T kr) | k_res | k_nope,g] * 1/rms_g (448 wide; k_up on the decoded latent) and values
    W_v,g latent (256 wide), then a Triton flash kernel (OpenDecider's wide_attention: online softmax, no attention
    matrix) for the chunk's queries (Q_EXP_h maps + RoPE) against all keys so far;
  * the last position's logits through the engine's NF4 head (as decode), the normed final hidden of every position
    returned for the MTP cache.
The engine state afterwards (MLA code cache, GDN-2 conv / recurrent states, position, first token) is what decode
continues from."""
import math

import torch
import torch.nn.functional as F

from .mercurius_engine import CONV, DEV, DKH, FF, HD, HID, NG, NH, NK, NV, RD, VALD
from .mercurius_kernels import ext


def _bf(x):
    return x.to(torch.bfloat16)


def _rms(x, gain, eps=1e-6):
    xf = x.float()
    return _bf(xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gain)


def _rope_tables(inv_freq, pos):
    ang = pos.float()[:, None] * inv_freq[None]
    return ang.cos().to(torch.bfloat16).float(), ang.sin().to(torch.bfloat16).float()


def _rope64(x, cos, sin):
    """x (..., 64) fp32, cos/sin (T, 32) broadcast over the middle dims"""
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x0, x1 = x[..., :32], x[..., 32:64]
    return torch.cat([x0 * cos - x1 * sin, x1 * cos + x0 * sin], -1)


class Prefill:
    def __init__(self, E, chunk=2048, profile=False):
        self.E, self.chunk, self.profile = E, chunk, profile
        self.last_profile = None
        self.gprof = {}
        assert E.gemv == "dp4a", "prefill dequantizes the dp4a code layout"
        nmax = max(l.N * l.K for d in E.layers for l in d.values() if hasattr(l, "codes") and hasattr(l, "K"))
        self.wbuf = torch.empty(nmax, dtype=torch.bfloat16, device=DEV)
        self.levels = E.emb_code.contiguous()                      # the bitsandbytes NF4 code table (fp32)
        # Triton tiling for 720-wide keys (the default 64 x 64 tiles need 229 KB of shared memory)
        self.wa = dict(BM=64, BN=32, BD=64, num_warps=4, num_stages=1)
        # expanded MLA form for prefill (448-wide keys instead of 720): the per-head query maps Q_EXP_h and the key
        # up-projection on the rotated latent, from the same converter as the GGUF (fold_attention), fp16
        import os
        ex = torch.load(os.environ.get("MERC_PREFILL_EXTRA", os.path.join(os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius"), "prefill_extra.pt")))
        self.qexp = {i: v["q_exp"].to(DEV, torch.bfloat16).transpose(1, 2).contiguous() for i, v in ex.items()}  # (H,256,448)
        self.kup = {i: v["k_up"].to(DEV, torch.bfloat16) for i, v in ex.items()}                                 # (960, r)

    def W(self, lin):
        """bf16 (N, K) view of a dequantized NF4 linear (scratch buffer, valid until the next call)"""
        out = self.wbuf[:lin.N * lin.K].view(lin.N, lin.K)
        ext().nf4_dequant(lin.codes, lin.scales, self.levels, out, 16256.0)
        return out

    # ------------------------------------------------------------------------------------------------ layers
    def gdn(self, i, d, h):
        E, T, C = self.E, h.shape[0], self.chunk
        Wqkvz = self.W(d["qkvz"]).clone()                           # (12288, 2560); the scratch is reused for out
        Wout = None
        conv = torch.zeros(3, CONV, device=DEV)                     # last 3 raw inputs (fp32 of bf16)
        S = torch.zeros(1, NV, DKH, DKH, device=DEV)
        A_log, dtb = d["A_log"], d["dt_bias"]
        from fla.ops.gdn2 import chunk_gdn2
        import time
        pf = self.gprof if self.profile else None

        def tick(k, t0):
            if pf is None:
                return t0
            torch.cuda.synchronize()
            t1 = time.time()
            pf[k] = pf.get(k, 0.0) + t1 - t0
            return t1
        outs = []
        for a in range(0, T, C):
            b = min(T, a + C)
            t0 = tick("_", time.time()) if pf is not None else 0.0
            x = _rms(h[a:b], d["ln1"])
            qkvz = x @ Wqkvz.T                                       # bf16
            t0 = tick("qkvz gemm", t0)
            sg = _bf(x @ d["Ws"].T).float().contiguous()            # base | A x | lora A x (bf16 values)
            t_ = b - a
            u = sg[:, 3 * NV:3 * NV + E.rank]
            W2 = _bf(d["vd"][:, None, :] * u[None]).view(3 * t_, E.rank)   # rows (g, m): bf16(vd_g u_m)
            Tm = (W2 @ d["B"].T).float()                               # (3t, 4096): the VeRA B products
            gdec = torch.empty(t_, VALD, device=DEV)
            ber, bwr = torch.empty_like(gdec), torch.empty_like(gdec)
            ext().gdn2_gates_epi(sg, sg.shape[1], NV, E.rank, d["lr"], Tm, d["Bl"], d["vb"], A_log, dtb, gdec, ber, bwr,
                                 t_, DKH)
            be, bw = _bf(ber), _bf(bwr)
            t0 = tick("gates", t0)
            raw = qkvz[:, :CONV].float()
            win = torch.cat([conv, raw], 0)                          # (3 + t, C): previous 3 inputs + the chunk
            cw = d["conv_w"]
            acc = win[0:t_] * cw[:, 0]
            for jj in range(1, 4):
                acc = torch.addcmul(acc, win[jj:jj + t_], cw[:, jj])
            mixed = _bf(F.silu(acc))
            conv = win[-3:]
            t0 = tick("conv", t0)
            q, k, v = torch.split(mixed, [NK * DKH, NK * DKH, VALD], -1)
            q = q.reshape(1, t_, NK, DKH).repeat_interleave(NV // NK, 2)
            k = k.reshape(1, t_, NK, DKH).repeat_interleave(NV // NK, 2)
            o, S = chunk_gdn2(q=q, k=k, v=v.reshape(1, t_, NV, DKH), g=gdec.view(1, t_, NV, DKH).contiguous(),
                              b=be.view(1, t_, NV, DKH).contiguous(), w=bw.view(1, t_, NV, DKH).contiguous(),
                              initial_state=S, output_final_state=True, use_qk_l2norm_in_kernel=True)
            t0 = tick("chunk_gdn2", t0)
            of = o[0].float()                                        # (t, NV, DKH), bf16 values
            on = _bf(_bf(of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + d["gn_eps"])).float() * d["gn"].float())
            on = _bf(on.float() * F.silu(qkvz[:, CONV:].float().view(t_, NV, DKH)))
            outs.append(on.view(t_, VALD))
            t0 = tick("gated norm", t0)
        del Wqkvz
        Wout = self.W(d["out"])
        for a, on in zip(range(0, T, C), outs):
            b = min(T, a + C)
            h[a:b] = _bf((on @ Wout.T).float() + h[a:b].float())
        E.conv[i].copy_(torch.cat([torch.zeros(1, CONV, device=DEV), conv], 0).T)   # (C, 4); only the last 3 are used
        E.rec[i].copy_(S[0].float())

    def mla(self, i, d, h):
        E, X, T, C = self.E, ext(), h.shape[0], self.chunk
        r, Ep, RW = d["r"], d["Ep"], d["RW"]
        NQD = NH * 2 * HD + r + NG * HD
        Wq = self.W(d["qdk"]).clone()
        Kg = torch.empty(1, NG, T, 448, dtype=torch.bfloat16, device=DEV)
        Vg = torch.empty(1, NG, T, HD, dtype=torch.bfloat16, device=DEV)
        o_all = torch.empty(T, NH * HD, dtype=torch.bfloat16, device=DEV)
        from opendecider.wide_attention import wide_attention
        import time
        pf = self.gprof if self.profile else None

        def tick(k, t0):
            if pf is None:
                return t0
            torch.cuda.synchronize()
            t1 = time.time()
            pf["mla/" + k] = pf.get("mla/" + k, 0.0) + t1 - t0
            return t1
        cbl, cbr = d["cbl"][:16], d["cbr"][:16]
        R0f, Wvf = d["R0"].float(), d["Wv"]
        for a in range(0, T, C):
            b = min(T, a + C)
            t_ = b - a
            t0 = time.time()
            x = _rms(h[a:b], d["ln1"])
            yq = x @ Wq.T                                            # (t, NQD) bf16
            kt0 = _bf(x @ d["Wr"].T)
            kr = _bf(kt0 @ d["R0"].T).float().contiguous()
            crot = _bf(yq[:, NH * 2 * HD:NH * 2 * HD + r] @ d["R"].T).float().contiguous()
            X.mla_encode(yq[:, NH * 2 * HD + r:], NQD, kr, crot, r, d["cbl"], d["cbr"],
                         torch.tensor([a], device=DEV), d["cache"], RW, d["k_eps"], t_)
            t0 = tick("proj+encode", t0)
            # keys / values of this chunk, decoded back from the codes just written
            rows = d["cache"][a:b]
            inv = rows[:, 4:20].contiguous().view(torch.float32).view(t_, NG)
            nl = rows[:, 20:22].contiguous().view(torch.float16).view(t_).float().clamp_min(1e-12)
            nr = rows[:, 22:24].contiguous().view(torch.float16).view(t_).float().clamp_min(1e-12)
            unpack = lambda by, w: torch.stack([by & 15, by >> 4], -1).view(t_, -1)[:, :w].long()
            rk = cbr[unpack(rows[:, 24:24 + RD // 2], RD)] * nr[:, None]
            lat = _bf(cbl[unpack(rows[:, 24 + RD // 2:24 + RD // 2 + (r + 1) // 2], r)] * nl[:, None])
            pos = torch.arange(a, b, device=DEV)
            cos, sin = _rope_tables(E.inv_freq, pos)
            kt = _bf(_bf(rk) @ d["R0"]).float()
            krope = _rope64(kt, cos, sin)                                               # (t, 64)
            kup = (lat @ self.kup[i].T).float()                                         # (t, 192 + 4 x 192)
            for gg in range(NG):                                                        # [rope | shared res | nope_g]
                kexp = torch.cat([krope, kup[:, :192], kup[:, 192 * (gg + 1):192 * (gg + 2)]], -1)
                Kg[0, gg, a:b] = _bf(kexp * inv[:, gg:gg + 1])
            Vg[0, :, a:b] = _bf(torch.einsum("tc,gdc->gtd", lat.float(), Wvf.float()))
            t0 = tick("decode K/V", t0)
            # queries: y_h = RMSNorm(q_h) * qn, [q_rope | q_abs] = P_h y_h, RoPE
            qh = yq[:, :NH * 2 * HD].view(t_, NH, 2 * HD)[..., :HD]
            y = _rms(qh, d["qn"], d["q_eps"])                                            # (t, H, 256) bf16
            qq = torch.bmm(y.transpose(0, 1), self.qexp[i]).float()                     # (H, t, 448)
            qf = torch.cat([_rope64(qq[..., :RD], cos[None], sin[None]), qq[..., RD:]], -1)
            t0 = tick("queries", t0)
            o = wide_attention(_bf(qf)[None], Kg[:, :, :b], Vg[:, :, :b], d["scale"], **self.wa)   # (1, H, t, 256)
            t0 = tick("attention", t0)
            gate = yq[:, :NH * 2 * HD].view(t_, NH, 2 * HD)[..., HD:]
            o_all[a:b] = _bf(_bf(o[0].transpose(0, 1)).float() * _bf(torch.sigmoid(gate.float())).float()).reshape(t_, -1)
        del Wq, Kg, Vg
        Wo = self.W(d["o"])
        for a in range(0, T, C):
            b = min(T, a + C)
            h[a:b] = _bf((o_all[a:b] @ Wo.T).float() + h[a:b].float())

    def mlp(self, d, h):
        T, C = h.shape[0], self.chunk
        Wgu = self.W(d["gu"]).clone()
        acts = []
        for a in range(0, T, C):
            b = min(T, a + C)
            gu = _rms(h[a:b], d["ln2"]) @ Wgu.T
            acts.append(_bf(_bf(F.silu(gu[:, :FF].float())).float() * gu[:, FF:].float()))
        del Wgu
        Wd = self.W(d["down"])
        for a, act in zip(range(0, T, C), acts):
            b = min(T, a + C)
            h[a:b] = _bf((act @ Wd.T).float() + h[a:b].float())
        del acts

    # ------------------------------------------------------------------------------------------------ API
    @torch.no_grad()
    def __call__(self, ids):
        """ids (T,) long. Fills the engine (cache, states, pos = T, tok_in[0] = greedy first token, logits[0]) and
        returns a snapshot-like dict {"ids", "hidden" (T, HID) normed final hidden bf16} for MTPHead.prefill."""
        E = self.E
        ids = ids.to(DEV)
        T = ids.shape[0]
        assert T + 64 < E.Tmax
        h = E.emb_rows(ids)                                           # (T, HID) bf16
        import time
        prof = {"gdn2": 0.0, "mla": 0.0, "mlp": 0.0} if self.profile else None
        for i, d in enumerate(E.layers):
            if prof is not None:
                torch.cuda.synchronize(); t0 = time.time()
            if i in E.lin_ids:
                self.gdn(i, d, h)
            else:
                self.mla(i, d, h)
            if prof is not None:
                torch.cuda.synchronize(); t1 = time.time()
                prof["gdn2" if i in E.lin_ids else "mla"] += t1 - t0
            self.mlp(d, h)
            if prof is not None:
                torch.cuda.synchronize(); prof["mlp"] += time.time() - t1
        if prof is not None:
            prof.update({(k if k.startswith("mla/") else "gdn2/" + k): v for k, v in self.gprof.items() if k != "_"})
            self.gprof = {}
        self.last_profile = prof
        hidden = torch.cat([_rms(h[a:a + self.chunk], E.norm_f) for a in range(0, T, self.chunk)])
        # first token through the decode head (int8 activation groups + NF4 GEMV), as every decoded token
        X = ext()
        xq, xd = E.xq(HID)
        E.x[0].copy_(hidden[-1])
        X.quant_rows(E.x, HID, xq, xd, 1, HID)
        E.head(xq, xd, 1, E.logits, E.Vrows)
        X.argmax_rows(E.logits, E.Vrows, E.V, E.tok_out, 1)
        E.pos.fill_(T)
        E.nacc_prev.fill_(-1)
        E.tok_in.zero_()
        E.tok_in[0:1].copy_(E.tok_out[0:1])
        E.T0 = T
        return {"ids": ids, "hidden": hidden}
