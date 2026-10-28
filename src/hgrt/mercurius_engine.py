"""HgRT decode engine for Mercurius-1-4B (24 GDN-2 + 8 absorbed-MLA layers, NF4 weights, TurboQuant 4-bit KV latent).

One token (M = 1) or a block of M <= 4 tokens (speculative verification) runs as ONE captured CUDA graph: NF4 embedding,
per layer RMSNorm fused with the int8 activation groups, NF4 GEMVs (exact to the deployed bitsandbytes codes and block
scales), GDN-2 (factored channel-wise gates, conv, recurrence with the state in registers, gated RMSNorm), absorbed
MLA (TurboQuant encode of this step's latent / RoPE key straight into the 4-bit cache, query map, split-KV tensor-core
attention reading the codes, V up-projection + output gate), the NF4 LM head (the tied embedding table) and greedy pick.
Positions and tokens live on the GPU; no host synchronization inside a generation loop.

Every kernel computes each of the M rows with the arithmetic of the single-token case, so the M-row verification pass
produces bit-identical logits for the tokens it shares with plain decoding (tests/test_mercurius.py checks it).
The weights come from experiments/mercurius/prep.py (one load of the deployed transformers model)."""
import math

import torch

from .mercurius_kernels import ext, nf4_tables, pack_nf4

DEV = "cuda"
HID, FF, NV, NK, DKH = 2560, 9216, 32, 16, 128
KEYD, VALD = NK * DKH, NV * DKH
CONV = 2 * KEYD + VALD                       # 8192
NH, HD, NG, RD = 16, 256, 4, 64


def pad16(x):
    return (x + 15) // 16 * 16


# GEMV tiling (R rows per warp * 10 + U groups in flight per lane), chosen by bench (experiments/mercurius/gemv_tune.py)
GEMV_ROWS = {}


APPROX_NF4 = False          # True: one-table NF4 levels (|error| <= 3.9e-3, NOT the deployed weights); for measurement


def rows_for(N, K, M):
    code = GEMV_ROWS.get((N, K, M), 41 if M == 1 else 21)
    return code + 100 if APPROX_NF4 else code


def load_gemv_rows(path=None):
    import json
    import os
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "results", "mercurius", "gemv_rows.json")
    if os.path.exists(path):
        for k, v in json.load(open(path)).items():
            GEMV_ROWS[tuple(int(x) for x in k.split(","))] = int(v)


GEMV_MODE = {"mode": "dp4a"}      # "dp4a" (CUDA-core dp4a GEMV) or "mma" (int8 tensor cores); set by MercuriusEngine
MMA_KS = {}                       # (N, K) -> K slices per block for nf4_mma (tuned)


def mma_ks_for(N, K):
    if (N, K) in MMA_KS:
        return MMA_KS[(N, K)]
    return 8 if N <= 4096 else (2 if N <= 16384 else 1)


class NF4Lin:
    """one bitsandbytes NF4 linear (or several concatenated along N), kernel layout; y = W x (+ res) for M rows.
    The same kernel and configuration serve every M (bit identity of the verification pass)."""

    def __init__(self, parts, mode=None):
        packed = torch.cat([p for p, _ in parts]) if len(parts) > 1 else parts[0][0]
        absmax = torch.cat([a for _, a in parts]) if len(parts) > 1 else parts[0][1]
        self.mode = mode or GEMV_MODE["mode"]
        packed = packed.to(DEV)
        self.N, self.K = packed.shape[0], packed.shape[1] * 2
        if self.mode == "mma" and self.K % 128 == 0:
            from .mercurius_kernels import mma_layout
            self.codes = mma_layout(packed)
            self.scales = (absmax.float() / 16256.0).to(DEV).contiguous()
            self.ks = mma_ks_for(self.N, self.K)
        else:
            self.mode = "dp4a"
            self.codes, self.scales = pack_nf4(packed, absmax.to(DEV))

    def __call__(self, xq, xd, M, y, ldy, res=None, ldr=0):
        if self.mode == "mma":
            ext().nf4_mma(self.codes, self.scales, xq, xd, res, ldr, y, ldy, M, self.ks)
        else:
            ext().nf4_gemv(self.codes, self.scales, xq, xd, res, ldr, y, ldy, M, rows_for(self.N, self.K, M))

    def packed(self):
        """(N, K/2) uint8, element 2i in the low nibble"""
        if self.mode == "mma":
            w = self.codes                                         # (N, nsup, t, j) int32
            lo = (w & 0xFFFF).to(torch.int16).permute(0, 1, 3, 2)  # (N, nsup, j, t)
            hi = ((w >> 16) & 0xFFFF).to(torch.int16).permute(0, 1, 3, 2)
            h = torch.cat([lo, hi], -1).contiguous()               # (N, nsup, j, 8 halfwords)
            return h.view(torch.uint8).view(self.N, self.K // 2)
        return self.codes.view(torch.uint8).view(self.N, self.K // 2)

    def dequant(self, levels):
        """fp32 (N, K) weights the kernel uses (levels = the int16-represented NF4 levels)."""
        w = self.packed()
        idx = torch.stack([w & 15, w >> 4], -1).view(self.N, self.K).long()
        return (levels.to(DEV).float()[idx].view(self.N, self.K // 64, 64) * (self.scales * 16256.0)[..., None]).view(self.N, self.K)


class MercuriusEngine:
    @torch.no_grad()
    def __init__(self, weights, Tmax=40960, maxM=4, NS=32, gemv="dp4a", attn="v2"):
        W = torch.load(weights, map_location="cpu", weights_only=False) if isinstance(weights, str) else weights
        meta, t, nf = W["meta"], W["t"], W["nf4"]
        load_gemv_rows()
        GEMV_MODE["mode"] = gemv
        self.gemv = gemv
        self.attn = attn                                  # "fd": flash-decoding from the codes (v3); "v2": smem K tile
        hi, lo, self.levels = nf4_tables(meta["nf4_code"])
        ext().nf4_set_tables(hi, lo)
        self.V = meta["vocab_tok"]
        self.maxM, self.NS, self.Tmax = maxM, NS, Tmax
        self.L = 32
        self.lin_ids = [i for i in range(self.L) if f"l{i}.qkv" in nf]
        self.attn_ids = [i for i in range(self.L) if f"l{i}.q" in nf]
        g = lambda k: t[k].to(DEV)
        f32 = lambda k: t[k].float().to(DEV)
        bf16 = lambda x: x.to(DEV, torch.bfloat16).contiguous()
        self.layers = []
        self.eps = 1e-6
        self.rank = t["vera_A"].shape[0]
        for i in range(self.L):
            p = f"l{i}."
            d = {"ln1": f32(p + "ln1"), "ln2": f32(p + "ln2"),
                 "gu": NF4Lin([nf[p + "gate_proj"], nf[p + "up_proj"]]), "down": NF4Lin([nf[p + "down_proj"]])}
            if i in self.lin_ids:
                d["qkvz"] = NF4Lin([nf[p + "qkv"], nf[p + "z"]])
                d["out"] = NF4Lin([nf[p + "out"]])
                d["conv_w"] = f32(p + "conv").contiguous()
                d["A_log"] = f32(p + "A_log").reshape(-1).contiguous()
                d["dt_bias"] = f32(p + "dt_bias").reshape(-1).contiguous()
                d["gn"] = bf16(t[p + "gn"])
                d["gn_eps"] = meta.get(p + "gn_eps", 1e-6)
                lr = t[p + "loraA"].shape[0] if p + "loraA" in t else 0
                small = [t[p + "U_a"], t[p + "U_be"], t[p + "U_bw"], t["vera_A"]] + ([t[p + "loraA"]] if lr else [])
                d["Ws"] = bf16(torch.cat([x.to(torch.bfloat16) for x in small]))
                d["lr"] = lr
                d["B"] = bf16(t["vera_B"]) if i == self.lin_ids[0] else self.layers[self.lin_ids[0]]["B"]
                d["Bl"] = bf16(t[p + "loraB"]) if lr else None
                d["vd"] = torch.stack([t[p + f"vd_{x}"].float() for x in ("a", "be", "bw")]).to(DEV).contiguous()
                d["vb"] = torch.stack([t[p + f"vb_{x}"].float() for x in ("a", "be", "bw")]).to(DEV).contiguous()
                d["pe_c"] = f32(p + "pe_c") if p + "pe_c" in t else None
            else:
                a = meta[p + "attn"]
                r = a["r"]
                d["r"], d["E"], d["Ep"] = r, RD + r, pad16(RD + r)
                d["RW"] = pad16(24 + RD // 2 + (r + 1) // 2)
                d["scale"], d["q_eps"], d["k_eps"] = a["scale"], a["q_eps"], a["k_eps"]
                d["qdk"] = NF4Lin([nf[p + "q"], nf[p + "down"], nf[p + "krms"]])
                d["o"] = NF4Lin([nf[p + "o"]])
                d["qn"] = f32(p + "qn")
                d["P"] = bf16(t[p + "P"])                        # (H, 64 + r, 256)
                d["Wv"] = bf16(t[p + "Wv"])                      # (G, 256, r)
                d["Wr"] = bf16(t[p + "Wr"])                      # (64, 2560)
                d["R0"] = bf16(t[p + "R0"])                      # (64, 64)
                d["R0h"] = t[p + "R0"].to(DEV, torch.float16).contiguous()
                d["R"] = bf16(t[p + "R"])                        # (r, r)
                d["cbl"], d["cbr"] = f32(p + "cbl"), f32(p + "cbr")
                d["cache"] = torch.zeros(Tmax, d["RW"], dtype=torch.uint8, device=DEV)
            self.layers.append(d)
        self.norm_f = f32("norm_f")
        ec = t["emb_codes"]                                      # bitsandbytes packing (element 2i high nibble)
        low_first = ((ec >> 4) | ((ec & 15) << 4)).contiguous()
        self.head = NF4Lin([(low_first, t["emb_absmax"])])
        self.emb_code = f32("emb_code")
        self.emb_bytes = None if self.head.mode == "mma" else self.head.codes.view(torch.uint8).view(self.head.N, HID // 2)
        self.emb_absmax = (self.head.scales * 16256.0).contiguous()
        self.Vrows = self.head.N
        # the deployed model's rotary buffer is bf16 (the model was cast as a whole): its values, used in fp32
        self.inv_freq = (1.0 / (1e7 ** (torch.arange(0, RD, 2, dtype=torch.int64).float() / RD))).to(torch.bfloat16).float().to(DEV)
        del W
        self._alloc()
        self.graphs = {}

    # ------------------------------------------------------------------------------------------------ buffers
    def _alloc(self):
        M = self.maxM
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=DEV)
        self.pos = z(1, dt=torch.long)
        self.tok_in = z(M, dt=torch.long)                 # tokens fed at positions pos .. pos + M - 1
        self.tok_out = z(M, dt=torch.long)                # greedy pick after each of them
        self.h = z(M, HID, dt=torch.bfloat16)
        self.x = z(16, HID, dt=torch.bfloat16)            # 16 rows: the tensor-core skinny GEMMs read 16 (extra rows unused)
        self._xq = {}
        for K in (HID, VALD, FF, NH * HD):
            self._xq[K] = (z(M * K, dt=torch.int8), z(M * K // 32))
        self.qkvz = z(M, CONV + VALD, dt=torch.bfloat16)
        self.gu = z(M, 2 * FF, dt=torch.bfloat16)
        lr_max = max(self.layers[i]["lr"] for i in self.lin_ids)
        self.NSG = 3 * NV + self.rank + lr_max
        self.sg = z(M, self.NSG)
        self.W2 = z(16, self.rank, dt=torch.bfloat16)
        self.T = z(16, VALD)
        self.gates_tc = True
        self.side = torch.cuda.Stream()
        self.gdn_out = z(M, VALD, dt=torch.bfloat16)
        self.conv = {i: z(CONV, 4) for i in self.lin_ids}
        self.rec = {i: z(NV, DKH, DKH) for i in self.lin_ids}
        # per-step GDN-2 inputs, two sets (parity) so a cycle can replay the previous cycle's accepted steps (fused commit)
        self.mixed2 = {i: [z(M, CONV), z(M, CONV)] for i in self.lin_ids}
        self.gd2 = {i: [z(M, VALD), z(M, VALD)] for i in self.lin_ids}
        self.be2 = {i: [z(M, VALD), z(M, VALD)] for i in self.lin_ids}
        self.bw2 = {i: [z(M, VALD), z(M, VALD)] for i in self.lin_ids}
        self.xsave2 = {i: [z(M, CONV, dt=torch.bfloat16), z(M, CONV, dt=torch.bfloat16)] for i in self.lin_ids}
        self.mixed = {i: v[0] for i, v in self.mixed2.items()}
        self.gd = {i: v[0] for i, v in self.gd2.items()}
        self.be = {i: v[0] for i, v in self.be2.items()}
        self.bw = {i: v[0] for i, v in self.bw2.items()}
        self.xsave = {i: v[0] for i, v in self.xsave2.items()}
        self.nacc = z(1, dt=torch.int32)                  # accepted drafts of the last verification (GPU)
        self.nacc_prev = torch.full((1,), -1, dtype=torch.int32, device=DEV)   # pending commit (fused mode); -1: none
        rmax = max(self.layers[i]["r"] for i in self.attn_ids)
        Epm = pad16(RD + rmax)
        self.NQD = {i: NH * 2 * HD + self.layers[i]["r"] + NG * HD for i in self.attn_ids}
        self.yq = z(M, max(self.NQD.values()), dt=torch.bfloat16)
        self.kt0 = z(M, RD, dt=torch.bfloat16)
        self.kr = z(M, RD)
        self.crot = z(M, rmax)
        self.Q = z(M, NH, Epm, dt=torch.float16)
        self.part_o = z(self.NS, M, NH, Epm - RD)
        self.part_ml = z(self.NS, M, NH, 2)
        self.olat = z(M, NH, rmax)
        self.yattn = z(M, NH * HD, dt=torch.bfloat16)
        self.logits = z(M, self.Vrows)

    def embed(self, tok, y, M):
        if self.emb_bytes is None:
            ext().emb_nf4_mma(tok, self.head.codes, self.emb_absmax, self.emb_code, y, HID, M)
        else:
            ext().emb_nf4(tok, self.emb_bytes, self.emb_absmax, self.emb_code, y, HID, M)

    def emb_rows(self, tok):
        """bf16 embedding rows for a 1-D token tensor (PyTorch, any length)"""
        w8 = self.head.packed()[tok].long()
        idx = torch.stack([w8 & 15, w8 >> 4], -1).view(tok.shape[0], HID)
        return (self.emb_code[idx].view(tok.shape[0], -1, 64) * self.emb_absmax[tok][..., None]).view(tok.shape[0], HID).to(torch.bfloat16)

    def xq(self, K):
        return self._xq[K]

    # ------------------------------------------------------------------------------------------------ one block of M tokens
    def _gdn(self, i, d, M, mode, parity=0):
        """mode 0 plain, 1 verify (state untouched, commit later), 3 fused: replay the pending commit of the previous
        cycle (inputs of set 1 - parity, count nacc_prev) then verify with input set `parity`."""
        E = ext()
        cur = parity if mode == 3 else 0
        gd, be, bw, mixed, xsave = self.gd2[i][cur], self.be2[i][cur], self.bw2[i][cur], self.mixed2[i][cur], self.xsave2[i][cur]
        xq, xd = self.xq(HID)
        E.rms_rows(self.h, HID, d["ln1"], self.x, HID, xq, xd, M, HID, self.eps)
        if self.gates_tc:                                 # tensor cores (bf16 inputs exact, fp32 accumulate), on a side
            main = torch.cuda.current_stream()            # stream: they overlap the qkv|z GEMV (same input, no deps)
            self.side.wait_stream(main)
            with torch.cuda.stream(self.side):
                E.tc_skinny(d["Ws"], HID, d["Ws"].shape[0], HID, self.x, HID, self.sg, self.NSG, M, True)
                E.gdn2_gates_tc(self.sg, self.NSG, NV, self.rank, d["lr"], d["B"], d["Bl"], d["vd"], d["vb"], d["A_log"],
                                d["dt_bias"], self.W2, self.T, gd, be, bw, M, DKH)
            d["qkvz"](xq, xd, M, self.qkvz, CONV + VALD)
            main.wait_stream(self.side)
        else:
            d["qkvz"](xq, xd, M, self.qkvz, CONV + VALD)
            E.dgemv(d["Ws"], HID, d["Ws"].shape[0], HID, self.x, HID, self.sg, self.NSG, M, True)
            E.gdn2_gates(self.sg, self.NSG, NV, self.rank, d["lr"], d["B"], d["Bl"], d["vd"], d["vb"], d["A_log"],
                         d["dt_bias"], gd, be, bw, M, DKH)
        xq4, xd4 = self.xq(VALD)
        if mode == 3:
            pv = 1 - cur
            E.gdn2_conv_fused(self.qkvz, CONV + VALD, self.conv[i], d["conv_w"], mixed, xsave, self.xsave2[i][pv],
                              self.nacc_prev, M)
            E.gdn2_rec_fused(mixed, self.mixed2[i][pv], self.qkvz[:, CONV:], CONV + VALD, gd, be, bw, self.gd2[i][pv],
                             self.be2[i][pv], self.bw2[i][pv], d["gn"], self.rec[i], self.gdn_out, xq4, xd4, NK, NV,
                             d["gn_eps"], M, self.nacc_prev, d["pe_c"])
        else:
            E.gdn2_conv(self.qkvz, CONV + VALD, self.conv[i], d["conv_w"], mixed, xsave, M, mode == 0)
            E.gdn2_rec(mixed, self.qkvz[:, CONV:], CONV + VALD, gd, be, bw, d["gn"], self.rec[i], self.gdn_out, xq4,
                       xd4, NK, NV, d["gn_eps"], M, mode, None, d["pe_c"])
        d["out"](xq4, xd4, M, self.h, HID, self.h, HID)

    def _gdn_commit(self, i, parity=0, nacc=None):
        """apply nacc + 1 steps of input set `parity` to the GDN-2 conv / recurrent states."""
        E = ext()
        d = self.layers[i]
        n = self.nacc if nacc is None else nacc
        E.gdn2_conv_commit(self.conv[i], self.xsave2[i][parity], n)
        xq4, xd4 = self.xq(VALD)
        E.gdn2_rec(self.mixed2[i][parity], self.qkvz[:, CONV:], CONV + VALD, self.gd2[i][parity], self.be2[i][parity],
                   self.bw2[i][parity], d["gn"], self.rec[i], self.gdn_out, xq4, xd4, NK, NV, d["gn_eps"], 1, 2, n,
                   d["pe_c"])

    @torch.no_grad()
    def finalize_pending(self, last_parity):
        """after fused-mode cycles: apply the last cycle's pending commit (input set last_parity)."""
        for i in self.lin_ids:
            self._gdn_commit(i, last_parity, self.nacc_prev)
        self.nacc_prev.fill_(-1)

    def _mla(self, i, d, M):
        E = ext()
        r, NQD = d["r"], self.NQD[i]
        xq, xd = self.xq(HID)
        E.rms_rows(self.h, HID, d["ln1"], self.x, HID, xq, xd, M, HID, self.eps)
        d["qdk"](xq, xd, M, self.yq, NQD)
        E.dgemv(d["Wr"], HID, RD, HID, self.x, HID, self.kt0, RD, M, True)
        E.dgemv(d["R0"], RD, RD, RD, self.kt0, RD, self.kr, RD, M, True)
        c = self.yq[:, NH * 2 * HD:]
        E.dgemv(d["R"], r, r, r, c, NQD, self.crot, r, M, True)
        E.mla_encode(self.yq[:, NH * 2 * HD + r:], NQD, self.kr, self.crot, r, d["cbl"], d["cbr"], self.pos, d["cache"],
                     d["RW"], d["k_eps"], M)
        Ep = d["Ep"]
        Q = self.Q.view(-1)[:M * NH * Ep]
        E.mla_query(self.yq, NQD, d["qn"], d["q_eps"], d["P"], d["E"], Ep, self.pos, self.inv_freq, Q, M)
        po = self.part_o.view(-1)[:self.NS * M * NH * (Ep - RD)]
        pm = self.part_ml.view(-1)[:self.NS * M * NH * 2].view(self.NS, -1)
        olat = self.olat.view(-1)[:M * NH * r]
        attn = E.mla_attn_fd if self.attn == "fd" else E.mla_attn
        attn(Q, Ep, d["cache"], d["RW"], d["R0h"], d["cbl"], d["cbr"], self.pos, self.inv_freq, po, pm, olat, r,
             d["scale"], M)
        E.mla_vup(olat, r, d["Wv"], self.yq, NQD, self.yattn, M)
        xq2, xd2 = self.xq(NH * HD)
        E.quant_rows(self.yattn, NH * HD, xq2, xd2, M, NH * HD)
        d["o"](xq2, xd2, M, self.h, HID, self.h, HID)

    def _mlp(self, d, M):
        E = ext()
        xq, xd = self.xq(HID)
        E.rms_rows(self.h, HID, d["ln2"], None, 0, xq, xd, M, HID, self.eps)
        d["gu"](xq, xd, M, self.gu, 2 * FF)
        xq2, xd2 = self.xq(FF)
        E.silu_mul_q(self.gu, 2 * FF, None, xq2, xd2, M, FF)
        d["down"](xq2, xd2, M, self.h, HID, self.h, HID)

    def _block(self, M, mode, parity=0):
        """M tokens tok_in[:M] at positions pos .. pos + M - 1 -> logits[:M], tok_out[:M]. mode 0: plain (states written,
        pos += M), 1: verify (GDN states untouched; commit() later), 3: fused (pending commit of the previous cycle
        applied first, see _gdn)."""
        E = ext()
        self.embed(self.tok_in, self.h, M)
        for i, d in enumerate(self.layers):
            if i in self.lin_ids:
                self._gdn(i, d, M, mode, parity)
            else:
                self._mla(i, d, M)
            self._mlp(d, M)
        xq, xd = self.xq(HID)
        E.rms_rows(self.h, HID, self.norm_f, self.x, HID, xq, xd, M, HID, self.eps)   # x = normed final hidden (MTP input)
        self.head(xq, xd, M, self.logits, self.Vrows)
        E.argmax_rows(self.logits, self.Vrows, self.V, self.tok_out, M)
        if mode == 0:
            self.pos.add_(M)
            if M == 1:
                self.tok_in[:1].copy_(self.tok_out[:1])

    # ------------------------------------------------------------------------------------------------ graphs
    def _capture(self, key, fn):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        saved = self.save_state()
        with torch.cuda.stream(s):
            fn()                                          # warm-up (allocator, attributes)
        torch.cuda.current_stream().wait_stream(s)
        self.load_state(saved)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        self.load_state(saved)
        self.graphs[key] = g
        return g

    def save_state(self):
        return {"pos": self.pos.clone(), "tok_in": self.tok_in.clone(), "conv": {i: v.clone() for i, v in self.conv.items()},
                "rec": {i: v.clone() for i, v in self.rec.items()}, "nacc_prev": self.nacc_prev.clone(),
                "inputs": {i: [t.clone() for t in self.mixed2[i] + self.gd2[i] + self.be2[i] + self.bw2[i] + self.xsave2[i]]
                           for i in self.lin_ids}}

    def load_state(self, s):
        self.pos.copy_(s["pos"]); self.tok_in.copy_(s["tok_in"]); self.nacc_prev.copy_(s["nacc_prev"])
        for i in self.lin_ids:
            for dst, src in zip(self.mixed2[i] + self.gd2[i] + self.be2[i] + self.bw2[i] + self.xsave2[i], s["inputs"][i]):
                dst.copy_(src)
        for i in self.conv:
            self.conv[i].copy_(s["conv"][i]); self.rec[i].copy_(s["rec"][i])

    @torch.no_grad()
    def step(self, graph=True):
        """plain greedy decode of one token (feeds tok_in[0], leaves the next greedy token in tok_in[0])."""
        if not graph:
            self._block(1, 0)
            return
        g = self.graphs.get("plain") or self._capture("plain", lambda: self._block(1, 0))
        g.replay()

    # ------------------------------------------------------------------------------------------------ state hand-off
    @torch.no_grad()
    def load(self, snap):
        """state after a prompt, from experiments/mercurius/prep.py snapshots (deployed transformers prefill)."""
        T = int(snap["ids"].shape[0])
        assert T + 64 < self.Tmax
        for i in self.attn_ids:
            d, a = self.layers[i], snap["attn"][i]
            RW, r = d["RW"], d["r"]
            rows = torch.zeros(T, RW, dtype=torch.uint8)
            rows[:, 0:4] = a["pos"].to(torch.int32).contiguous().view(torch.uint8).view(T, 4)
            rows[:, 4:20] = a["inv_r"].float().contiguous().view(torch.uint8).view(T, 16)
            rows[:, 20:22] = a["lat_norm"].half().contiguous().view(torch.uint8).view(T, 2)
            rows[:, 22:24] = a["rope_norm"].half().contiguous().view(torch.uint8).view(T, 2)
            rows[:, 24:24 + RD // 2] = a["rope_codes"]
            rows[:, 24 + RD // 2:24 + RD // 2 + (r + 1) // 2] = a["lat_codes"]
            d["cache"][:T].copy_(rows.to(DEV))
        for i in self.lin_ids:
            cs, rs = snap["lin"][i]
            self.conv[i].copy_(cs.float().to(DEV)[:, -4:])
            self.rec[i].copy_(rs.float().to(DEV))
        self.pos.fill_(T)
        self.nacc_prev.fill_(-1)
        ll = snap["last_logits"].float()
        self.tok_in.zero_()
        self.tok_in[0] = int(ll[:self.V].argmax())
        self.logits.zero_()
        self.logits[0, :ll.shape[0]].copy_(ll.to(DEV))
        self.T0 = T

    @torch.no_grad()
    def generate(self, n, graph=True):
        out = torch.empty(n, dtype=torch.long, device=DEV)
        for k in range(n):
            out[k] = self.tok_in[0]
            self.step(graph)
        return out
