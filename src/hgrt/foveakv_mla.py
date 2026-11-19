"""FoveaKV for the CARE-latent MLA cache, as Mercurius 0.5 was trained (Mercurius-Pipeline quant/kv.py FoveaKV, the
reference scheme of github.com/Minerva-Laboratories/FoveaKV): per 128-channel head of the latent (the 64-channel RoPE
key is one head of its own) -- block mean (128-token blocks, fp16), randomized Hadamard rotation, 8-bit log gain whose
range is calibrated on the prompt, 16-level Gaussian Lloyd-Max shape nested as 2-bit coarse + 2-bit residual planes,
shrinkage (keys x (1 - D)^-1/2, values x 1 / (1 - D)).

The engine encodes a token once, when it leaves the exact window (its block is complete by then), into its CODES:
the 4-bit fine index per rotated channel (nibble-packed; the coarse plane is its top 2 bits), an 8-bit gain code per
head, and the block's mean (fp16). The decode attention (csrc/packed_ops.cu fovea_codes_attn) reads the codes: with
x_hat = M + (g lev s) @ R, q . x_hat = q . M + g s sum_d (R q)_d lev_d, so only the query is rotated.
"""
import math

import torch

_CB = {}


def _phi(x):
    return torch.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def _cond_mean(a, b):
    return (_phi(a) - _phi(b)) / (_Phi(b) - _Phi(a)).clamp_min(1e-300)


def gaussian_lloyd_max(n, iters=500):
    lev = torch.linspace(-2.5, 2.5, n, dtype=torch.float64)
    inf = torch.tensor([float("inf")], dtype=torch.float64)
    for _ in range(iters):
        bnd = (lev[1:] + lev[:-1]) / 2
        e = torch.cat([-inf, bnd, inf])
        lev = _cond_mean(e[:-1], e[1:])
    return (lev[1:] + lev[:-1]) / 2, lev


def nested_codebook():
    """(bnd16, lut16, lut4, mse_fine, mse_coarse), fp64: FoveaKV's NestedCodebook(4, 2)."""
    if "nested" not in _CB:
        bnd, lut16 = gaussian_lloyd_max(16)
        inf = torch.tensor([float("inf")], dtype=torch.float64)
        e = torch.cat([-inf, bnd, inf])
        ce = e[::4]
        if ce[-1] != float("inf"):
            ce = torch.cat([ce, inf])
        lut4 = _cond_mean(ce[:-1], ce[1:])

        def mse(b, lv):
            ee = torch.cat([-inf, b, inf])
            tot = 0.0
            for i, c in enumerate(lv):
                a, bb = ee[i], ee[i + 1]
                p = _Phi(bb) - _Phi(a)
                m2 = p - (bb * _phi(bb) if torch.isfinite(bb) else 0.0) + (a * _phi(a) if torch.isfinite(a) else 0.0)
                tot += float(m2 - 2 * c * (_phi(a) - _phi(bb)) + c * c * p)
            return tot
        _CB["nested"] = (bnd, lut16, lut4, mse(bnd, lut16), mse(ce[1:-1], lut4))
    return _CB["nested"]


def hadamard(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def rht(n, seed):
    import numpy as np
    s = torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], n))
    return hadamard(n) * s.unsqueeze(0) / math.sqrt(n)


class FoveaCodec:
    """encode / calibrate one cached tensor family (the latent: roles k and v; k_rope: role k) of one layer."""

    def __init__(self, C, hd, block, gain_bits, shrink, seed, roles, dev="cuda"):
        self.C, self.hd = C, (hd if C % hd == 0 else C)
        self.block, self.n, self.roles = block, 1 << gain_bits, roles
        bnd, lut16, lut4, Df, Dc = nested_codebook()
        self.bnd, self.lut16, self.lut4 = (t.to(dev, torch.float32) for t in (bnd, lut16, lut4))
        self.R = rht(self.hd, seed).to(dev).to(torch.float32)
        sk = lambda D, role: ((1 - D) ** -0.5 if role == "k" else 1.0 / (1 - D)) if shrink else 1.0
        self.s = {(pl, role): sk(D, role) for pl, D in (("f", Df), ("c", Dc)) for role in roles}
        nh = C // self.hd
        self.lo = torch.zeros(nh, device=dev)
        self.span = torch.ones(nh, device=dev)

    def _heads(self, x):
        return x.float().view(*x.shape[:-1], -1, self.hd)

    @torch.no_grad()
    def calibrate(self, x, P):
        """gain range over the prompt's tokens x[:P] (T, C), centred on the prompt's block means (a trailing partial
        block on its own partial mean), as FoveaKV.encode does with the prefill split P."""
        xf = self._heads(x[:P])
        M = torch.empty_like(xf)
        nb = P // self.block
        if nb:
            M[:nb * self.block] = xf[:nb * self.block].view(nb, self.block, *xf.shape[1:]).mean(1).repeat_interleave(self.block, 0)
        if P % self.block:
            M[nb * self.block:] = xf[nb * self.block:].mean(0, keepdim=True)
        M = M.half().float()
        lg = torch.log(((xf - M) @ self.R.T).norm(dim=-1).clamp_min(1e-30))            # (P, heads)
        self.lo.copy_(lg.amin(0))
        self.span.copy_((lg.amax(0) - self.lo).clamp_min(1e-12))

    @torch.no_grad()
    def encode(self, x, s_idx, out):
        """tokens s_idx (n,) of the cache x (Tmax, C), their 128-token blocks complete -> out[(plane, role)][s_idx]."""
        b0 = (s_idx // self.block) * self.block
        rows = b0.unsqueeze(1) + torch.arange(self.block, device=x.device)                # (n, block)
        M = self._heads(x[rows]).mean(1).half().float()                                     # (n, heads, hd)
        y = (self._heads(x[s_idx]) - M) @ self.R.T
        g = y.norm(dim=-1)
        z = y / g.clamp_min(1e-30).unsqueeze(-1) * math.sqrt(self.hd)
        idx = torch.bucketize(z, self.bnd)
        c = torch.floor((torch.log(g.clamp_min(1e-30)) - self.lo) / self.span * self.n).clamp(0, self.n - 1)
        gq = torch.exp(self.lo + (c + 0.5) / self.n * self.span).unsqueeze(-1) / math.sqrt(self.hd)
        for pl, lev in (("f", self.lut16[idx]), ("c", self.lut4[idx >> 2])):
            for role in self.roles:
                xh = (M + (gq * lev * self.s[pl, role]) @ self.R).reshape(len(s_idx), self.C)
                out[pl, role].index_copy_(0, s_idx, xh.to(out[pl, role].dtype))

    @torch.no_grad()
    def encode_codes(self, rows, s_idx, codes, gains, means):
        """tokens s_idx (n,), blocks complete; rows(idx) -> exact rows (len(idx), C) of any positions. Writes
        codes[s] (C/2 uint8, channel 2i low nibble), gains[s] (heads, uint8), means[s // block] (C fp16)."""
        n = s_idx.shape[0]
        b0 = (s_idx // self.block) * self.block
        blk = (b0.unsqueeze(1) + torch.arange(self.block, device=s_idx.device)).reshape(-1)
        Mf = self._heads(rows(blk).view(n, self.block, self.C)).mean(1).half()                  # (n, heads, hd) fp16
        y = (self._heads(rows(s_idx)) - Mf.float()) @ self.R.T
        g = y.norm(dim=-1)
        z = y / g.clamp_min(1e-30).unsqueeze(-1) * math.sqrt(self.hd)
        idx = torch.bucketize(z, self.bnd).reshape(n, self.C).to(torch.uint8)
        c = torch.floor((torch.log(g.clamp_min(1e-30)) - self.lo) / self.span * self.n).clamp(0, self.n - 1)
        codes.index_copy_(0, s_idx, (idx[:, 0::2] | (idx[:, 1::2] << 4)).contiguous())
        gains.index_copy_(0, s_idx, c.to(torch.uint8).view(n, -1)[:, :gains.shape[1]] if gains.dim() == 2 else c.to(torch.uint8).view(n))
        means.index_copy_(0, s_idx // self.block, Mf.reshape(n, self.C))
