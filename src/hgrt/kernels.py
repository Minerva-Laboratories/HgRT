"""Weight-only quantized GEMVs for single-token decode (csrc/gemv.cu): JIT-built once, cached under ~/.cache/hgrt."""
import os

import torch

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability()
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        bd = os.path.join(os.environ.get("HGRT_BUILD_DIR", os.path.expanduser("~/.cache/hgrt")), "gemv")
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "gemv.cu")
        _EXT = load("hgrt_gemv", [src], build_directory=bd, verbose=False,
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler"])
    return _EXT


def quant_w4(w, chunk=4096):
    """(N, K) float -> (codes (N, K/32, 4) int32, scales (N, K/32) half): llama.cpp Q4_0 rounding (d = signed absmax / -8).
    Row chunks keep the fp32 temporaries small."""
    if w.shape[0] > chunk:
        parts = [quant_w4(w[i:i + chunk], chunk) for i in range(0, w.shape[0], chunk)]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    N, K = w.shape
    g = w.float().view(N, K // 32, 32)
    idx = g.abs().argmax(-1, keepdim=True)
    d = (g.gather(-1, idx) / -8).squeeze(-1)
    inv = torch.where(d != 0, 1 / d, torch.zeros_like(d))
    q = (g * inv[..., None] + 8.5).floor().clamp(0, 15).to(torch.int64)          # Q4_0: (int8)(x*id + 8.5), min 15
    words = (q.view(N, K // 32, 4, 8) << (4 * torch.arange(8, device=w.device))).sum(-1)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.contiguous(), d.half().contiguous()


def dequant_w4(codes, scales):
    N, G, _ = codes.shape
    q = (codes.to(torch.int64) & 0xFFFFFFFF)[..., None] >> (4 * torch.arange(8, device=codes.device))
    q = (q & 15).view(N, G, 32).float() - 8
    return (q * scales.float()[..., None]).view(N, G * 32)


def quant_w8(w, chunk=16384):
    if w.shape[0] > chunk:
        parts = [quant_w8(w[i:i + chunk], chunk) for i in range(0, w.shape[0], chunk)]
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    s = w.float().abs().amax(1).clamp_min(1e-12) / 127
    q = (w.float() / s[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q.contiguous(), s.float().contiguous()


class W4Linear:
    """y = x W^T for one token, 4-bit weights; fused residual add when res is given."""

    def __init__(self, w, variant=1):
        self.codes, self.scales = quant_w4(w)
        self.N, self.K = w.shape
        self.variant = variant

    def __call__(self, x, out=None, res=None):
        y = out if out is not None else torch.empty(self.N, dtype=torch.bfloat16, device=x.device)
        ext().w4(self.codes, self.scales, x.reshape(-1), None if res is None else res.reshape(-1), y, self.variant)
        return y

    def dequant(self):
        return dequant_w4(self.codes, self.scales)


class W8Linear:
    def __init__(self, w):
        self.codes, self.scales = quant_w8(w)
        self.N, self.K = w.shape

    def __call__(self, x, out=None):
        y = out if out is not None else torch.empty(self.N, dtype=torch.float32, device=x.device)
        ext().w8(self.codes, self.scales, x.reshape(-1), y)
        return y


_OPS = None


def ops():
    """fused single-token ops (csrc/ops.cu)."""
    global _OPS
    if _OPS is None:
        from torch.utils.cpp_extension import load
        ext()                                                     # sets TORCH_CUDA_ARCH_LIST
        bd = os.path.join(os.environ.get("HGRT_BUILD_DIR", os.path.expanduser("~/.cache/hgrt")), "ops")
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "ops.cu")
        _OPS = load("hgrt_ops", [src], build_directory=bd, verbose=False,
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler"])
    return _OPS
