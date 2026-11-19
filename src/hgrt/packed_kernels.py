"""JIT build of csrc/packed_ops.cu (Q4_0 / Q8_0 kernels for mercurius-packed-v1), cached under $HGRT_BUILD_DIR."""
import os

import torch

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        if "TORCH_CUDA_ARCH_LIST" not in os.environ:
            major, minor = torch.cuda.get_device_capability() if torch.cuda.is_available() else (8, 7)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        bd = os.path.join(os.environ.get("HGRT_BUILD_DIR", os.path.expanduser("~/.cache/hgrt")), "packed")
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "packed_ops.cu")
        _EXT = load("hgrt_packed", [src], build_directory=bd, verbose=bool(os.environ.get("HGRT_BUILD_VERBOSE")),
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler"])
    return _EXT


FMT = {"q4_0": 0, "q8_0": 1}


class PackedLin:
    """one packed projection (or several concatenated along N, same format), y = W x (+ res) for M <= 4 rows.
    codes / scales are the checkpoint's tensors as stored (q4_0: weight_packed int32 (N, K/8); q8_0: weight_q8 int8)."""

    def __init__(self, parts, fmt, R=2, device="cuda"):
        self.fmt, self.R = fmt, R
        key = "weight_packed" if fmt == "q4_0" else "weight_q8"
        self.codes = torch.cat([p[key] for p in parts]).to(device).contiguous()
        self.scales = torch.cat([p["weight_scale"] for p in parts]).to(device, torch.float16).contiguous()
        self.N, self.K = self.scales.shape[0], self.scales.shape[1] * 32

    def __call__(self, x, M, y, ldy=None, res=None, ldr=0, ldx=None):
        ext().packed_gemv(self.codes, self.scales, FMT[self.fmt], x, ldx or self.K, res, ldr, y, ldy or self.N, M, self.R)
        return y

    def dequant(self):
        """(N, K) bf16: the weights the kernel uses."""
        if self.fmt == "q4_0":
            w = self.codes.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            q = ((w.unsqueeze(-1) >> (4 * torch.arange(8, device=w.device))) & 15).reshape(self.N, -1, 32).float() - 8
        else:
            q = self.codes.view(torch.int8).float().reshape(self.N, -1, 32)
        return (q * self.scales.float().unsqueeze(-1)).reshape(self.N, self.K).to(torch.bfloat16)
