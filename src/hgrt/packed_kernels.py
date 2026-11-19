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
_WS = None                       # bf16 workspace of the prefill GEMMs (the largest projection)
ROUND = {"bf16": False}          # True: weights used as bf16((q - 8) d) (the PyTorch QAT forward); False: exact (q - 8) d
                                 # with the scale factored out of each group (GGUF / llama.cpp numerics, faster)


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
        if M > 4:
            return self.mm(x, M, y, res)
        f = ext().packed_gemv if ROUND["bf16"] else ext().packed_gemv_u
        f(self.codes, self.scales, FMT[self.fmt], x, ldx or self.K, res, ldr, y, ldy or self.N, M, self.R)
        return y

    def mm(self, x, M, y, res=None):
        """M > 4 rows (prefill): y[:M, :N] = x[:M, :K] @ W^T (+ res) as one bf16 GEMM with fp32 accumulation, W =
        bf16((q - 8) d) dequantized into a shared workspace (the pipeline's reconstructed weights)."""
        global _WS
        n = self.N * self.K
        if _WS is None or _WS.numel() < n:
            _WS = torch.empty(n, device=self.codes.device, dtype=torch.bfloat16)
        W = _WS[:n].view(self.N, self.K)
        ext().dequant_bf16(self.codes, self.scales, FMT[self.fmt], W)
        xx, yy = x[:M, :self.K], y[:M, :self.N]
        if yy.dtype != torch.bfloat16:
            yy.copy_(torch.mm(xx, W.t()) if res is None else torch.addmm(res[:M, :self.N], xx, W.t()))
        elif res is None:
            torch.mm(xx, W.t(), out=yy) if yy.is_contiguous() else yy.copy_(torch.mm(xx, W.t()))
        elif res.data_ptr() == y.data_ptr():
            yy.addmm_(xx, W.t())
        else:
            yy.copy_(torch.addmm(res[:M, :self.N], xx, W.t()))
        return y

    def dequant_exact(self):
        """(N, K) fp32: (q - 8) d / q d exactly (the unrounded kernel's weights)."""
        if self.fmt == "q4_0":
            w = self.codes.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            q = ((w.unsqueeze(-1) >> (4 * torch.arange(8, device=w.device))) & 15).reshape(self.N, -1, 32).float() - 8
        else:
            q = self.codes.view(torch.int8).float().reshape(self.N, -1, 32)
        return (q * self.scales.float().unsqueeze(-1)).reshape(self.N, self.K)

    def dequant(self):
        """(N, K) bf16: bf16((q - 8) d), the weights of the rounded kernel and of the pipeline's reconstruction."""
        if self.fmt == "q4_0":
            w = self.codes.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            q = ((w.unsqueeze(-1) >> (4 * torch.arange(8, device=w.device))) & 15).reshape(self.N, -1, 32).float() - 8
        else:
            q = self.codes.view(torch.int8).float().reshape(self.N, -1, 32)
        return (q * self.scales.float().unsqueeze(-1)).reshape(self.N, self.K).to(torch.bfloat16)
