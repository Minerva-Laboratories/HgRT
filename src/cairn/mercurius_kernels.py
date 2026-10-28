"""JIT build of csrc/mercurius_ops.cu (Mercurius decode kernels), cached under $CAIRN_BUILD_DIR (default ~/.cache/cairn)."""
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
        bd = os.path.join(os.environ.get("CAIRN_BUILD_DIR", os.path.expanduser("~/.cache/cairn")), "mercurius")
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "mercurius_ops.cu")
        flags = ["-O3", "--use_fast_math", "-allow-unsupported-compiler"]
        if os.environ.get("CAIRN_PTXAS_V"):
            flags.append("-Xptxas=-v")
        _EXT = load("cairn_mercurius", [src], build_directory=bd, verbose=bool(os.environ.get("CAIRN_BUILD_VERBOSE")),
                    extra_cuda_cflags=flags)
    return _EXT


def nf4_tables(code):
    """int8 tables h, l with 128 h + l = round(16256 L) for the 16 NF4 levels L (|L - (128 h + l) / 16256| <= 3.1e-5)."""
    L = code.double().cpu()
    h = torch.round(L * 127)
    l = torch.round(L * 16256 - 128 * h)
    assert h.abs().max() <= 127 and l.abs().max() <= 127
    return h.to(torch.int8), l.to(torch.int8), (128 * h + l) / 16256


def pack_nf4(packed_low_first, absmax):
    """(out, in/2) uint8 codes (element 2i in the low nibble) + (out, in/64) absmax -> kernel layout:
    codes viewed as (out, in/32) 16-byte words (int32 x 4), scales absmax / 16256 (fp32)."""
    out, half = packed_low_first.shape
    codes = packed_low_first.contiguous().view(torch.int32).view(out, half // 16, 4)
    return codes, (absmax.float() / 16256.0).contiguous()


def mma_layout(packed_low_first):
    """(out, in/2) uint8 codes (element 2i low nibble) -> nf4_mma layout (out, in/128, 4, 4) int32: per 128 values,
    thread t's 4 words = for group j: halfword(bytes 16j + 2t, +1) | halfword(bytes 16j + 8 + 2t, +1) << 16."""
    out, half = packed_low_first.shape
    assert half % 64 == 0
    b = packed_low_first.contiguous().view(out, half // 64, 4, 16)           # (row, super, group j, byte)
    h = b.view(torch.int16).view(out, half // 64, 4, 8)                       # halfwords: (row, super, j, 8)
    t = torch.arange(4, device=b.device)
    lo, hi = h[..., t], h[..., 4 + t]                                         # (row, super, j, t)
    w = (lo.to(torch.int32) & 0xFFFF) | (hi.to(torch.int32) << 16)
    return w.permute(0, 1, 3, 2).contiguous()                                 # (row, super, t, j)
