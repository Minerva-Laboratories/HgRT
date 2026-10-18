"""Load a compressed-tensors `pack-quantized` int4 checkpoint (llm-compressor AWQ, symmetric, groups of 32) for the engine.

The packed codes are already in the engine's 4-bit layout: weight_packed (N, K/8) int32 with nibble i of word j =
column 8j + i, code - 8 = signed weight, one scale per 32 columns, i.e. the same numerics as the engine's Q4_0 GEMV.
They are used as stored (no re-quantization). The Hugging Face model is overwritten in place with the checkpoint:
quantized linears dequantized to bf16 (for the prompt pass), everything else as stored (AWQ folds its activation
scales into the norms, so the norms differ from the base model's)."""
import os

import torch


@torch.no_grad()
def load_awq(hf_model, path, device="cuda"):
    """-> {hf parameter name: (codes (N, K/32, 4) int32, scales (N, K/32) fp16)}"""
    from safetensors import safe_open
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".safetensors")]
    hs = [safe_open(f, "pt", device="cpu") for f in files]
    where = {k: h for h in hs for k in h.keys()}
    params = dict(hf_model.named_parameters())
    packed = {}

    def ck(name):                               # hf name -> checkpoint name
        return "lm_head.weight" if name == "lm_head.weight" else "model.language_model." + name[len("model."):]

    for name, p in params.items():
        c = ck(name)
        base = c[:-len(".weight")] if c.endswith(".weight") else None
        if base and base + ".weight_packed" in where:
            pk = where[base + ".weight_packed"].get_tensor(base + ".weight_packed").to(device)
            sc = where[base + ".weight_scale"].get_tensor(base + ".weight_scale").to(device)
            N, K = p.shape
            assert sc.shape == (N, K // 32), (name, sc.shape)
            codes = pk.view(N, K // 32, 4).contiguous()
            packed[name] = (codes, sc.to(torch.float16).contiguous())
            q = ((codes.to(torch.int64) & 0xFFFFFFFF)[..., None] >> (4 * torch.arange(8, device=device))) & 15
            p.copy_(((q.view(N, K // 32, 32).float() - 8) * sc.float()[..., None]).view(N, K).to(p.dtype))
        elif c in where:
            p.copy_(where[c].get_tensor(c).to(device=p.device, dtype=p.dtype))
        else:
            raise KeyError(c)
    return packed
