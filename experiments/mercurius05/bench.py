"""Mercurius-0.5 engine benchmark: prefill time, decode ms/token (plain and MTP k drafts, CUDA graphs), MLA cache size,
peak GPU memory, and whether speculative decoding reproduced the plain greedy tokens.

    python experiments/mercurius05/bench.py /path/to/Mercurius-0.5-4B-Preview --kv foveakv --ctx 512,8192 --n 128 --ks 2,auto

The prompt is HgRT's own source code (repeated to the length) followed by a question about it, in the chat template.
One line per (context, k) is printed and appended as JSON to --out.
"""
import argparse
import glob
import json
import os
import time

import torch
from transformers import AutoTokenizer

from hgrt.mercurius_packed import PackedMercuriusEngine

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default=os.environ.get("HGRT_MERCURIUS05"))
ap.add_argument("--kv", default="foveakv", choices=["foveakv", "exact"])
ap.add_argument("--ctx", default="512,8192")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--ks", default="2,auto")             # fixed draft counts and / or auto (adaptive, up to 7)
ap.add_argument("--out", default="")
a = ap.parse_args()
ctxs = [int(x) for x in a.ctx.split(",")]
ks = [0] + [x if x == "auto" else int(x) for x in a.ks.split(",") if x == "auto" or int(x) > 0]

tok = AutoTokenizer.from_pretrained(a.ckpt)
src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "hgrt")
body = tok("".join(open(f).read() for f in sorted(glob.glob(src + "/**/*.py", recursive=True)))).input_ids
q = tok(tok.apply_chat_template([{"role": "user", "content": "Which function in the code above implements the prefill?"}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)).input_ids

e = PackedMercuriusEngine(a.ckpt, Tmax=max(ctxs) + a.n + 64, kv=a.kv)
e.capture()                                        # k = 0 .. 7
cache = lambda: sum(t.numel() * t.element_size() for d in e.layers if d["kind"] == "mla" for k_, t in d.items()
                    if isinstance(t, torch.Tensor) and ((k_ in ("kr", "c", "ir") and a.kv == "exact") or k_.startswith("f_")))
for P in ctxs:
    prompt = (body * (P // len(body) + 1))[:P - len(q)] + q
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time(); e.generate(prompt, 1, k=0); torch.cuda.synchronize(); tp = time.time() - t0
    ref = None
    for k in ks:
        torch.cuda.synchronize(); t0 = time.time(); out, st = e.generate(prompt, a.n + 1, k=k); torch.cuda.synchronize()
        ms = (time.time() - t0 - tp) / a.n * 1000
        ref = out if k == 0 else ref
        r = {"kv": a.kv, "context": P, "k": k, "decode_ms_per_token": round(ms, 2), "prefill_s": round(tp, 2),
             "mla_cache_mib": round(cache() / 2 ** 20, 1), "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}
        if k:
            r.update(acceptance=round(st["acceptance"], 3), lossless=out == ref)
        if k == "auto":
            r.update(depths=list(st["depths"]))
        print(json.dumps(r))
        if a.out:
            open(a.out, "a").write(json.dumps(r) + "\n")
