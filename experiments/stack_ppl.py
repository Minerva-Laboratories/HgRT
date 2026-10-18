"""End-to-end perplexity of the decode stack on Qwen3.5 against bf16, plain 4-bit, FoveaKV and Riffle on bf16 pages.
Protocol (riffle.qwen): exact prefill of L-N tokens, N = 256 tokens teacher-forced through each cache; prose
(WikiText-2) and code (transformers sources). Env: CAIRN_MODEL (Qwen/Qwen3.5-9B), CAIRN_LENS (16384,32768),
CAIRN_NDOC (3), CAIRN_SRCS (prose,code), CAIRN_SCHEMES, CAIRN_OUT (results/<model>/stack_ppl.jsonl, resumable);
CAIRN_STACK_KERNEL=0 runs the stack's PyTorch reference."""
import json, math, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cairn.caches import factory                                      # noqa: E402
from riffle.qwen import DEV, Runner, code_text, doc_windows, prose_text   # noqa: E402

MODEL = os.environ.get("CAIRN_MODEL", "Qwen/Qwen3.5-9B")
LENS = [int(x) for x in os.environ.get("CAIRN_LENS", "16384,32768").split(",")]
NDOC, N = int(os.environ.get("CAIRN_NDOC", 3)), 256
SRCS = os.environ.get("CAIRN_SRCS", "prose,code").split(",")
SCHEMES = os.environ.get("CAIRN_SCHEMES", "bf16,gs4,dyn10,stack10,stack10-bound,stack10-mean,stack5,stack20").split(",")
tag = MODEL.rstrip("/").split("/")[-1].lower()
out_p = os.environ.get("CAIRN_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", tag, "stack_ppl.jsonl"))
os.makedirs(os.path.dirname(out_p), exist_ok=True)
done = {(r["src"], r["L"], r["doc"], r["scheme"]) for r in map(json.loads, open(out_p))} if os.path.exists(out_p) else set()

R = Runner(MODEL)
t0 = time.time()
for src in SRCS:
    text = prose_text() if src == "prose" else code_text()
    for L in LENS:
        for d, ids in enumerate(doc_windows(R.tok, text, L, NDOC, seed=L)):
            todo = [s for s in SCHEMES if (src, L, d, s) not in done]
            if not todo:
                continue
            x = torch.tensor(ids, device=DEV)
            snap = R.prefill(x[:L - N])
            for s in todo:
                ts = time.time()
                nll, st = R.decode_teacher(factory(s), snap, x[L - N:])
                rec = dict(src=src, L=L, doc=d, scheme=s, nll=float(nll.mean()), nll_tok=[float(v) for v in nll],
                           bits=st.get("read_bits", 16.0))
                with open(out_p, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{time.time() - t0:6.0f}s] {src} L={L} doc{d} {s:14s} PPL {math.exp(nll.mean()):8.4f} "
                      f"bits {rec['bits']:.2f} ({time.time() - ts:.0f}s)", flush=True)
            del snap
            torch.cuda.empty_cache()
print("STACK_PPL_DONE")
