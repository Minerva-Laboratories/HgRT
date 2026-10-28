"""The whole stack end to end: perplexity with a DepthSkip prompt pass and a compressed / selected decode cache, against
the exact prompt pass and bf16 decode. Protocol as stack_ppl.py (L-N prompt tokens, N = 256 teacher-forced).
Env: HGRT_MODEL, HGRT_LENS (16384,32768), HGRT_NDOC (3), HGRT_SRCS (prose,code), HGRT_PLANS
(exact,g512,f64_g512,cut16), HGRT_DECODES (bf16,stack10,stack20), HGRT_OUT (results/<model>/e2e_ppl.jsonl,
resumable), HGRT_MAPS_DIR."""
import json, math, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from hgrt.pipeline import Pipeline                                   # noqa: E402
from sparsepage.qwen import DEV, code_text, doc_windows, prose_text      # noqa: E402

MODEL = os.environ.get("HGRT_MODEL", "Qwen/Qwen3.5-9B")
LENS = [int(x) for x in os.environ.get("HGRT_LENS", "16384,32768").split(",")]
NDOC, N = int(os.environ.get("HGRT_NDOC", 3)), 256
SRCS = os.environ.get("HGRT_SRCS", "prose,code").split(",")
PLANS = os.environ.get("HGRT_PLANS", "exact,g512,f64_g512,cut16").split(",")
DECODES = os.environ.get("HGRT_DECODES", "bf16,stack10,stack20").split(",")
tag = MODEL.rstrip("/").split("/")[-1].lower()
out_p = os.environ.get("HGRT_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", tag, "e2e_ppl.jsonl"))
os.makedirs(os.path.dirname(out_p), exist_ok=True)
done = {(r["src"], r["L"], r["doc"], r["plan"], r["decode"]) for r in map(json.loads, open(out_p))} if os.path.exists(out_p) else set()

P = Pipeline(MODEL)
for c in sorted({c for p in PLANS if p != "exact" for _, c in __import__("hgrt.pipeline").pipeline.PREFILLS[p] if c}):
    t = time.time()
    P.maps(c)
    print(f"maps cut {c} ready ({time.time() - t:.0f}s)", flush=True)
t0 = time.time()
for src in SRCS:
    text = prose_text() if src == "prose" else code_text()
    for L in LENS:
        for d, ids in enumerate(doc_windows(P.R.tok, text, L, NDOC, seed=L)):
            x = torch.tensor(ids, device=DEV)
            for plan in PLANS:
                todo = [s for s in DECODES if (src, L, d, plan, s) not in done]
                if not todo:
                    continue
                torch.cuda.synchronize(); tp = time.time()
                snap = P.prefill(x[:L - N], plan)
                torch.cuda.synchronize(); prefill_s = time.time() - tp
                for s in todo:
                    ts = time.time()
                    nll, st = P.perplexity(snap, x[L - N:], s)
                    rec = dict(src=src, L=L, doc=d, plan=plan, decode=s, nll=float(nll.mean()),
                               nll_tok=[float(v) for v in nll], bits=st.get("read_bits", 16.0), prefill_s=prefill_s)
                    with open(out_p, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"[{time.time() - t0:6.0f}s] {src} L={L} doc{d} {plan:9s} {s:8s} PPL {math.exp(nll.mean()):8.4f} "
                          f"bits {rec['bits']:.2f} prefill {prefill_s:.0f}s ({time.time() - ts:.0f}s)", flush=True)
                del snap
                torch.cuda.empty_cache()
print("E2E_PPL_DONE")
