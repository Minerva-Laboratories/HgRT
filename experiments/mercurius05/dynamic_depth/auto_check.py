"""k = "auto" (adaptive depth, up to KMAX drafts) vs fixed k: identical tokens to plain greedy, ms per token, depths used."""
import glob, os, sys, time, torch
sys.path.insert(0, os.environ["HGRT_SRC"])
from hgrt.mercurius_packed import PackedMercuriusEngine
from transformers import AutoTokenizer
CK = os.environ["HGRT_MERCURIUS05"]
KMAX = int(os.environ.get("KMAX", 7)); NGEN = int(os.environ.get("NGEN", 192))
tok = AutoTokenizer.from_pretrained(CK)
chat = lambda s: tok(tok.apply_chat_template([{"role": "user", "content": s}], add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)).input_ids
src = "".join(open(f).read() for f in sorted(glob.glob(os.environ["HGRT_SRC"] + "/hgrt/**/*.py", recursive=True)))
body = tok(src).input_ids
q = chat("Which function in the code above implements the prefill?")
prompts = [("chat", chat("Describe the water cycle for a high-school class, with one example per stage.")),
           ("code", chat("Write a Python class implementing an LRU cache with O(1) get and put, with docstrings.")),
           ("json", chat("Return a JSON list of 12 objects for the planets and dwarf planets, keys name, type, moons.")),
           ("long8k", (body * 9)[:8192 - len(q)] + q), ("long32k", (body * 30)[:32768 - len(q)] + q)]
kv = sys.argv[1]
e = PackedMercuriusEngine(CK, Tmax=32768 + NGEN + 128, kv=kv, maxM=KMAX + 1)
e.capture(ks=tuple(range(KMAX + 1)))
for name, ids in prompts:
    res = {}
    for k in (0, 2, 3, "auto"):
        e.generate(ids, 4, k=k)
        torch.cuda.synchronize(); t0 = time.time(); _ = e.generate(ids, 1, k=k); torch.cuda.synchronize(); t1 = time.time() - t0
        torch.cuda.synchronize(); t0 = time.time(); out, st = e.generate(ids, NGEN, k=k); torch.cuda.synchronize()
        res[k] = out
        ms = (time.time() - t0 - t1) / (len(out) - 1) * 1e3
        print(f"kv={kv} {name:8s} k={k!s:5s} {ms:6.2f} ms/token identical={out == res[0]}" +
              (f" depths={st['depths']}" if k == "auto" else ""), flush=True)
    e._policy.hist = [0] * (KMAX + 1)
