"""Adaptive depth benchmark: 11 prompts x k in (0, 2, 3, auto [up to 7 drafts]) x REPS runs (median), NGEN tokens each.
Every run checked identical to plain greedy. One engine per cache, graphs for k = 0 .. 7 (the policy's statistics carry
over between calls, as in normal use; each prompt gets one short warm-up call per k). JSON lines to argv[2]."""
import glob, json, os, statistics, sys, time, torch
sys.path.insert(0, os.environ["HGRT_SRC"])
from hgrt.mercurius_packed import PackedMercuriusEngine
from transformers import AutoTokenizer
CK = os.environ["HGRT_MERCURIUS05"]
KMAX, NGEN, REPS = 7, int(os.environ.get("NGEN", 256)), int(os.environ.get("REPS", 3))
tok = AutoTokenizer.from_pretrained(CK)
chat = lambda s: tok(tok.apply_chat_template([{"role": "user", "content": s}], add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)).input_ids
src = "".join(open(f).read() for f in sorted(glob.glob(os.environ["HGRT_SRC"] + "/hgrt/**/*.py", recursive=True)))
body = tok(src).input_ids
notes = tok(open(os.path.join(os.environ["HGRT_SRC"], "..", "README.md")).read()).input_ids   # HgRT's README
q = chat("Which function in the code above implements the prefill?")
P = [("json-planets", "json", chat("Return a JSON list of 12 objects for the planets and dwarf planets, keys name, type, moons.")),
     ("json-employees", "json", chat("Return a JSON object describing five fictional employees: name, role, skills (list), start_date.")),
     ("json-command", "json", chat('Convert to JSON with keys action, object, location, relation: "pon la taza roja junto al '
                                   'fregadero de la cocina y luego trae el libro azul del salón"')),
     ("chat-science", "chat", chat("Describe the water cycle for a high-school class, with one example per stage.")),
     ("chat-es", "chat", chat("Explica en español cómo funciona la fotosíntesis, en cinco pasos breves.")),
     ("chat-advice", "chat", chat("I have three days in Mexico City. Suggest a plan with museums, food and one day trip.")),
     ("code-lru", "code", chat("Write a Python class implementing an LRU cache with O(1) get and put, with docstrings.")),
     ("code-iso", "code", chat("Write a Python function that parses ISO-8601 durations like P3DT4H5M into seconds, with tests.")),
     ("math", "math", chat("A train leaves at 3:15 pm at 72 km/h; a second leaves the same station at 4:00 pm at 90 km/h on "
                           "the same track. When does the second catch up? Solve step by step.")),
     ("summary-2k", "long", notes[:2000] + chat("List the main features described above.")),
     ("code-8k", "long", (body * 9)[:8192 - len(q)] + q),
     ("code-32k", "long", (body * 30)[:32768 - len(q)] + q)]
kv, outp = sys.argv[1], sys.argv[2]
only = os.environ.get("ONLY")
e = PackedMercuriusEngine(CK, Tmax=32768 + NGEN + 128, kv=kv, maxM=KMAX + 1)
e.capture(ks=tuple(range(KMAX + 1)))
with open(outp, "a") as f:
    for name, cat, ids in P:
        if only and name not in only.split(","):
            continue
        reps = 2 if len(ids) > 16000 else REPS
        base = None
        for k in (0, 2, 3, "auto"):
            e.generate(ids, 8, k=k)
            if k == "auto":
                e._policy.hist = [0] * (KMAX + 1)            # depth histogram of the timed runs only
            ms, hist = [], None
            for _ in range(reps):
                torch.cuda.synchronize(); t0 = time.time(); e.generate(ids, 1, k=k); torch.cuda.synchronize(); t1 = time.time() - t0
                torch.cuda.synchronize(); t0 = time.time(); out, st = e.generate(ids, NGEN, k=k); torch.cuda.synchronize()
                ms.append((time.time() - t0 - t1) / (len(out) - 1) * 1e3)
                base = out if base is None else base
                assert out == base, (name, k)
                hist = list(st["depths"]) if "depths" in st else None
            r = {"kv": kv, "prompt": name, "cat": cat, "P": len(ids), "k": k, "ms": statistics.median(ms), "runs": ms,
                 "ntok": len(out), "depths": hist}
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"kv={kv} {name:15s} k={k!s:5s} {r['ms']:6.2f} ms/token (runs {', '.join(f'{x:.2f}' for x in ms)})"
                  + (f" depths={hist}" if k == "auto" else ""), flush=True)
