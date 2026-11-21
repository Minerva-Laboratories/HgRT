"""Log speculative blocks at fixed depth K (default 3): per block the draft confidences (top-1 probability, entropy over
the draft vocabulary), the accepted count, the context length, and timings. Plus per-context costs of a verification
pass for M = 1..4 rows and of one MTP draft step. Output: JSON lines.
    python dyn_log.py <kv> <out.jsonl>"""
import glob, json, os, sys, time, torch
sys.path.insert(0, os.environ["HGRT_SRC"])
from transformers import AutoTokenizer
from hgrt.mercurius_packed import PackedMercuriusEngine
CK = os.environ["HGRT_MERCURIUS05"]
kv, out = sys.argv[1], sys.argv[2]
K, NGEN = int(os.environ.get("K", 3)), int(os.environ.get("NGEN", 192))
tok = AutoTokenizer.from_pretrained(CK)
chat = lambda s: tok(tok.apply_chat_template([{"role": "user", "content": s}], add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)).input_ids
src = "".join(open(f).read() for f in sorted(glob.glob(os.environ["HGRT_SRC"] + "/hgrt/**/*.py", recursive=True)))
body = tok(src).input_ids
prompts = [("chat", chat("Explain how a refrigerator works, step by step, for a curious ten-year-old.")),
           ("code", chat("Write a Python function that parses an ISO 8601 date string without using datetime, with tests.")),
           ("math", chat("A train leaves at 3:40 pm going 72 km/h; a second leaves the same station at 4:10 pm going 90 km/h. "
                         "When does the second catch up? Show the reasoning.")),
           ("json", chat("Convert this robot command into JSON with keys target, anchor, relation, room: "
                         "'bring me the red mug from the kitchen table'. Answer with the JSON only."))]
q = chat("Which function in the code above implements the prefill?")
for T in (8192, 16384, 32768):
    prompts.append((f"long{T // 1024}k", (body * (T // len(body) + 1))[:T - len(q)] + q))
e = PackedMercuriusEngine(CK, Tmax=32768 + NGEN + 128, kv=kv, maxM=max(4, K + 1))
e.capture(ks=tuple(range(K + 1)))
f = open(out, "w")
probe = {}
orig_mtp = e._mtp
def mtp(*a, **kw):
    r = orig_mtp(*a, **kw)
    if r is not None:
        lg = e.mtp["logits"][0].float()
        p = torch.softmax(lg, -1)
        probe["s"] = torch.stack([p.max(), -(p * torch.log(p.clamp_min(1e-30))).sum()])
    return r
e._mtp = mtp
def sync(): torch.cuda.synchronize(); return time.perf_counter()
for name, ids in prompts:
    e.generate(ids, 2, k=K)                               # warm (allocations, prompt-mode graphs)
    e.reset(); e.ensure_prefill(len(ids)); e.prefill(ids, mtp=True)
    x = int(e.pick[0]); e.calibrate_fovea(); e.free_prefill(); e.release_prefill()
    out_toks = [x]
    def draft(hrows, x, base, fill):
        ds, ps = [], []
        R = hrows.shape[0]; nxt = fill if fill is not None else [x]
        t0 = sync()
        d = e._mtp(R, hrows, torch.tensor(nxt, device="cuda"), base); ds.append(d); ps.append(probe["s"].tolist())
        hcur = e.mtp["b_out"][R - 1:R].clone()
        for j in range(1, K):
            d = e._mtp(1, hcur, torch.tensor([ds[-1]], device="cuda"), base + R - 1 + j); ds.append(d); ps.append(probe["s"].tolist())
            hcur = e.mtp["b_out"][:1].clone()
        return ds, ps, sync() - t0
    drafts, ps, td = draft(e.hn[:1].clone(), x, len(ids) - 1, None)
    while len(out_toks) < NGEN:
        block = [x] + drafts
        t0 = sync(); e._run(block, slots=True); y = e.pick[:len(block)].tolist(); tv = sync() - t0
        nacc = 0
        while nacc < len(drafts) and drafts[nacc] == y[nacc]: nacc += 1
        emit = y[:nacc + 1]
        f.write(json.dumps({"prompt": name, "kv": kv, "T": int(e.pos), "p": [a for a, b in ps], "H": [b for a, b in ps],
                            "nacc": nacc, "t_draft": td, "t_verify": tv, "R": len(emit)}) + "\n")
        out_toks += emit
        for d in e.layers:
            if d["kind"] == "gdn2":
                d["conv"].copy_(d["conv_slots"][nacc]); d["S"].copy_(d["S_slots"][nacc])
        hrows = e.hn[:nacc + 1].clone(); base = int(e.pos); e.pos.add_(nacc + 1)
        if tok.eos_token_id in emit: break
        x = emit[-1]
        drafts, ps, td = draft(hrows, x, base, emit)
    f.flush()
    print(name, len(ids), "tokens:", len(out_toks), flush=True)
# costs: a verification pass with M = 1..K+1 rows (captured graphs), and one draft step, by context length
for T in (512, 8192, 16384, 32768):
    for M in range(1, K + 2):
        g = e._graph(M, M > 1)
        e.posM[:M].copy_(torch.arange(T, T + M, device="cuda"))
        for _ in range(3): g.replay()
        t0 = sync()
        for _ in range(20): g.replay()
        f.write(json.dumps({"cost": "verify", "kv": kv, "T": T, "M": M, "s": (sync() - t0) / 20}) + "\n")
f.close()
