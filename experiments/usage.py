"""Ordinary LLM use through the stack: long prompt in, answer out, timed end to end.

Two workloads on Qwen3.5 (chat template, thinking off, greedy):
  summary  a prose or code document of L tokens + "Summarize the text above." -> MAX_NEW tokens. Measured: time to
           first token (prompt pass + first logits), decode time per token (median over the answer), peak GPU memory,
           KV cache bytes, and how far the answer drifts from the exact configuration's answer (tokens identical before
           the first divergence, share of identical tokens, ROUGE-L F1 of the texts).
  needle   RULER-style retrieval: K "The special magic number for <word> is <7 digits>." sentences at random depths
           in a prose haystack of L tokens, asked for one of them; accuracy = the digits appear in the first 24 tokens.
Configurations "plan+decode": prefill plan (cairn.pipeline.PREFILLS) and decode cache (cairn.caches). The decode
harness runs Qwen3.5's layers one token at a time in eager PyTorch (no CUDA graph), so absolute per-token times include
Python overhead common to every configuration; the attention share is what the stack changes.
  scale    decode speed past the model's 262,144-token window: after the longest summary prompt, the stack caches are
           rebuilt from that prompt's real K/V repeated to SCALE_LENS tokens (positions continue, DeltaNet state from
           the real prompt) and MAX_NEW tokens are decoded. Speed and memory only: the content is a repeat, and Qwen3.5
           needs RoPE scaling beyond 262K, so no quality is claimed there.
One prompt pass per (document, plan), shared by that plan's decode caches; TTFT = prompt pass + cache build + first
logits. Env: CAIRN_MODEL, CAIRN_USAGE_LENS (prose; 4096,16384,32768,65536,131072,262144), CAIRN_CODE_LENS (4096,16384,
32768,65536), CAIRN_USAGE_CONFIGS, CAIRN_USAGE_NEW (128), CAIRN_NEEDLE (8192:10,32768:10,131072:3), CAIRN_SCALE_LENS
(524288,1048576), CAIRN_SCALE_DECODE (stack10), CAIRN_OUT (results/<model>/usage.jsonl, resumable)."""
import json, os, random, re, statistics, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cairn.caches import factory                                       # noqa: E402
from cairn.deps import foveakv_gpu                                     # noqa: E402
from cairn.pipeline import Pipeline                                    # noqa: E402
from riffle.qwen import DEV, code_text, prose_text                     # noqa: E402

MODEL = os.environ.get("CAIRN_MODEL", "Qwen/Qwen3.5-9B")
LENS = [int(x) for x in os.environ.get("CAIRN_USAGE_LENS", "4096,16384,32768,65536,131072,262144").split(",")]
CODE_LENS = [int(x) for x in os.environ.get("CAIRN_CODE_LENS", "4096,16384,32768,65536").split(",") if x]
CONFIGS = os.environ.get("CAIRN_USAGE_CONFIGS", "full+bf16,full+stack10,g512+bf16,g512+stack10,cut24+stack10").split(",")
MAX_NEW = int(os.environ.get("CAIRN_USAGE_NEW", 128))
NEEDLE = [tuple(map(int, x.split(":"))) for x in os.environ.get("CAIRN_NEEDLE", "8192:10,32768:10,131072:3").split(",") if x]
SCALE_LENS = [int(x) for x in os.environ.get("CAIRN_SCALE_LENS", "524288,1048576").split(",") if x]
SCALE_DECODE = os.environ.get("CAIRN_SCALE_DECODE", "stack10")
tag = MODEL.rstrip("/").split("/")[-1].lower()
out_p = os.environ.get("CAIRN_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", tag, "usage.jsonl"))
os.makedirs(os.path.dirname(out_p), exist_ok=True)
rows = [json.loads(l) for l in open(out_p)] if os.path.exists(out_p) else []
done = {(r["task"], r["src"], r["L"], r.get("i", 0), r["config"]) for r in rows}
exact_out = {(r["task"], r["src"], r["L"], r.get("i", 0)): r["out_ids"] for r in rows if r["config"] == CONFIGS[0]}

P = Pipeline(MODEL)
tok = P.R.tok
fg = foveakv_gpu()


def chat(user):
    msgs = [{"role": "user", "content": user}]
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(s, add_special_tokens=False)["input_ids"]


def body(text, n, seed):
    ids = tok(text[: 6 * n + 100000], add_special_tokens=False)["input_ids"]
    s = random.Random(seed).randrange(0, max(1, len(ids) - n))
    return tok.decode(ids[s:s + n])


def kv_bytes(caches):
    return sum(fg.footprint(c)["gpu"] for c in caches.values())


@torch.no_grad()
def prefill(ids, plan):
    torch.cuda.empty_cache()
    torch.cuda.synchronize(); t0 = time.time()
    snap = P.prefill(torch.tensor(ids, device=DEV), plan)
    torch.cuda.synchronize()
    return snap, time.time() - t0


@torch.no_grad()
def decode(snap, prefill_s, decode_name, max_new, stop_digits=False, kv=None, T=None):
    """build the decode caches from the snapshot (or from kv = {layer: (k, v)} with T tokens), then greedy decode."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.time()
    s = snap if kv is None else dict(snap, kv=kv, T=T)
    caches = P.R.build(factory(decode_name), s, extra=max_new + 8)
    nxt = int(snap["last_logits"].argmax())
    torch.cuda.synchronize(); build = time.time() - t0
    lin, pos = snap["lin"].clone(), s["T"]
    out, steps = [nxt], []
    for _ in range(max_new - 1):
        if nxt == tok.eos_token_id or (stop_digits and len(out) > 4 and "\n" in tok.decode(out[-1:])):
            break
        torch.cuda.synchronize(); ts = time.time()
        h = P.R.forward(torch.tensor([nxt], device=DEV), pos, caches, lin, prefill=False)
        nxt = int(P.R.logits(h[-1:])[0].argmax())
        torch.cuda.synchronize(); steps.append(time.time() - ts)
        pos += 1
        out.append(nxt)
    res = dict(ttft_s=prefill_s + build, prefill_s=prefill_s, cache_build_s=build,
               decode_ms_per_token=1e3 * statistics.median(steps) if steps else None,
               peak_gpu_gb=torch.cuda.max_memory_allocated() / 2 ** 30, kv_cache_gb=kv_bytes(caches) / 2 ** 30,
               out_ids=out, text=tok.decode(out), n_ctx=s["T"])
    del caches
    return res


class LazyRepeat(dict):
    """{layer: (k, v)} repeated to S tokens, materialized one layer at a time (8 layers x 4 GB at 1M otherwise)."""

    def __init__(self, kv, S):
        super().__init__(kv)
        self.S = S

    def items(self):
        for i, (k, v) in super().items():
            rep = -(-self.S // k.shape[0])
            yield i, (k.repeat(rep, 1, 1)[:self.S], v.repeat(rep, 1, 1)[:self.S])


def by_plan(configs):
    plans = {}
    for c in configs:
        p, d = c.split("+")
        plans.setdefault(p, []).append(d)
    return plans


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def drift(out, ref):
    first = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), min(len(out), len(ref)))
    same = sum(a == b for a, b in zip(out, ref)) / max(len(ref), 1)
    wa, wb = tok.decode(out).split(), tok.decode(ref).split()
    l = lcs(wa, wb)
    f1 = 0.0 if not l else 2 * l / (len(wa) + len(wb))
    return dict(first_divergence=first, same_token_share=same, rougeL_f1=f1)


def record(rec):
    with open(out_p, "a") as f:
        f.write(json.dumps(rec) + "\n")


t_start = time.time()
# ------------------------------------------------------------------------------------------------ summaries
def scale_test(snap, src, L):
    """decode speed with stack caches rebuilt from this prompt's K/V repeated to each SCALE_LENS length."""
    for S in SCALE_LENS:
        if ("scale", src, S, 0, "full+" + SCALE_DECODE) in done:
            continue
        kv = LazyRepeat(snap["kv"], S)
        r = decode(snap, 0.0, SCALE_DECODE, MAX_NEW, kv=kv, T=S)
        del kv
        r.pop("out_ids")
        record(dict(task="scale", src=src, L=S, i=0, config="full+" + SCALE_DECODE, base_L=L, **r))
        print(f"[{time.time() - t_start:6.0f}s] scale {S} ({SCALE_DECODE}, from {L}) build {r['cache_build_s']:.0f}s "
              f"decode {r['decode_ms_per_token']:.1f} ms/tok peak {r['peak_gpu_gb']:.1f} GB kv {r['kv_cache_gb']:.2f} GB", flush=True)


jobs = [(L, "prose") for L in LENS] + [(L, "code") for L in CODE_LENS]
for L, src in sorted(jobs):
    text = prose_text() if src == "prose" else code_text()
    ids = chat(body(text, L, seed=L) + "\n\nSummarize the text above in a few sentences.")
    key = ("summary", src, L, 0)
    want_scale = src == "prose" and L == max(LENS) and SCALE_LENS
    for plan, decodes in by_plan(CONFIGS).items():
        todo = [d for d in decodes if key + (f"{plan}+{d}",) not in done]
        need_scale = want_scale and plan == "full" and any(("scale", "prose", S, 0, "full+" + SCALE_DECODE) not in done for S in SCALE_LENS)
        if not todo and not need_scale:
            continue
        snap, pre_s = prefill(ids, plan)
        for d in todo:
            cfg = f"{plan}+{d}"
            r = decode(snap, pre_s, d, MAX_NEW)
            if cfg == CONFIGS[0]:
                exact_out[key] = r["out_ids"]
            elif key in exact_out:
                r |= drift(r["out_ids"], exact_out[key])
            record(dict(task="summary", src=src, L=L, i=0, config=cfg, **r))
            print(f"[{time.time() - t_start:6.0f}s] summary {src} L={L} {cfg:15s} TTFT {r['ttft_s']:6.1f}s "
                  f"decode {r['decode_ms_per_token'] or 0:6.1f} ms/tok peak {r['peak_gpu_gb']:.1f} GB kv {r['kv_cache_gb']:.2f} GB "
                  f"{'' if 'rougeL_f1' not in r else 'rougeL %.2f same %.2f' % (r['rougeL_f1'], r['same_token_share'])}", flush=True)
        if need_scale:
            scale_test(snap, src, L)
        del snap
        torch.cuda.empty_cache()

# ------------------------------------------------------------------------------------------------ needles
WORDS = ["apple", "river", "falcon", "copper", "violet", "harbor", "meadow", "cobalt", "lantern", "glacier", "orchid",
         "saffron", "thunder", "willow", "quartz", "ember", "canyon", "marble", "nebula", "pepper"]
prose = prose_text()
for L, n_samples in NEEDLE:
    for i in range(n_samples):
        rng = random.Random(1000 * L + i)
        words = rng.sample(WORDS, 4)
        nums = [str(rng.randint(1000000, 9999999)) for _ in words]
        sents = tok(body(prose, L - 120, seed=L + i), add_special_tokens=False)["input_ids"]
        for w, n in zip(words, nums):
            at = rng.randrange(0, len(sents))
            sents = sents[:at] + tok(f" The special magic number for {w} is {n}. ", add_special_tokens=False)["input_ids"] + sents[at:]
        k = rng.randrange(len(words))
        ids = chat(tok.decode(sents) + f"\n\nWhat is the special magic number for {words[k]} mentioned in the text above? "
                   "Answer with the number only.")
        for plan, decodes in by_plan(CONFIGS).items():
            todo = [d for d in decodes if ("needle", "prose", L, i, f"{plan}+{d}") not in done]
            if not todo:
                continue
            snap, pre_s = prefill(ids, plan)
            for d in todo:
                cfg = f"{plan}+{d}"
                r = decode(snap, pre_s, d, 24, stop_digits=True)
                ok = nums[k] in re.sub(r"[^0-9]", "", r["text"])
                r.pop("out_ids")
                record(dict(task="needle", src="prose", L=L, i=i, config=cfg, correct=ok, answer=nums[k], **r))
                print(f"[{time.time() - t_start:6.0f}s] needle L={L} #{i} {cfg:15s} {'OK ' if ok else 'MISS'} "
                      f"TTFT {r['ttft_s']:6.1f}s -> {r['text'][:40]!r}", flush=True)
            del snap
            torch.cuda.empty_cache()
print("USAGE_DONE")
