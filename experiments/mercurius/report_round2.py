"""Markdown for the follow-up round (before/after tables) from the result JSONs; printed to stdout."""
import json
import os
import re

RES = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'results', 'mercurius')
L = []
before = json.load(open(os.path.join(RES, "bench.json")))["ctx"]
after = json.load(open(os.path.join(RES, "bench_v2.json")))["ctx"]
ref = json.load(open(os.path.join(RES, "raw/transformers_reference_timing.json")))


def best(r):
    k = max((1, 2, 3), key=lambda k: r[f"mtp_k{k}"]["tok_s"])
    return k, r[f"mtp_k{k}"]["tok_s"]


L.append("### Decode before / after (tok/s; 256 tokens; every speculative run lossless)\n")
L.append("| context | plain | MTP k=1 before → after | k=2 before → after | k=3 before → after | adaptive k (after) | best before → after |")
L.append("|---|---|---|---|---|---|---|")
for c in before:
    b, a = before[c], after[c]
    kb, sb = best(b)
    ka, sa = best(a)
    ad = a.get("mtp_adaptive", {})
    L.append(f"| {c} (T={a['T']}) | {b['plain']['tok_s']:.1f} → {a['plain']['tok_s']:.1f} | " +
             " | ".join(f"{b[f'mtp_k{k}']['tok_s']:.1f} → {a[f'mtp_k{k}']['tok_s']:.1f}" for k in (1, 2, 3)) +
             f" | {ad.get('tok_s', float('nan')):.1f} (k groups {ad.get('k_choices', {})}) | {sb:.1f} (k={kb}) → {sa:.1f} (k={ka}) |")
L.append("")
L.append("Before: separate GDN-2 commit kernels, bf16 MTP cache. After: commit fused into the next verification, int8 MTP "
         "cache. Plain decode is unchanged (same kernels).\n")

# attention kernels
txt = open(os.path.join(RES, "raw/attn_v2_vs_flashdecoding.out")).read()
L.append("### MLA attention from the codes: v2 (shared-memory K tile, used) vs flash-decoding v3 (register-decoded K, kept as option)\n")
L.append("| layer / context | NQ=1 v2 / v3 | NQ=2 | NQ=3 | NQ=4 | code bytes |")
L.append("|---|---|---|---|---|---|")
for line in txt.splitlines():
    m = re.match(r"layer (\d+) r=(\d+) L=(\d+):(.*)\| code bytes ([\d.]+) MB", line)
    if not m:
        continue
    vals = dict(re.findall(r"NQ(\d) (v2|fd) ([\d.]+)", m.group(4)) and {} or {})
    cells = []
    for nq in "1234":
        v2 = re.search(rf"NQ{nq} v2 ([\d.]+)", m.group(4)).group(1)
        fd = re.search(rf"NQ{nq} fd ([\d.]+)", m.group(4)).group(1)
        cells.append(f"{v2} / {fd}")
    L.append(f"| layer {m.group(1)} (r={m.group(2)}), {m.group(3)} cells | " + " | ".join(cells) + f" | {m.group(5)} MB |")
L.append("")
L.append("ms per call. The memory floor at 32K is ~0.06-0.08 ms (12.6 MB at ~150 GB/s); both kernels are 10x+ above it.\n")

# MTP cache forms
L.append("### MTP cache form (fused commit; τ / tok/s)\n")
L.append("| context | k | bf16 | int8 (per row) | int4 (per 32) |")
L.append("|---|---|---|---|---|")
kv = {f: json.load(open(os.path.join(RES, f"raw/mtpkv_{f}.json")))["ctx"] for f in ("bf16", "int8", "int4")}
for c in kv["bf16"]:
    for k in (2, 3):
        L.append(f"| {c} | {k} | " + " | ".join(f"{kv[f][c][f'mtp_k{k}']['tau']:.2f} / {kv[f][c][f'mtp_k{k}']['tok_s']:.1f}"
                                                for f in ("bf16", "int8", "int4")) + " |")
L.append("")
L.append("Cache size per token (4 kv heads x 256, K + V): bf16 4096 B, int8 2064 B, int4 1088 B (34k positions: 139 / 70 / 37 MB).\n")

# prefill
pf = json.load(open(os.path.join(RES, "prefill.json")))
lp = open(os.path.join(RES, "raw/llama_prefill.log")).read()
lrows = json.loads(lp[lp.find("["):lp.rfind("]") + 1])
llama = {r["n_prompt"]: r["avg_ts"] for r in lrows}
L.append("### Native prefill (no transformers model, no snapshot)\n")
L.append("| context | HgRT native s (tok/s) | transformers deployed s (tok/s) | llama-mercurius NF4 tok/s | peak GB | first token = reference | ref. top-2 margin | last-logits rel. diff | MTP k=2 lossless from native state |")
L.append("|---|---|---|---|---|---|---|---|---|")
for c, r in pf.items():
    n = {"long1024": 1024, "long8192": 8192, "long32768": 32768}.get(c)
    L.append(f"| {c} (T={r['T']}) | {r['prefill_s']:.2f} ({r['prefill_tok_s']:.0f}) | {r['transformers_prefill_s']:.2f} "
             f"({r['T'] / r['transformers_prefill_s']:.0f}) | {llama[n]:.0f} | {r['peak_gb']:.2f} | {r['first_token_equal']} | "
             f"{r['reference_first_token_margin']:.3f} | {r['last_logits_rel_vs_reference']:.3f} | {r['mtp_k2_lossless_from_native']} |"
             if n else
             f"| {c} (T={r['T']}) | {r['prefill_s']:.2f} ({r['prefill_tok_s']:.0f}) | {r['transformers_prefill_s']:.2f} | – | "
             f"{r['peak_gb']:.2f} | {r['first_token_equal']} | {r['reference_first_token_margin']:.3f} | "
             f"{r['last_logits_rel_vs_reference']:.3f} | {r['mtp_k2_lossless_from_native']} |")
L.append("")
k32 = pf.get("long32768", {})
if k32:
    L.append("Profile at 32K (s): " + ", ".join(f"{k} {v}" for k, v in k32.get("profile_s", {}).items()) + ".\n")
print("\n".join(L))
