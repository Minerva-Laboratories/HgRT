# Results

Every file was measured on a Jetson AGX Orin 64 GB in MODE_50W, unless the file says otherwise. The summary tables are
in the [top-level README](../README.md#results).

**Naming.** HgRT used to be called Cairn, SparsePage used to be Riffle, and DepthSkip used to be Terrace. Raw logs and
JSON outputs written before the rename still use the old names and paths, for example `/mnt/ext/cairn/...`,
`cairn.*` and `CAIRN_*`. They are kept unchanged as historical records. The reports (`*.md`) use the current names.

## Mercurius-1-4B engine: `mercurius/`

| file | content | written by |
|---|---|---|
| [`mercurius/README.md`](mercurius/README.md) | the full report: decode / MTP / prefill tables, design, verification, limitations, how to run | hand-written from the files below |
| `mercurius/bench_v2.json` | decode tok/s, plain and MTP k = 1/2/3/adaptive, τ, losslessness (follow-up round: fused commit, int8 MTP cache) | `experiments/mercurius/bench.py` |
| `mercurius/bench.json` | first-round decode benchmark (separate commit kernels, bf16 MTP cache), agreement with the reference | `experiments/mercurius/bench.py` |
| `mercurius/summary.json` | first-round tables collected in one file | `experiments/mercurius/report.py` |
| `mercurius/prefill.json` | native prefill: speed, agreement with the transformers prefill, per-part profile | `experiments/mercurius/prefill_bench.py` |
| `mercurius/diag.json`, `mercurius/diag_cycle.json` | teacher-forced disagreements vs reference margins, per-component and per-cycle timings | `experiments/mercurius/diag.py` |
| `mercurius/gemv_tune.json` | NF4 GEMV tiling sweep | `experiments/mercurius/gemv_tune.py` |
| `mercurius/gemv_rows.json` | the chosen tiling, **read by the engine at load time** | `experiments/mercurius/gemv_tune.py` |
| `mercurius/raw/` | console logs, earlier runs (`raw/history/`), the llama.cpp baseline, draft-vocabulary and MTP-cache sweeps, attention phase profiles | various |

## Long-context stack (Qwen3.5-9B)

| file | content | written by |
|---|---|---|
| `stack_bench_h4.json` | decode attention per call (eager and in a CUDA graph), bits read, kernel vs reference (synthetic caches, 4 KV heads) | `experiments/stack_bench.py` |
| `qwen3.5-9b/stack_ppl.jsonl` | per-document, per-token NLL for each decode cache (prose / code, 16K / 32K, 3 documents each) | `experiments/stack_ppl.py` (in FoveaKV's harness, before the split) |
| `qwen3.5-9b/usage.jsonl` | ordinary use: summaries with TTFT, decode ms/token, peak memory, KV bytes, drift from the exact answer (partial run) | `experiments/usage.py` |
| `profile/long_response.json` | long-response decode profile: graph replay vs decode-time block encode per generated token | `experiments/profile/long_response.py` |

`experiments/e2e_ppl.py` writes `qwen3.5-9b/e2e_ppl.jsonl`. That file is not here yet.
