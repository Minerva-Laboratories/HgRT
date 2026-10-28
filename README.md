# HgRT

The Mercurius runtime and long-context stack (Python package `hgrt`, its working name; the component packages keep
theirs too: DepthSkip = `depthskip`, SparsePage = `sparsepage`). The long-context stack as one system. A prompt goes through **DepthSkip** (graded per-token depth: old tokens stop at
layer 16, the last 512 at 24, the last 32 run every layer; upper-layer caches come from closed-form maps), its cache is
stored as **FoveaKV** 2+2-bit gain-shape pages with a bf16 recent window, and every decoded token reads only the pages
**SparsePage** selects (per-page summaries scored on the GPU, top 10-20% opened, the rest dropped). **OpenDecider** runs on
the same stack for decisions (`/v1/decide`), where only the prompt side applies.

| Component | What it does | Here |
|---|---|---|
| [DepthSkip](https://github.com/Minerva-Laboratories/DepthSkip) (`depthskip`) | prompt pass: layer skipping with ridge maps, graded depth, part skips | `third_party/depthskip` |
| [FoveaKV](https://github.com/Minerva-Laboratories/FoveaKV) | KV codecs: rotated gain-shape 2+2-bit pages, decode kernels with a page-list mode | `third_party/foveakv` |
| [SparsePage](https://github.com/Minerva-Laboratories/SparsePage) (`sparsepage`) | decode: page summaries, score / select / combine kernels | `third_party/sparsepage` |
| [OpenDecider](https://github.com/Minerva-Laboratories/OpenDecider) | decision model on a frozen LM (no text generation) | `third_party/opendecider` |

HgRT itself only holds the glue: `src/hgrt/stack.py` (FoveaKV pages + SparsePage selection + window, one CUDA-graph-
capturable step), `pipeline.py` (DepthSkip prefill -> cache handoff -> decode), `decide.py` (OpenDecider with the
DepthSkip state pass), `caches.py` (cache factory), and the end-to-end experiments.

## Setup

```bash
git clone --recursive git@github.com:Minerva-Laboratories/HgRT.git && cd HgRT
scripts/setup.sh                   # submodules + editable installs + CPU tests
python -m pytest -q                # with CUDA: kernels against their references
```

Kernels are JIT-compiled with `torch.utils.cpp_extension` for the local GPU on first use: `ninja` must be on `PATH`
and nvcc must match torch's CUDA major. Decode-time block encodes replay a captured CUDA graph
(`HGRT_ENCODE_GRAPH=0` for the eager path).

## Mercurius-1-4B engine (decode + MTP self-speculative decoding)

`src/hgrt/mercurius_engine.py` (decode step, one CUDA graph per token), `mercurius_mtp.py` (stock Qwen3.5-4B MTP
head, lossless speculative decoding), `mercurius_prefill.py` (native prefill). Results: `results/mercurius/README.md`.

```bash
export HGRT_MERCURIUS_DIR=/path/to/work          # weights.pt, snapshots, references (default /mnt/ext/hgrt-kernels/mercurius)
export OPENDECIDER_ROOT=third_party/opendecider  # prep.py loads the deployed model through OpenDecider's deploy code
python experiments/mercurius/prep.py             # once: dump NF4 weights + references + prefill snapshots (~8 min)
python tests/test_mercurius.py                   # kernels, engine vs reference, graph == eager, lossless speculation
python experiments/mercurius/bench.py --ctx chat0,long1024,long8192,long32768 --n 256 --ks 1,2,3
```
Other overrides: `MERCURIUS_ROOT` (the deployed checkpoint, prep.py), `HGRT_WIKITEXT`, `MERC_PREFILL_EXTRA`.

## Use

```bash
python -m hgrt generate --prompt "$(cat long_document.txt) Summarize:" --plan g512 --decode stack10
python -m hgrt fit-maps --model Qwen/Qwen3.5-9B --cuts 16 24          # DepthSkip maps, cached in HGRT_MAPS_DIR
python -m hgrt decide-fit --ckpt runs/x9b/model.pt --out maps_cut24.pt --cut 24 --calib states.jsonl
python -m hgrt decide --ckpt runs/x9b/model.pt --maps maps_cut24.pt  # OpenDecider /v1/decide, DepthSkip state pass
```

```python
from hgrt.pipeline import Pipeline
p = Pipeline("Qwen/Qwen3.5-9B")
snap = p.prefill(ids, "g512")                 # DepthSkip prompt pass, handed off to the decode loop
nll, stats = p.perplexity(snap, targets, "stack10")
```

Prefill plans: `exact`, `cut24`, `cut16`, `cut16_tail32`, `g512`, `f64_g512` (first 64 tokens at full depth too).
Decode caches: `stack5/10/20[-bound|-mean]`, `sparsepage10/20` (bf16 pages), and any FoveaKV scheme (`bf16`, `gs4`,
`dyn10`, ...).

## Results (Qwen3.5-9B, Jetson AGX Orin 64 GB, MODE_50W)

**Decode attention, per call** (`experiments/stack_bench.py`, synthetic caches, 4 KV heads, inside a CUDA graph of 8
calls = the 9B's attention layers; kernel vs PyTorch reference: output 1e-6, identical page selection):

| context | bf16 kernel | FoveaKV 2-bit | FoveaKV dyn10 | **stack 10%** | stack bits / token / step |
|---|---|---|---|---|---|
| 32K | 5.50 ms | 1.14 ms | 1.86 ms | **0.36 ms** | 0.74 |
| 128K | 21.5 ms | 3.79 ms | 5.96 ms | **0.79 ms** | 0.68 |
| 256K | 42.3 ms | 7.43 ms | 11.5 ms | **1.45 ms** (2.57 in a rerun) | 0.68 |

Per decoded token (8 attention layers): 2.9 / 6.4 / 11.6 ms of attention at 32K / 128K / 256K, against 44 / 172 /
339 ms with a bf16 cache; the rest of a decode step is reading the weights.

**Decode quality** (`experiments/stack_ppl.py`; exact prefill, 256 tokens teacher-forced through the cache, prose and
code, 16K and 32K, 3 documents each; ΔNLL with a 95% token bootstrap; measured with the same kernels in FoveaKV's
harness before the split, `results/qwen3.5-9b/stack_ppl.jsonl`):

| decode cache | bits read / token / step | ΔPPL | ΔNLL [95% CI] | worst document |
|---|---|---|---|---|
| FoveaKV gs4 | 4.29 | +0.27% | +0.0027 [-0.0006, +0.0058] | +0.73% |
| FoveaKV dyn10 | 2.52 | +0.29% | +0.0028 [-0.0002, +0.0059] | +0.90% |
| **stack 20%** | 1.27 | +0.21% | +0.0021 [-0.0029, +0.0069] | +0.99% |
| **stack 10%** | 0.86 | +0.63% | +0.0062 [-0.0002, +0.0126] | +3.44% |
| stack 10%, bound only | 0.73 | +0.85% | +0.0083 [+0.0013, +0.0154] | +3.23% |
| stack 5% | 0.65 | +1.86% | +0.0181 [+0.0069, +0.0290] | +6.10% |

**Prefill** (DepthSkip README; 9B, serving kernels): cut 16 is 1.65-1.86x faster than exact from 1K to 131K tokens, cut 24
1.25-1.31x; graded g512 keeps lettered multiple choice at the exact model's accuracy with about a quarter of cut 16's
generation KL (0.024), at 1.43x faster than exact at 15K tokens.

**End to end** (`experiments/e2e_ppl.py`: DepthSkip prefill x decode cache): running; results will appear in
`results/qwen3.5-9b/e2e_ppl.jsonl`.

## Status and limits

Batch 1; GQA group 4 and head_dim 256 (the Qwen3.5 family); DepthSkip's maps are per model and fitted in minutes from
unlabeled text. The decode harness runs Qwen3.5's layers directly (not `generate()`), one token at a time; the bench
numbers are attention only. Nothing here is trained.

License: Apache-2.0 (each component under its own license, see `third_party/`).
