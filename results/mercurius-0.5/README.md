# Mercurius-0.5 engine: results

Engine: `hgrt.mercurius_packed.PackedMercuriusEngine` with
[Mercurius-0.5-4B-Preview](https://huggingface.co/Minerva-Laboratories/Mercurius-0.5-4B-Preview) (`mercurius-packed-v1`:
24 GDN-2 + 8 CARE-latent MLA layers, Q4_0 / Q8_0 weights, MTP head, FoveaKV-trained KV cache).

**Hardware:** one RTX 4050 Laptop GPU (6 GB). The Mercurius-1-4B and long-context results elsewhere in this repository
were measured on a Jetson AGX Orin, so the two sets of absolute numbers are not comparable.

Run with `experiments/mercurius05/bench.py` (speed, memory) and the test suite (`HGRT_MERCURIUS05=<checkpoint>`).

## Decode, short context

The prompt is 420 tokens, followed by 160 greedy tokens. Times are ms per generated token with CUDA graphs. MTP uses the
checkpoint's own head (draft vocabulary 32K). **The speculative runs emitted exactly the plain greedy tokens** with both
caches.

| KV cache | plain | MTP k=1 | MTP k=2 | MTP k=3 | MTP auto | auto speed-up |
|---|---:|---:|---:|---:|---:|---:|
| exact | 22.2 | 13.4 | 10.6 | 10.0 | **8.8** | 2.53x |
| FoveaKV (as trained) | 23.2 | 14.4 | 12.0 | 10.6 | **9.7** | 2.38x |

Since 0.3.0 the decode GEMVs serve up to 8 rows on the tensor cores (Q4_0 and Q8_0), so a verification pass of 8 rows
reads the weights once, like a single token, and `k="auto"` can verify up to 7 drafts.

## Adaptive draft depth (`k="auto"`)

After each draft the policy (`spec_policy.py`) reads its top-1 probability over the draft vocabulary, estimates from
online statistics (per draft position and probability bin) how likely it is to be accepted, and keeps drafting while the
expected extra tokens are worth the extra draft step and the larger verification pass. Verification costs per row count
and context length are measured once, from the captured graphs, in `capture()`. Capturing k = 0..7 costs about 0.4 s
and 0.24 GiB of peak memory more than k = 0..2.

12 prompts, 256 greedy tokens each, ms per generated token, median of 3 runs on one engine (the statistics carry over
between calls, as in normal use), `experiments/mercurius05/dynamic_depth/bench_auto.py`. Every run emitted exactly the
plain greedy tokens. Bold: faster than both fixed k.

| prompt | tokens | exact k=0 | k=2 | k=3 | **auto** | FoveaKV k=0 | k=2 | k=3 | **auto** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| json-planets | 35 | 22.1 | 9.19 | 7.35 | **4.72** | 23.0 | 9.55 | 7.87 | **5.35** |
| json-employees | 32 | 22.1 | 9.92 | 8.53 | **6.79** | 23.0 | 10.42 | 8.99 | **7.16** |
| json-command | 47 | 22.1 | 11.54 | 10.13 | **9.31** | 23.0 | 12.14 | 10.70 | **9.85** |
| chat-science | 28 | 22.1 | 11.95 | 12.29 | **11.51** | 23.0 | 12.33 | 12.61 | **11.75** |
| chat-es | 30 | 22.1 | 14.08 | 13.88 | **13.48** | 23.0 | 14.77 | 14.64 | **14.19** |
| chat-advice | 33 | 22.1 | 12.45 | 11.31 | 11.55 | 23.0 | 13.50 | 12.39 | **12.34** |
| code-lru | 34 | 22.1 | 11.24 | 9.59 | **9.43** | 23.0 | 11.92 | 11.17 | **11.08** |
| code-iso | 39 | 22.1 | 10.12 | 9.38 | **8.67** | 23.0 | 11.16 | 10.47 | **9.44** |
| math | 65 | 22.1 | 10.33 | 9.40 | **8.73** | 23.1 | 10.66 | 9.71 | **8.85** |
| summary-2k | 2019 | 22.8 | 12.23 | 11.17 | **9.44** | 24.0 | 13.57 | 12.26 | **10.86** |
| code-8k | 8192 | 25.2 | 15.30 | 14.73 | **14.30** | 25.8 | 15.91 | 15.82 | **14.69** |
| code-32k | 32768 | 34.0 | 23.03 | 22.19 | **20.66** | 32.4 | 21.41 | 20.06 | **19.81** |

Geometric mean, `auto` against k = 2: 1.22x (exact), 1.20x (FoveaKV); against the better of k = 2 and 3 for each
prompt: 1.11x and 1.10x. Fixed depths above 3 are slower on most prompts (only predictable output such as JSON gains);
the adaptive policy goes to 7 drafts there and stays at 1-3 elsewhere. A first call on a new engine runs on prior
statistics: in `bench.py`'s single cold runs below it ranged from 11 % slower to 5 % faster than the best fixed k.
Source: `adaptive.jsonl`.

## Decode step vs context

One captured decode step (M = 1) and one verification pass for k = 2 (M = 3), in ms, by context length.

| context | exact M=1 | FoveaKV M=1 | exact M=3 | FoveaKV M=3 |
|---:|---:|---:|---:|---:|
| 512 | 21.4 | 22.5 | 23.4 | 25.4 |
| 8K | 24.5 | 25.3 | 30.0 | 31.9 |
| 16K | 27.1 | 27.7 | 36.8 | 38.7 |
| 32K | 32.7 | 32.3 | 50.4 | 51.2 |

FoveaKV's attention scores every token with the 2-bit codes, then reads the hot 10 % at 4 bits: two passes over the
codes. Both run on the tensor cores (scores and value sums as bf16 hi + lo MMAs with fp32 accumulation), and the second
pass reuses the first pass's scores for chunks without hot tokens. In 0.2.1 the same steps took 30.2 / 37.1 / 50.5 ms at
8K / 16K / 32K.

## Long context, end to end

The prompt is HgRT's source code followed by a question about it (`experiments/mercurius05/bench.py`). Each run
generates 128 tokens after the prompt. Times are ms per generated token, one run each (`auto`: the engine's first call at
that context). Both caches gave the same answer text, and MTP was lossless in every run.

| context | exact plain | exact k=2 | exact auto | FoveaKV plain | FoveaKV k=2 | FoveaKV auto |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 25.2 | 15.6 | 14.8 | 25.8 | 15.7 | 15.5 |
| 16K | 28.4 | 18.2 | 18.7 | 28.2 | 18.1 | 16.5 |
| 32K | 34.0 | 24.4 | 23.3 | **32.5** | 23.1 | **22.1** |

Source: `bench.jsonl` (k = 1..3 and auto, prefill time, peak memory, acceptance).

## Memory

This is the MLA KV cache over the 8 MLA layers. GDN-2 state is constant: 24 layers × 16 heads × 128 × 128 fp32.

| context | exact | FoveaKV | ratio |
|---:|---:|---:|---:|
| 8K | 75 MiB | 22 MiB | 3.4x |
| 16K | 148 MiB | 42 MiB | 3.5x |
| 32K | 292 MiB | 80 MiB | 3.7x |

Per token and layer, FoveaKV stores:
- 4-bit codes of the rotated latent and of k_rope;
- an 8-bit gain per 128-channel head;
- fp16 1/rms per KV group;
- fp16 block means per 128 tokens.

Exact copies are kept only for the 4 sinks and a ring of 2 × 128 + 4 tokens. The decode-time attention scratch is fixed
at about 30 MB, whatever the context length.

## Prefill

The prompt runs through the model in chunks of 1,024 tokens. The Q4_0 projections run on the bf16 tensor cores straight
from the codes: exact (q − 8) x products, fp32 accumulation and the scale applied per group in fp32, the decode numerics.
The Q8_0 projections run as bf16 GEMMs over bf16(q d). Since 0.3.0 GDN-2 runs as a chunked scan: WY form over 32-token
chunks (fp32), the chunk-to-chunk state on the tensor cores with split bf16 hi + lo operands (fp32-level, ~1e-5 against
the step kernel), in 128-token slices so its scratch stays in L2; per layer, 280 rows take 0.77 ms instead of 1.70 ms
token by token, and 1,024 rows 2.3 ms instead of 6.2 ms. The Mercurius-1 engine's prefill used fla's `chunk_gdn2`;
`HGRT_GDN_PREFILL=step` keeps the step kernel. Causal MLA attention runs as one fused bf16 attention per KV group
(PyTorch's memory-efficient kernel; fp32 softmax and accumulation), with each key's 1/rms folded into the keys. The
times include filling the MTP cache and, for FoveaKV, encoding the prompt into codes.

| prompt | exact | FoveaKV | 0.2.4 (FoveaKV) | peak GPU (FoveaKV) |
|---:|---:|---:|---:|---:|
| 512 | 0.60 s | 0.71 s | 0.75 s | 4.30 GiB |
| 8K | 7.6 s | 8.0 s | 8.4 s | 4.50 GiB |
| 16K | 17.3 s | 18.1 s | 18.8 s | 4.69 GiB |
| 32K | 43.2 s | 44.7 s | 46.5 s | 5.09 GiB |

The peak includes the CUDA graphs for k = 0..7 (0.2.4 captured k = 0..3 here). A short prompt reaches its first token
sooner: `generate(prompt, 1)` with a 280-token prompt takes 0.28 s with FoveaKV (0.30 s in 0.2.4, 0.36 s in 0.2.2). The
bf16 tensor cores bound it: at 280 tokens the projections are 2.6 TFLOP, and this GPU runs bf16 products with fp32
accumulation at about 18 TFLOPS.

## Accuracy

Each check teacher-forces the same 420-token prompt plus 220 tokens. "Top-1" counts positions where the engine and the
reference predict the same next token; KL is the mean per-token divergence.

| engine | reference | top-1 | KL mean |
|---|---|---:|---:|
| exact cache, prompt token by token | pipeline forward, exact KV | 220/220 | 2.8e-4 |
| exact cache, batched prefill | pipeline forward, exact KV | 219/220 | 3.0e-4 |
| FoveaKV, prompt token by token | pipeline FoveaKV forward | 216/220 | 2.6e-3 |
| FoveaKV, batched prefill | pipeline FoveaKV forward | 217/220 | 2.2e-3 |

A wider check, 5 prompts × 160 generated tokens (Spanish, code, JSON, math, a 1.4K-token document), 0.3.0 against 0.2.4:

| engine | reference | 0.3.0 top-1 | 0.3.0 KL | 0.2.4 top-1 | 0.2.4 KL |
|---|---|---:|---:|---:|---:|
| exact cache, token by token | pipeline, exact KV | 793/800 | 3.2e-4 | 795/800 | 2.9e-4 |
| exact cache, batched prefill | pipeline, exact KV | 793/800 | 3.2e-4 | 793/800 | 3.3e-4 |
| FoveaKV, token by token | pipeline FoveaKV | 791/800 | 1.1e-3 | 790/800 | 9.6e-4 |
| FoveaKV, batched prefill | pipeline FoveaKV | 790/800 | 1.1e-3 | 790/800 | 1.0e-3 |

The decode differences come from the Q8_0 tensor-core GEMV alone (with the FMA kernel 0.3.0 reproduces 0.2.4 exactly).
Both kernels are within 1-2e-7 of fp64 on the model's gate weights (`dynamic_depth/q8err.py`); the end-to-end numbers
move with where the last-bit roundings land, as they did between earlier kernel orderings.

The references are the training pipeline's own modules (Mercurius-Pipeline `src/mercurius/arch`) on the same packed
weights. The pipeline is not public, so these accuracy checks cannot be reproduced from this repository. The test suite
covers the rest:
- every kernel against PyTorch, and the chunked GDN-2 scan against the step kernel;
- per-row results identical for any number of rows;
- eager equal to graph;
- prefill against token-by-token;
- speculative decoding at k = 1..7 and `k="auto"` emitting exactly the plain greedy tokens.
