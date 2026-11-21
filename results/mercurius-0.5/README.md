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

| KV cache | plain | MTP k=1 | MTP k=2 | MTP k=3 | k=2 speed-up | k=2 acceptance |
|---|---:|---:|---:|---:|---:|---:|
| exact | 21.5 | 13.2 | **10.5** | 10.0 | 2.05x | 0.73 |
| FoveaKV (as trained) | 22.6 | 13.9 | **11.7** | 10.2 | 1.94x | 0.69 |

Since 0.2.3 the Q4_0 GEMVs run on the tensor cores, which makes verification passes cheaper (k = 2: 23.4 ms vs a 21.4 ms
plain step at 512 tokens). At short context k = 3 is now slightly faster than k = 2.

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
generates 128 tokens after the prompt. Times are ms per generated token. Both caches gave the same answer text, and MTP
was lossless in every run.

| context | exact plain | exact k=2 | FoveaKV plain | FoveaKV k=2 |
|---:|---:|---:|---:|---:|
| 8K | 24.7 | 15.1 | 25.1 | 15.5 |
| 16K | 27.6 | 16.7 | 27.5 | 17.5 |
| 32K | 33.5 | 23.7 | **31.7** | **22.4** |

Source: `bench.jsonl` (k = 1..3, prefill time, peak memory, acceptance).

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
The Q8_0 projections run as bf16 GEMMs over bf16(q d). GDN-2 processes a chunk in one launch.
Causal MLA attention runs as one fused bf16 attention per KV group (PyTorch's memory-efficient kernel; fp32 softmax and
accumulation), with each key's 1/rms folded into the keys. The times include filling the MTP cache and, for FoveaKV,
encoding the prompt into codes.

| prompt | exact | FoveaKV | 0.2.1 (FoveaKV) | peak GPU (FoveaKV) |
|---:|---:|---:|---:|---:|
| 512 | 0.65 s | 0.76 s | | 4.08 GiB |
| 8K | 8.1 s | 8.5 s | 12.1 s | 4.27 GiB |
| 16K | 18.4 s | 19.1 s | 29.8 s | 4.46 GiB |
| 32K | 45.2 s | 46.7 s | 82.5 s | 4.86 GiB |

## Accuracy

Each check teacher-forces the same 420-token prompt plus 220 tokens. "Top-1" counts positions where the engine and the
reference predict the same next token; KL is the mean per-token divergence.

| engine | reference | top-1 | KL mean |
|---|---|---:|---:|
| exact cache, prompt token by token | pipeline forward, exact KV | 220/220 | 2.9e-4 |
| exact cache, batched prefill | pipeline forward, exact KV | 218/220 | 3.0e-4 |
| FoveaKV, prompt token by token | pipeline FoveaKV forward | 218/220 | 2.0e-3 |
| FoveaKV, batched prefill | pipeline FoveaKV forward | 218/220 | 2.5e-3 |

The two mismatches with the exact cache and batched prefill (0.2.3; 220/220 in 0.2.2) are near-ties between two tokens.

The references are the training pipeline's own modules (Mercurius-Pipeline `src/mercurius/arch`) on the same packed
weights. The pipeline is not public, so these accuracy checks cannot be reproduced from this repository. The test suite
covers the rest:
- every kernel against PyTorch;
- per-row results identical for any number of rows;
- eager equal to graph;
- prefill against token-by-token;
- speculative decoding at k = 1..3 emitting exactly the plain greedy tokens.
