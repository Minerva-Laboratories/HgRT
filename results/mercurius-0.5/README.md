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
| exact | 22.0 | 13.8 | **12.7** | 13.0 | 1.73x | 0.63 |
| FoveaKV (as trained) | 23.4 | | **13.1** | | 1.79x | 0.66 |

## Decode step vs context

One captured decode step (M = 1) and one verification pass for k = 2 (M = 3), in ms, by context length.

| context | exact M=1 | FoveaKV M=1 | exact M=3 | FoveaKV M=3 |
|---:|---:|---:|---:|---:|
| 512 | 21.7 | 23.2 | 26.4 | 28.0 |
| 8K | 24.7 | 30.2 | 31.6 | 42.7 |
| 16K | 27.3 | 37.1 | 38.3 | 59.2 |
| 32K | 32.3 | 50.5 | 51.1 | 90.2 |

FoveaKV saves memory but is slower than the exact cache at long context. Its attention scores every token with the
2-bit codes, then reads the hot 10 % at 4 bits, so it makes two passes over the codes. Both of its attention kernels are
instruction-bound, not memory-bound (see Limitations in the main README).

## Long context, end to end

The prompt is HgRT's source code followed by a question about it (`experiments/mercurius05/bench.py`). Each run
generates 128 tokens after the prompt. Times are ms per generated token. Both caches gave the same answer text, and MTP
was lossless in every run.

| context | exact plain | exact k=2 | FoveaKV plain | FoveaKV k=2 |
|---:|---:|---:|---:|---:|
| 8K | 25.4 | 17.7 | 30.4 | 20.4 |
| 16K | 28.0 | 21.6 | 36.7 | 31.4 |

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

The prompt runs through the model in chunks of 256 tokens. The packed projections run as bf16 GEMMs with fp32
accumulation, from the pipeline's own reconstructed weights bf16((q − 8) d). GDN-2 processes a chunk in one launch.
Causal MLA attention runs as fp32 GEMMs over the cache in tiles of 64 rows. The times below include filling the MTP
cache.

| prompt | prefill | token by token |
|---:|---:|---:|
| 420 | 0.53 s | 9.3 s |
| 1,024 | 1.2 s | |
| 2,048 | 2.6 s | |
| 8K | 11.7 s | |
| 16K | 29 s | |
| 32K | 78 s (peak GPU 4.7 GiB) | |

## Accuracy

Each check teacher-forces the same 420-token prompt plus 220 tokens. "Top-1" counts positions where the engine and the
reference predict the same next token; KL is the mean per-token divergence.

| engine | reference | top-1 | KL mean |
|---|---|---:|---:|
| exact cache, prompt token by token | pipeline forward, exact KV | 220/220 | 2.8e-4 |
| exact cache, batched prefill | pipeline forward, exact KV | 220/220 | 4.2e-4 |
| FoveaKV, prompt token by token | pipeline FoveaKV forward | 218/220 | 2.4e-3 |
| FoveaKV, batched prefill | pipeline FoveaKV forward | 217/220 | 3.2e-3 |

The references are the training pipeline's own modules (Mercurius-Pipeline `src/mercurius/arch`) on the same packed
weights. The pipeline is not public, so these accuracy checks cannot be reproduced from this repository. The test suite
covers the rest:
- every kernel against PyTorch;
- per-row results identical for any number of rows;
- eager equal to graph;
- prefill against token-by-token;
- speculative decoding at k = 1..3 emitting exactly the plain greedy tokens.
