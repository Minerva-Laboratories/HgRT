# Mercurius-1-4B on HgRT: decode engine + MTP self-speculative decoding

Jetson AGX Orin 64 GB, MODE_50W (GPU ≤ 816 MHz, EMC DVFS-governed), CUDA 13.2, torch 2.14. All numbers below were
measured on this board in this session; every GPU job ran under `gpu_lock.sh` with memwatch.

**Summary.** HgRT now decodes Mercurius-1-4B from its exact deployed NF4 weights and its TurboQuant 4-bit KV cache at
38.6 / 37.1 / 32.7 tok/s at ~1K / 8K / 32K context (1.8-1.9x llama-mercurius on the same NF4 weights, ~15-26x the
deployed transformers path). With the stock Qwen3.5-4B MTP head, lossless self-speculative decoding reaches 70.9 / 60.7 /
49.7 tok/s (best k per context; 1.84x / 1.64x / 1.52x over plain), up to 88.5 tok/s on chat prompts. All 27 speculative
runs produced exactly the plain greedy token stream; the verification pass is bit-identical to plain decoding by
construction and by test. Long-context attention (and with it MTP at 32K) is the main remaining inefficiency.

**Follow-up round.** Native prefill now replaces the transformers hand-off (578 / 609 / 503 tok/s at ~1K / 8K / 32K:
above llama-mercurius at every length, ~1.6-1.8x slower than the transformers path); the fused GDN-2 commit, the int8
MTP cache and adaptive k are in and lossless (36/36 speculative runs), with small gains (best MTP 72.1 / 62.3 / 50.5
tok/s at ~1K / 8K / 32K, +1-2% over the first round). The flash-decoding attention rewrite is correct but slower than
the existing kernel and is not used — the long-context attention floor was not reached.

## Follow-up round (same day): flash-decoding attempt, compressed MTP cache, fused commit, adaptive k, native prefill

What changed, in one place (all new code in the same new files plus `src/hgrt/mercurius_prefill.py`):

1. **Flash-decoding rewrite of the code-reading MLA attention (`mla_attn_fd`, v3)** — K fragments decoded from the 4-bit
   codes straight into `mma.sync` registers (no fp16 K tile in shared memory), Q staged once and read with `ldmatrix`,
   every decoded value reused for all query rows, fixed virtual splits (plain == verify bit for bit; tested). It is
   correct but **not faster**: 6-80% slower than v2 except NQ = 3 at 32K (12% faster there). Phase profiles (`raw/attn_fd_phase_profile_*`)
   show every phase at low IPC; batching the fragment decodes made it slower again. The engine keeps v2 (default
   `attn="v2"`); the memory floor (~0.07 ms per layer at 32K) is **not** reached — both kernels are ~10x above it.
2. **Compressed MTP cache** (`MTPHead(kv="int8"|"int4")`): per-row int8 or int4/32 K and V, written by the MTP attention
   prep kernel and read by a split-K decode kernel. Acceptance is unchanged within noise; speed is unchanged too (the
   MTP decode kernel is bound by its per-token warp reductions, not by cache bytes); memory 2x / 4x smaller.
3. **GDN-2 commit fused into the next verification** (two parity sets of per-step inputs; the next cycle's conv and
   recurrence kernels replay the previous cycle's accepted steps, write the committed state once, then verify) and
   **adaptive k** (k per 8-cycle group from a geometric acceptance estimate over measured cycle times). The tests check
   that after `finalize()` the GDN states and the MLA cache equal plain decoding bit for bit (k = 1, 3, adaptive).
4. **Native tensor-core prefill** (`Prefill`): NF4 weights dequantized with the exact levels, cuBLAS bf16 GEMMs per
   token chunk, fla `chunk_gdn2` with the state carried across chunks, MLA codes written by the decode encode kernel and
   attended in the expanded 448/256 form (key up-projection and query maps from the same converter as the GGUF, decoded
   back from the codes) with a Triton flash kernel — no attention matrix. The engine no longer needs the transformers
   model or a snapshot.

### Decode before / after (tok/s; 256 tokens; every speculative run lossless)

| context | plain | MTP k=1 before → after | k=2 before → after | k=3 before → after | adaptive k (after) | best before → after |
|---|---|---|---|---|---|---|
| chat0 (T=27) | 38.9 → 38.9 | 58.0 → 59.4 | 65.7 → 66.3 | 62.5 → 64.4 | 64.4 (k groups {'1': 1, '2': 6, '3': 7}) | 65.7 (k=2) → 66.3 (k=2) |
| chat1 (T=31) | 38.7 → 38.9 | 62.4 → 63.4 | 73.0 → 74.0 | 77.2 → 78.3 | 75.9 (k groups {'1': 1, '2': 1, '3': 9}) | 77.2 (k=3) → 78.3 (k=3) |
| chat2 (T=26) | 38.8 → 38.9 | 57.4 → 58.3 | 60.0 → 61.5 | 56.5 → 58.1 | 59.5 (k groups {'1': 1, '2': 12, '3': 3}) | 60.0 (k=2) → 61.5 (k=2) |
| chat3 (T=27) | 38.7 → 38.9 | 58.3 → 59.8 | 63.1 → 64.8 | 64.4 → 66.4 | 63.8 (k groups {'1': 1, '2': 5, '3': 8}) | 64.4 (k=3) → 66.4 (k=3) |
| chat4 (T=29) | 38.8 → 38.9 | 53.1 → 54.0 | 53.5 → 53.6 | 51.6 → 51.1 | 53.1 (k groups {'1': 4, '2': 13, '3': 1}) | 53.5 (k=2) → 54.0 (k=1) |
| chat5 (T=42) | 38.7 → 38.9 | 64.6 → 65.5 | 78.5 → 79.5 | 88.5 → 88.6 | 86.4 (k groups {'1': 1, '2': 1, '3': 8}) | 88.5 (k=3) → 88.6 (k=3) |
| long1024 (T=1006) | 38.6 → 38.6 | 59.6 → 60.5 | 69.5 → 70.5 | 70.9 → 72.1 | 69.8 (k groups {'1': 1, '2': 3, '3': 9}) | 70.9 (k=3) → 72.1 (k=3) |
| long8192 (T=8174) | 37.1 → 37.1 | 55.6 → 56.2 | 60.7 → 62.3 | 60.4 → 61.1 | 58.7 (k groups {'1': 1, '2': 8, '3': 5}) | 60.7 (k=2) → 62.3 (k=2) |
| long32768 (T=32750) | 32.7 → 32.7 | 45.7 → 46.3 | 49.7 → 50.5 | 49.0 → 48.9 | 49.4 (k groups {'1': 1, '2': 12, '3': 1}) | 49.7 (k=2) → 50.5 (k=2) |

Before: separate GDN-2 commit kernels, bf16 MTP cache. After: commit fused into the next verification, int8 MTP cache. Plain decode is unchanged (same kernels).

### MLA attention from the codes: v2 (shared-memory K tile, used) vs flash-decoding v3 (register-decoded K, kept as option)

| layer / context | NQ=1 v2 / v3 | NQ=2 | NQ=3 | NQ=4 | code bytes |
|---|---|---|---|---|---|
| layer 11 (r=390), 1024 cells | 0.048 / 0.069 | 0.062 / 0.100 | 0.078 / 0.130 | 0.093 / 0.158 | 0.3 MB |
| layer 11 (r=390), 8192 cells | 0.150 / 0.261 | 0.257 / 0.328 | 0.339 / 0.367 | 0.396 / 0.424 | 2.1 MB |
| layer 11 (r=390), 32768 cells | 0.553 / 0.904 | 0.876 / 1.035 | 1.081 / 1.211 | 1.282 / 1.499 | 8.4 MB |
| layer 31 (r=656), 1024 cells | 0.063 / 0.093 | 0.085 / 0.156 | 0.152 / 0.204 | 0.202 / 0.260 | 0.4 MB |
| layer 31 (r=656), 8192 cells | 0.205 / 0.334 | 0.337 / 0.435 | 0.492 / 0.521 | 0.656 / 0.785 | 3.1 MB |
| layer 31 (r=656), 32768 cells | 0.736 / 1.147 | 1.104 / 1.338 | 1.822 / 1.596 | 2.158 / 2.503 | 12.6 MB |

ms per call. The memory floor at 32K is ~0.06-0.08 ms (12.6 MB at ~150 GB/s); both kernels are 10x+ above it.

### MTP cache form (fused commit; τ / tok/s)

| context | k | bf16 | int8 (per row) | int4 (per 32) |
|---|---|---|---|---|
| chat0 | 2 | 2.29 / 66.7 | 2.28 / 66.2 | 2.25 / 65.5 |
| chat0 | 3 | 2.50 / 63.3 | 2.54 / 64.3 | 2.54 / 64.4 |
| chat1 | 2 | 2.55 / 74.1 | 2.55 / 74.1 | 2.55 / 74.2 |
| chat1 | 3 | 3.09 / 78.3 | 3.09 / 78.0 | 3.01 / 76.1 |
| long8192 | 2 | 2.31 / 61.6 | 2.34 / 62.4 | 2.34 / 62.2 |
| long8192 | 3 | 2.68 / 61.1 | 2.68 / 61.1 | 2.70 / 61.5 |
| long32768 | 2 | 2.33 / 50.1 | 2.34 / 50.4 | 2.33 / 50.0 |
| long32768 | 3 | 2.77 / 49.4 | 2.73 / 48.8 | 2.73 / 48.6 |

Cache size per token (4 kv heads x 256, K + V): bf16 4096 B, int8 2064 B, int4 1088 B (34k positions: 139 / 70 / 37 MB).

### Native prefill (no transformers model, no snapshot)

| context | HgRT native s (tok/s) | transformers deployed s (tok/s) | llama-mercurius NF4 tok/s | peak GB | first token = reference | ref. top-2 margin | last-logits rel. diff | MTP k=2 lossless from native state |
|---|---|---|---|---|---|---|---|---|
| chat0 (T=27) | 0.35 (78) | 4.90 | – | 3.12 | True | 0.125 | 0.045 | True |
| long1024 (T=1006) | 1.74 (578) | 1.06 (953) | 538 | 3.52 | True | 0.125 | 0.048 | True |
| long8192 (T=8174) | 13.42 (609) | 7.98 (1025) | 551 | 4.03 | True | 0.250 | 0.084 | True |
| long32768 (T=32750) | 65.07 (503) | 41.72 (785) | 479 | 4.70 | False | 0.000 | 0.081 | True |

The transformers chat0 time (4.9 s) includes first-call warm-up inside `prep.py`; the HgRT times are a second call at the same length (Triton / fla autotuning is per shape). Profile at 32K (s): gdn2 22.01, mla 26.277, mlp 17.976, gdn2/qkvz gemm 3.237, gdn2/gates 7.407, gdn2/conv 3.589, gdn2/chunk_gdn2 3.812, gdn2/gated norm 2.464, mla/proj+encode 8.83, mla/decode K/V 1.333, mla/queries 0.902, mla/attention 14.447.


### One speculation cycle, before / after the fused commit (ms; benchmark time / cycles)

| context | k=1 before → after | k=2 | k=3 |
|---|---|---|---|
| chat1 | 30.17 → 29.68 | 35.04 → 34.56 | 40.20 → 39.62 |
| ~1K | 30.50 → 30.08 | 35.54 → 35.04 | 40.97 → 40.29 |
| ~8K | 32.47 → 32.00 | 38.22 → 37.72 | 44.53 → 43.99 |
| ~32K | 38.91 → 38.36 | 47.11 → 46.53 | 56.80 → 56.05 |

The separate commit cost 1.1-1.3 ms per cycle; fusing saves about half (the committed state is still read and written
once). Adaptive k never beat the best fixed k over 256 tokens (its calibration spends one 8-cycle group on each k); it
avoids the worst choice (e.g. chat4, chat5) and is the safer default when the text type is unknown.

## Results (first round)

All HgRT numbers: 256 generated tokens per run after a warm-up, greedy, CUDA graphs; MTP with the 32k-token draft head. "lossless" = the speculative token stream equals the engine's own plain greedy stream (256 tokens). Context = prompt length T from the prefill snapshot (wikitext document + "Summarize the text above" chat prompt for the long ones).

### Decode speed (tok/s)

| context | (a) transformers, deployed | (b) llama-mercurius NF4 / IQ4_NL | (c) HgRT plain | (d) HgRT + MTP k=1 | k=2 | k=3 |
|---|---|---|---|---|---|---|
| 1006 tokens (~1K) | 2.48 | 20.0 / 24.1 | **38.6** | **59.6** (τ 1.81) | **69.5** (τ 2.46) | **70.9** (τ 2.90) |
| 8174 tokens (~8K) | 2.52 | 19.5 / 23.2 | **37.1** | **55.6** (τ 1.80) | **60.7** (τ 2.31) | **60.4** (τ 2.68) |
| 32750 tokens (~32K) | 1.24 | 17.8 / 20.8 | **32.7** | **45.7** (τ 1.77) | **49.7** (τ 2.33) | **49.0** (τ 2.77) |
| chat prompts, T 26-42 (mean of 6) | 2.49 | – | 38.8 | 59.0 (range 53-65) | 65.6 (range 53-78) | 66.8 (range 52-89) |

(a) the deployed OpenDecider/transformers model (`load_deployed`, compressed cache, cache-path greedy loop), timed in `prep.py`. (b) `llama-bench -fa on -p 0 -n 128 -d <depth>` from the fork's build (`raw/llama_bench.log`); the NF4 file holds the same trained NF4 codes, IQ4_NL is a re-quantization. (c)/(d) `experiments/mercurius/bench.py`. Speed-up of (c) over (b, NF4): 1.93x / 1.90x / 1.83x; best MTP over plain: 1.84x / 1.64x / 1.52x.

### MTP acceptance (measured in the speculative runs)

| context | k=1: τ, P(draft accepted) | k=2: τ, histogram of accepted drafts per cycle [0,1,2] | k=3: τ, histogram [0..3] | lossless |
|---|---|---|---|---|
| chat0 (T=27) | 1.74, 0.74 | 2.29, [31, 23, 66] | 2.50, [31, 23, 17, 33] | True |
| chat1 (T=31) | 1.88, 0.88 | 2.55, [15, 17, 72] | 3.09, [11, 15, 17, 45] | True |
| chat2 (T=26) | 1.72, 0.72 | 2.09, [39, 38, 51] | 2.26, [37, 39, 20, 24] | True |
| chat3 (T=27) | 1.75, 0.75 | 2.20, [35, 26, 59] | 2.58, [27, 24, 19, 34] | True |
| chat4 (T=29) | 1.59, 0.59 | 1.86, [61, 42, 41] | 2.06, [48, 40, 24, 16] | True |
| chat5 (T=42) | 1.94, 0.94 | 2.74, [7, 11, 78] | 3.55, [3, 10, 7, 60] | True |
| long1024 (T=1006) | 1.81, 0.81 | 2.46, [19, 18, 67] | 2.90, [16, 15, 28, 37] | True |
| long8192 (T=8174) | 1.80, 0.80 | 2.31, [27, 23, 62] | 2.68, [25, 18, 16, 37] | True |
| long32768 (T=32750) | 1.77, 0.77 | 2.33, [25, 25, 62] | 2.77, [22, 17, 18, 39] | True |

τ = tokens emitted per verification. The draft-1 acceptance (k=1 column, mean 0.78) is a little below the 0.831 measured in PyTorch on the model's own chat output with the bf16 MTP and full head: here the MTP weights are NF4 and the draft head is restricted to 32k ids (full head: τ +0.03..0.14 but ~3 ms more per draft, net slower; `raw/draft_vocab_*.out`).

### Agreement with the reference (deployed transformers path)

| context | first divergence (of compared) | teacher-forced argmax agreement | reference median top-1/top-2 logit margin |
|---|---|---|---|
| chat0 | 9 / 256 | 0.977 | 2.25 |
| chat1 | 256 / 256 | 1.000 | – |
| chat2 | 31 / 256 | 0.957 | – |
| chat3 | 48 / 256 | 0.988 | – |
| chat4 | 23 / 256 | 0.953 | – |
| chat5 | 88 / 256 | 0.992 | – |
| long1024 | 64 / 64 | 1.000 | – |
| long8192 | 10 / 64 | 0.953 | 2.88 |
| long32768 | 7 / 64 | 0.922 | 1.50 |

Teacher forcing feeds the reference's tokens to the engine and compares each argmax with the reference's next token. Every disagreement found (chat0, long8192, long32768; `diag.json`) is at a reference near-tie: reference margins chat0: 0.125/0.125/0.125/0.5/0.125/0/0, long8192: 0.125/0.125/0.125, long32768: 0/0.125/0/0.125/0/0 (its logits are bf16, steps of 0.125 at this magnitude, so 0.0 is an exact tie). Once a free-running stream takes the other branch of a tie the two texts differ, which is why the first divergence is early on some prompts. Sources of difference vs the reference: int8 activation groups in the GEMVs, fp16 tensor-core attention (the reference decodes and expands the cache in bf16), the recurrent vs chunked GDN-2 form, accumulation orders.

### Per-token time, plain decode (ms; components timed alone in CUDA graphs, so they sum only approximately)

| component | chat (T=27) | 8K | 32K |
|---|---|---|---|
| 24 GDN-2 layers | 8.99 | 8.99 | 9.17 |
| 8 MLA layers | 2.59 | 3.82 | 7.53 |
| 32 MLPs | 10.53 | 10.77 | 12.52 |
| LM head (NF4, 248k rows) + argmax | 2.76 | 2.91 | 2.89 |
| **whole step, measured** | 25.95 | 27.30 | 30.75 |

One GDN-2 layer (µs): rms+quant 14, qkv|z NF4 GEMV 142, gates: base|A|loraA (tensor cores) 69, gates: VeRA B (tensor cores) + decay/sigmoid 123, conv 4, recurrence+gated norm 14, out NF4 GEMV 65. One MLA layer (r = 656; µs at chat / 8K / 32K): qdk NF4 GEMV 166/127/119, k_rope / R0 / R small GEMVs 14/14/14, query map 46/45/45, attention from codes (+reduce) 44/212/719, V up + gate 21/21/21, o NF4 GEMV 55/55/55.

Bytes per plain token: 2.00 GB NF4 trunk + 0.36 GB NF4 head + 0.33 GB bf16 VeRA factors (A, B read once per GDN-2 layer) + small tensors + GDN-2 states (96 MB) ≈ 2.8 GB → 108 GB/s effective at short context (read ceiling ~150 GB/s at MODE_50W).

### One speculation cycle (ms)

| | k=1 | k=2 | k=3 |
|---|---|---|---|
| verification block, M = k+1 (alone), chat1 | 27.95 | 31.50 | 35.44 |
| verification block, M = k+1 (alone), 8K | 29.46 | 33.63 | 38.76 |
| verification block, M = k+1 (alone), 32K | 35.55 | 41.31 | 48.61 |
| GDN-2 commit, chat1 | 1.09 | 1.25 | 1.33 |
| GDN-2 commit, 8K | 1.18 | 1.24 | 1.35 |
| GDN-2 commit, 32K | 1.09 | 1.27 | 1.32 |
| MTP step 1 (M catch-up rows) + draft, chat1 | 1.11 | 1.14 | 1.16 |
| MTP step 1 (M catch-up rows) + draft, 8K | 1.40 | 1.46 | 1.51 |
| MTP step 1 (M catch-up rows) + draft, 32K | 2.17 | 2.22 | 2.22 |
| MTP step j + draft, chat1 | 1.08 | 1.08 | 1.07 |
| MTP step j + draft, 8K | 1.40 | 1.44 | 1.37 |
| MTP step j + draft, 32K | 2.15 | 2.16 | 2.14 |
| **full cycle, measured**, chat1 | 30.12 | 35.00 | 40.45 |
| **full cycle, measured**, 8K | 32.20 | 38.09 | 44.44 |
| **full cycle, measured**, 32K | 38.79 | 46.97 | 56.66 |

Plain step for comparison: 25.9 / 27.0 / 30.6 ms. The verification block costs ~1.1x / 1.25x / 1.4x a plain step at short context for k = 1/2/3 (the NF4 GEMV with M token rows runs at 90-120 GB/s instead of 125-142, the head at M=4 is 3.7 vs 2.9 ms); at 32K the code-reading attention with M query rows dominates (one MLA layer 1.4 / 2.1 / 2.6 ms vs 0.7 ms plain).

### NF4 GEMV bandwidth (tuned tiling; M = 1 and M = 4)

| matrix | N x K | M=1 µs (GB/s) | M=4 µs (GB/s) | one-table approx., M=1 µs |
|---|---|---|---|---|
| mlp gate|up | 18432 x 2560 | 208 (128) | 282 (94) | 187 |
| mlp down | 2560 x 9216 | 108 (122) | 141 (94) | 100 |
| gdn qkv|z | 12288 x 2560 | 133 (134) | 192 (92) | 129 |
| gdn out | 2560 x 4096 | 48 (123) | 68 (86) | 39 |
| mla q|down|krms r=579 | 9795 x 2560 | 114 (124) | 157 (90) | 109 |
| mla q|down|krms r=451 | 9667 x 2560 | 111 (125) | 153 (91) | 109 |
| mla q|down|krms r=390 | 9606 x 2560 | 112 (123) | 152 (91) | 106 |
| mla q|down|krms r=427 | 9643 x 2560 | 113 (123) | 153 (91) | 108 |
| mla q|down|krms r=543 | 9759 x 2560 | 114 (123) | 158 (89) | 109 |
| mla q|down|krms r=471 | 9687 x 2560 | 111 (126) | 155 (90) | 106 |
| mla q|down|krms r=656 | 9872 x 2560 | 113 (126) | 163 (87) | 109 |
| mla o | 2560 x 4096 | 51 (117) | 71 (83) | 42 |
| lm head | 248320 x 2560 | 2510 (142) | 3743 (96) | 2485 |
| mtp fc | 2560 x 5120 | 60 (122) | 83 (89) | 56 |
| mtp q|k|v | 10240 x 2560 | 114 (129) | 161 (92) | 112 |
| draft head (32k vocab) | 32768 x 2560 | 354 (133) | 507 (93) | 333 |

### Memory (peak, GB)

| | chat | ~1K | 8K | 32K |
|---|---|---|---|---|
| HgRT engine + MTP (torch peak, engine sized for 34k positions) | 3.31 | 3.31 | 3.35 | 3.46 |
| transformers deployed (torch peak incl. prefill) | 3.46 | 3.61 | 4.92 | 9.46 |
| llama-mercurius | – | – | – | not re-measured; its README: 2.68 GB at 16k, 2.86 GB at 64k |

HgRT memory: 2.36 GB NF4 weights (trunk + tied table) + 0.14 GB bf16 small tensors (VeRA factors, folded MLA maps) + MTP (NF4, 68 MB) + MLA code cache (256-384 B per token per layer, 2.5 KB per token over the 8 layers: 34k positions ≈ 87 MB) + MTP bf16 cache (4 KB/token ≈ 140 MB) + buffers. The MLA cache stores per token and layer the position, 4 x fp32 1/rms, two fp16 norms, 32 B RoPE codes and ⌈r/2⌉ B latent codes (padded to 16 B).


## What was built

New files only. The uncommitted changes in `csrc/gemv.cu`, `csrc/ops.cu` and `engine.py` were not touched; the MTP
path imports hgrt's existing `ops.dense_attn` / `ops.attn_post` unchanged.

| file | what |
|---|---|
| `src/hgrt/csrc/mercurius_ops.cu` | all Mercurius kernels (below) |
| `src/hgrt/mercurius_kernels.py` | JIT build (same mechanism as `kernels.py`, build dir `~/.cache/hgrt/mercurius`), NF4 level tables, code layouts |
| `src/hgrt/mercurius_engine.py` | `MercuriusEngine`: weights, buffers, the M-token block (M = 1 plain decode, M = k+1 verification), CUDA graphs, snapshot hand-off |
| `src/hgrt/mercurius_prefill.py` | native prefill (follow-up round; needs `/mnt/ext/hgrt-kernels/mercurius/prefill_extra.pt`, the expanded-form maps extracted from the NF4 GGUF) |
| `src/hgrt/mercurius_mtp.py` | `MTPHead` (stock Qwen3.5-4B MTP, NF4-quantized, own bf16 KV cache, prompt-cache prefill) and `SpecDecoder` (one CUDA graph per speculation cycle) |
| `tests/test_mercurius.py` | kernels vs PyTorch references, engine vs a PyTorch re-implementation (per layer and end to end), graph == eager, verification == plain decoding bit for bit |
| `experiments/mercurius/prep.py` | the one load of the deployed transformers model (~8 min): inspects it, dumps the deployed tensors (`/mnt/ext/hgrt-kernels/mercurius/weights.pt`, 2.5 GB), prefill snapshots and reference generations with timing |
| `experiments/mercurius/{bench,diag,gemv_tune,gates_bench,attn_bench,mma_check,llama_bench.sh,report}.py`, `prefill_bench.py`, `wa_sweep.py`, `llama_prefill.sh`, `report_round2.py` | benchmarks, profiles, tuning, the llama.cpp baseline |
| `experiments/mercurius/lab/` | separate (non-engine) copies of the attention kernel: clock64 phase instrumentation and the variants tried |
| `results/mercurius/` | this file, `bench.json` (the tables), `diag.json`, `diag_cycle.json`, `gemv_tune.json`, `gemv_rows.json` (read by the engine), `raw/` (logs, earlier runs, baselines) |

## Design

**One CUDA graph per token** (plain decode) or per speculation cycle: positions, tokens, accepted counts all live on the
GPU; the host only replays graphs (and syncs every 8 cycles to check the emitted count).

**Weights: exact NF4.** The deployed weights are bitsandbytes NF4 (blocks of 64, fp32 absmax after its double
quantization), the grid Mercurius was QAT-trained on. The engine keeps those codes and scales bit for bit (repacked;
`prep.py` checks that decoding them reproduces `bnb.dequantize_4bit` exactly for all 200 matrices; the NF4 embedding table is verified row by row by `NF4Table`) and does **not**
re-quantize. The GEMV decodes each 4-bit code with byte-permute lookups into two int8 tables, `128·h + l = round(16256·L)`
(|level error| ≤ 2.5e-5, about 1/20 of a bf16 ulp of the smallest nonzero level and less for the others), and does two dp4a per 4 values against int8
activations (per-32 groups, the HgRT/llama.cpp Q8 convention). A half-warp-per-row mapping was needed for K = 2560
(80 groups do not divide among 32 lanes): +20-30% on those matrices. The LM head is the tied NF4 embedding table, so
it runs through the same exact NF4 kernel (fp32 logits) — cheaper than an int8 head (0.36 GB vs 0.64 GB) and exact to
the deployed model. Measured: 120-142 GB/s at M = 1 against a ~150 GB/s read ceiling; an approximate one-table
variant (levels rounded to 1/127, i.e. a re-quantization) was only 2-10% faster, so the exact kernel is the default.

**GDN-2 (24 layers).** RMSNorm (ScaleNorm gains) fused with the int8 activation groups; qkv|z NF4 GEMV; the three
channel-wise gates in their factored deployed form — tiled base rows + VeRA (rank 1024, A and B shared by all layers)
+ the rank-32 decay LoRA — as two skinny bf16 GEMMs on tensor cores (inputs are exact bf16, fp32 accumulation;
cp.async-staged, run on a side stream next to the qkv|z GEMV); causal conv (kernel 4); the recurrence with the 128x128
fp32 state column in registers (fla `fused_recurrent_gdn2` arithmetic: per-channel decay, erase gate on keys, write
gate on values), gated RMSNorm, quantization for the out-projection. No decay phase is active in the deployed model
(`pe_c` is None in all 24 layers; the kernel supports it).

**Absorbed MLA from the 4-bit codes (8 layers).** Per token: q|latent|k_rms in one NF4 GEMV; RoPE key (NF4-rounded bf16
weight, then R0) and latent rotation R as small GEMVs; TurboQuant encode (fp16 norm, Beta Lloyd-Max codebook,
`bucketize` semantics) straight into the cache row at the token's position (row: pos, 4 per-group 1/rms, two fp16
norms, 32 B RoPE codes, ⌈r/2⌉ B latent codes); the per-head query map P_h (post-norm map, RoRoPE mixing and absorbed
key up-projections folded, from the GGUF converter's `fold_attention`) + RoPE. Attention is split-KV over 32-cell
chunks on tensor cores: each chunk's packed rows are cp.async-staged, decoded in shared memory to fp16 (rope part
un-rotated by R0 with an MMA and re-roped at the cell's position), S = Q·Kᵀ, per-group 1/rms, online softmax, O += P·C
on the latent — the whole context is never decompressed and no attention matrix is materialized. Partials are merged
in a fixed order; then the per-group V up-projection and the sigmoid output gate. sin/cos use an explicit Cody-Waite
reduction (with `--use_fast_math`, `cosf` of 30k rad is off by ~1e-2; the reference uses torch's accurate cos).

**MTP self-speculative decoding.** The stock Qwen3.5-4B MTP module (15 tensors, unchanged) with the verified wiring
(normed trunk hidden, `fc([norm_e(emb(x_{t+1})); norm_h(h_t)])`, one gated attention layer with its own cache, `mtp.norm`,
tied head). Its bf16 weights are quantized once to NF4 blocks (they only affect drafting); its cache is bf16 head-major,
read by hgrt's split-K kernel; prompt positions are filled by a PyTorch prefill from the snapshot's normed hidden
states (the **full MTP cache**, which mtp_kv.log showed matters). The draft head is the tied head restricted to the 32k
most frequent tokens (verification always uses the full head, so this cannot affect correctness). One cycle (one CUDA
graph): MTP rows for the previous cycle's k+1 verified positions (their cache entries are rewritten from the true
hidden states; only the last accepted row is carried to a draft) → k−1 recursive drafts → trunk verification of
[x_P, d_1..d_k] in ONE M = k+1 pass (GEMVs with M activation rows, GDN-2 recurrence over the M steps without writing the
state, MLA cache rows for all M positions, NQ = M query attention) → greedy acceptance on the GPU → commit: the GDN-2
conv and recurrent states are recomputed for the n+1 accepted steps from the saved per-step inputs (exact rollback),
the MLA rows of rejected tokens are simply overwritten later.

**Lossless by construction, and tested.** Every kernel computes each of its M rows with the arithmetic of the M = 1
case: the GEMV's per-row reduction order depends only on the matrix (lane mapping fixed per matrix, never on M), the
tensor-core gate GEMMs give a column only its own input row, the attention assigns chunks to fixed "virtual splits"
so a query row's math does not depend on how many rows or cells there are (masked cells contribute exact zeros). The
test suite checks verification + commit == plain decoding **bit for bit** (logits, GDN states, MLA cache) for M = 2, 3, 4,
and every benchmark run compares the speculative token stream with the engine's own plain greedy stream.

## Verification

`tests/test_mercurius.py` (all on the real deployed weights):

- NF4 GEMV (all tilings, residual, fp32 head) vs an fp64 reference of the same integer math: rel 1.7e-3 (= bf16 output
  rounding) / 9e-8 (fp32 out); M-row results == single-row results bit for bit.
- RMSNorm+quant, conv, gates (CUDA-core and tensor-core paths): exact or ≤ 1e-7 vs PyTorch; GDN-2 recurrence vs fla's
  `naive_recurrent_gdn2`: output 1.7e-3 (bf16), state 1e-7; verify + commit(n) == plain for n = 0, 1, 2.
- TurboQuant encode == `mercurius_cache.tq_encode` (0 mismatching codes); query map 2e-4; attention from codes vs an
  fp32 decode-everything reference: 2e-4 .. 9e-4 for 1..5000 cells and 1..4 query rows, bit identity across row counts.
- Engine vs a PyTorch re-implementation of the whole step (same weights, int8 activations emulated), per layer from the
  engine's own input: GDN-2 layers ≤ 5e-6, MLA layers ≈ 1.1-1.35e-2 (fp16 tensor-core attention vs an fp32-key
  reference, on the real peaked score distributions); end to end over 32 layers 3.7e-2 relative on the logits, argmax
  6/6. CUDA graph == eager; verification == plain (bit).

## Not done / limitations (honest list)

- **Native prefill (follow-up) is 1.6-1.8x slower than the transformers path** (GEMM-bound MLP at ~8 TFLOP/s, a Triton
  attention kernel at ~6.5 TFLOP/s, and unfused elementwise gate / norm chains: 32K profile in the table above); its
  last-position logits differ from the transformers prefill by 4.5-8.4% relative (bf16 GEMM order, factored vs folded
  VeRA rounding, attention from the 4-bit codes), and the first token differs only at 32K, where the reference itself
  has an exact tie. The snapshot hand-off (`load(snap)`) is still supported.
- **Long-context attention is latency-bound.** At 32K one MLA layer costs ~0.7 ms (plain) and 1.4/2.1/2.6 ms in the
  M = 2/3/4 verification, against rough floors of ~0.1 ms (code bytes) / tensor-core throughput; a clock64 phase profile
  (`raw/attn_phase_profile.log`) shows decode / S / softmax / PV each a few thousand cycles per 32-cell chunk. Tried:
  merged per-token phases + virtual splits (correct, ±0%), flattened decode (no gain), 512-thread blocks (−14..23% for
  NQ ≥ 2, used; +5..15% for NQ = 1, not used). This is why MTP gains shrink at 32K.
- **MTP cache:** compressed per-row int8 / int4 forms instead of FoveaKV gain-shape pages (simpler, graph-friendly with
  speculative positions); they cut memory 2-4x with unchanged acceptance but did not change speed: the MTP step at 32K
  (2.15 ms vs 1.07 ms short) is bound by the decode kernel's per-token warp reductions, not by cache bytes, so page
  selection (FoveaKV) would not help either until that kernel is restructured. The DepthSkip ridge map is not
  applicable (prefill is full depth).
- **int8 tensor-core NF4 GEMM** (`nf4_mma`, kept in the extension, `gemv="mma"`): exact and M-invariant in cost
  (~260-320 µs for the MLP up-projection at any M) but 10-40% slower than the dp4a GEMV at M = 1, so plain decode and
  verification (which must share one kernel to stay lossless) use dp4a: summed over the model, 20.6 vs 24.9 ms (M = 1)
  and 25.8 vs 24.4 ms (M = 4).
- int8 activation groups (the HgRT/llama.cpp convention) and fp16 attention operands are deviations from the bf16
  reference arithmetic; their effect is the agreement table above (disagreements only at reference near-ties).
- The draft vocabulary (32k ids by frequency on wikitext + local code/markdown) is a heuristic; the full head gives a
  slightly higher τ but costs ~3 ms per draft (`raw/draft_vocab_*.out`).
- Measurements: the GPU (306-816 MHz) and EMC (2133/3199 MHz) clocks are DVFS-governed (MODE_50W untouched); every
  timing warms up for ≥ 0.5 s first, but micro-benchmarks still vary by ~±5% between runs.

## Recommended next steps (after the follow-up round)

1. Long-context attention is still the main lever and two designs (shared-memory K tile, register-decoded K) both sit
   ~10x above the memory floor at low IPC. What has not been tried: Nsight-level profiling (not available here:
   CUPTI is blocked for this user), a warp-specialized producer/consumer split (decode warps feeding MMA warps through
   a multi-stage ring), and 64-cell chunks with 16 warps sharing one K tile.
2. The MTP decode kernel: restructure the per-token warp reductions (multi-value butterfly or tensor cores for the
   4-query-head groups); only then does cache compression (int8/int4 here, or FoveaKV pages) turn into speed at 32K.
3. Prefill: fuse the gate / norm elementwise chains (a CUDA epilogue already exists for the gates), a CUTLASS-class
   attention kernel for 448/256, and NF4 GEMMs that dequantize in shared memory instead of materializing bf16 weights.
4. Adaptive k: a cheaper calibration (reuse costs across contexts, or predict them from the context length).

## How to run

```bash
# once (~15 min under the GPU lock): weights + snapshots + references
experiments/mercurius/prep.py                      # writes /mnt/ext/hgrt-kernels/mercurius/{weights.pt,snap_*,ref_*}
# tests, tuning, benchmarks (each under ~/qwen-surgery/gpu_lock.sh; see /mnt/ext/hgrt-kernels/mercurius/run.sh)
tests/test_mercurius.py
experiments/mercurius/gemv_tune.py                 # -> results/mercurius/gemv_rows.json (read by the engine)
experiments/mercurius/bench.py --ctx chat0,...,long32768 --n 256 --ks 1,2,3 --vocab 32768
experiments/mercurius/diag.py chat0,long8192,long32768          # margins of disagreements + component profile
DIAG_CYCLE=chat1,long32768 experiments/mercurius/diag.py        # speculation-cycle profile
experiments/mercurius/llama_bench.sh               # llama-mercurius baseline
```

```python
from hgrt.mercurius_engine import MercuriusEngine
from hgrt.mercurius_mtp import MTPHead, SpecDecoder
E = MercuriusEngine("/mnt/ext/hgrt-kernels/mercurius/weights.pt", Tmax=34000)
snap = torch.load(".../snap_long8192.pt")
E.load(snap); toks = E.generate(256)                                  # plain greedy (one CUDA graph per token)
H = MTPHead(E, draft_vocab=torch.load(".../draft_vocab.pt")[:32768])
S = SpecDecoder(E, H, k=2); E.load(snap); S.start(snap)
toks2, cycles, hist = S.generate(256)                                 # == toks
```

