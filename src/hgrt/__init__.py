"""HgRT: the Mercurius runtime and long-context stack.

Mercurius-1-4B decode engine (exact NF4 GEMVs, GDN-2, absorbed MLA over a TurboQuant 4-bit cache, one CUDA graph per
token, lossless MTP self-speculative decoding, native prefill) and the long-context stack for Qwen3.5: DepthSkip prefill
(graded per-token depth), FoveaKV 2+2-bit KV pages, SparsePage page selection for decode, and OpenDecider decisions on
top."""
__version__ = "0.1.0"
