"""One-time preparation for the HgRT Mercurius engine (loads the deployed transformers model ONCE, ~8 min):

  1. prints the module structure facts the engine relies on (norm types, eps, scaling, decay phase, ranks);
  2. dumps the deployed tensors to OUT/weights.pt: NF4 codes + block scales exactly as stored by bitsandbytes
     (repacked: element 2i in the low nibble), the folded absorbed-MLA matrices, gates (factored), small vectors;
  3. reference generations with the deployed transformers path (greedy, cache path): chat prompts (320 tokens) and
     long-context prompts (1K / 8K / 32K, 64 tokens), with per-token timing and peak memory;
  4. prefill snapshots for every prompt (TurboQuant codes per MLA layer, GDN conv / recurrent states, normed final
     hidden of every prompt position, last logits) for the engine's state hand-off.

    PYTHONPATH=... python prep.py [--skip-weights] [--only chat|long]
"""
import argparse
import json
import os
import sys
import time

import torch

OD = os.environ.get("OPENDECIDER_ROOT", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), "third_party", "opendecider"))
sys.path.insert(0, OD + "/src")
sys.path.insert(0, OD + "/deploy")
ROOT = os.environ.get("MERCURIUS_ROOT", "/mnt/ext/opendecider/mercurius-work")
OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")
PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, and explain its complexity.",
    "Summarize the causes of the French Revolution in a few paragraphs.",
    "What is the difference between TCP and UDP? Give use cases for each.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed in m/s?",
]
LONG = [1024, 8192, 32768]


def nf4_parts(lin):
    """bitsandbytes Linear4bit -> (packed codes (out, in/2) uint8, element 2i LOW nibble; absmax (out, in/64) fp32)."""
    import bitsandbytes.functional as bnbf
    w = lin.weight
    st = w.quant_state
    out, inn = st.shape
    assert st.blocksize == 64 and inn % 64 == 0, (st.blocksize, st.shape)
    q = w.data.reshape(-1)
    hi, lo = q >> 4, q & 15                                  # bitsandbytes: element 2i in the HIGH nibble
    packed = (hi | (lo << 4)).view(out, inn // 2).contiguous()
    if st.nested:
        absmax = (bnbf.dequantize_blockwise(st.absmax, st.state2) + st.offset).float()
    else:
        absmax = st.absmax.float()
    absmax = absmax.view(out, inn // 64).contiguous()
    # exactness: our decode == bitsandbytes' own dequantization, bit for bit
    code = st.code.float()
    ref = bnbf.dequantize_4bit(w.data, st)
    c2 = torch.stack([packed & 15, packed >> 4], -1).view(out, inn).long()
    mine = (code[c2].view(out, inn // 64, 64) * absmax[..., None]).view(out, inn).to(ref.dtype)
    ok = torch.equal(mine, ref)
    return packed.cpu(), absmax.cpu(), code.cpu(), ok


@torch.no_grad()
def dump_weights(trunk, tok, path):
    import bitsandbytes as bnb
    from mercurius_gguf import fold_attention
    from opendecider.mercurius_port import TiledLinear
    from mercurius.models.qat import tq_codebook, nf4_roundtrip
    W = {"nf4": {}, "t": {}, "meta": {}}
    bad = []

    def put_nf4(name, lin):
        assert isinstance(lin, bnb.nn.Linear4bit), (name, type(lin))
        p, a, code, ok = nf4_parts(lin)
        W["nf4"][name] = (p, a)
        W["meta"]["nf4_code"] = code
        if not ok:
            bad.append(name)

    def gain(norm):
        """the multiplier the fused rms applies: 1 + w (Qwen3.5 RMSNorm) or 1 + (g - 1) (ScaleNorm), fp32."""
        w = norm.weight.data.float()
        if w.numel() == 1:
            return (1.0 + (w - 1.0)).reshape(1)
        return 1.0 + w

    L = len(trunk.layers)
    for i, layer in enumerate(trunk.layers):
        p = f"l{i}."
        W["t"][p + "ln1"] = gain(layer.input_layernorm).cpu()
        W["t"][p + "ln2"] = gain(layer.post_attention_layernorm).cpu()
        W["meta"][p + "ln_types"] = (type(layer.input_layernorm).__name__, type(layer.post_attention_layernorm).__name__)
        for k in ("gate_proj", "up_proj", "down_proj"):
            put_nf4(p + k, getattr(layer.mlp, k))
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            put_nf4(p + "qkv", la.in_proj_qkv)
            put_nf4(p + "z", la.in_proj_z)
            put_nf4(p + "out", la.out_proj)
            W["t"][p + "conv"] = la.conv1d.weight.data.float().squeeze(1).cpu()                 # (C, 4)
            assert la.conv1d.bias is None
            W["t"][p + "A_log"] = la.A_log.data.float().cpu()
            W["t"][p + "dt_bias"] = la.dt_bias.data.float().cpu()
            W["t"][p + "gn"] = la.norm.weight.data.cpu()                                          # bf16 (gated norm)
            W["meta"][p + "gn_eps"] = float(getattr(la.norm, "variance_epsilon", getattr(la.norm, "eps", 1e-6)))
            for g in ("a", "be", "bw"):
                m = getattr(la, f"in_proj_{g}")
                assert isinstance(m.base, TiledLinear), (p, g, type(m.base))
                W["t"][p + f"U_{g}"] = m.base.unique.data.cpu()                                   # (nv, H) bf16
                W["t"][p + f"vd_{g}"] = m.vera_d.data.cpu()
                W["t"][p + f"vb_{g}"] = m.vera_b.data.cpu()
                if "vera_A" not in W["t"]:
                    W["t"]["vera_A"] = m.vera_A.data[:, :m.base.in_features].cpu()
                    W["t"]["vera_B"] = m.vera_B.data[:m.base.out_features].cpu()
                else:
                    assert torch.equal(W["t"]["vera_A"], m.vera_A.data[:, :m.base.in_features].cpu())
            if getattr(la, "a_lora_A", None) is not None and la.lora_rank > 0:
                W["t"][p + "loraA"] = la.a_lora_A.data.cpu()
                W["t"][p + "loraB"] = la.a_lora_B.data.cpu()
            if getattr(la, "pe_c", None) is not None:
                W["t"][p + "pe_c"] = la.pe_c.data.float().cpu()
        else:
            sa = layer.self_attn
            f = fold_attention(sa)
            put_nf4(p + "q", sa.q_proj)
            put_nf4(p + "o", sa.o_proj)
            put_nf4(p + "krms", sa.k_rms)
            lat = sa.k_proj.latent
            put_nf4(p + "down", lat.down)
            W["t"][p + "qn"] = (1.0 + sa.q_norm.weight.data.float()).cpu()
            W["t"][p + "P"] = f["q_map"].cpu()                                    # (H, rd + r, D) fp32
            W["t"][p + "Wv"] = f["w_v"].cpu()                                     # (G, D, r) fp32, on the rotated latent
            w = lat.k_rope.weight
            Wr = (w + (nf4_roundtrip(w) - w)).to(torch.bfloat16)                  # as _encode_codes computes it
            W["t"][p + "Wr"] = Wr.cpu()
            W["t"][p + "R0"] = sa._qat_R0.data.to(torch.bfloat16).cpu() if hasattr(sa._qat_R0, "data") else sa._qat_R0.to(torch.bfloat16).cpu()
            W["t"][p + "R"] = lat._tq_rot.to(torch.bfloat16).cpu()
            r, rd = f["r"], f["rd"]
            for nm, dim in (("cbl", r), ("cbr", rd)):
                cen, bnd = tq_codebook(dim, 4)
                W["t"][p + nm] = torch.cat([cen.float(), bnd.float()]).cpu()      # 16 centroids + 15 bounds
            W["meta"][p + "attn"] = {"r": r, "rd": rd, "H": f["H"], "G": f["G"], "D": f["D"], "scale": f["scale"],
                                     "q_eps": f["q_norm_eps"], "k_eps": f["k_norm_eps"]}
    W["t"]["norm_f"] = gain(trunk.norm).cpu()
    E = trunk.embed_tokens
    W["t"]["emb_codes"] = E.codes.cpu()                    # bitsandbytes packing (element 2i high nibble), (V, D/2)
    W["t"]["emb_absmax"] = E.absmax.float().cpu()          # (V, D/64)
    W["t"]["emb_code"] = E.code.float().cpu()
    W["meta"]["vocab_tok"] = len(tok)
    W["meta"]["nf4_mismatch"] = bad
    torch.save(W, path)
    print(f"weights -> {path}; NF4 decode mismatches vs bitsandbytes: {bad}", flush=True)


def inspect(trunk):
    l0, l3 = trunk.layers[0], trunk.layers[3]
    la, sa = l0.linear_attn, l3.self_attn
    print("layer0", type(l0.input_layernorm).__name__, l0.input_layernorm.weight.shape,
          type(l0.post_attention_layernorm).__name__, "| final", type(trunk.norm).__name__, trunk.norm.weight.shape)
    print("linear_attn", type(la).__name__, "pe_c", getattr(la, "pe_c", None) is not None, "lora_rank",
          getattr(la, "lora_rank", None), "in_proj_a", type(la.in_proj_a).__name__, "base", type(la.in_proj_a.base).__name__,
          "vera rank", getattr(la.in_proj_a, "rank", None), "norm", type(la.norm).__name__,
          "conv bias", la.conv1d.bias is not None, "act", la.activation)
    print("self_attn", type(sa).__name__, "q_proj", type(sa.q_proj).__name__, "q_norm", type(sa.q_norm).__name__,
          "k_norm", type(sa.k_norm).__name__, "k_rms", type(sa.k_rms).__name__, "scaling", sa.scaling,
          "rd", sa._rd, "G", sa._G, "H", sa._H, "n_rope", sa._n_rope)
    print("latent ranks", [l.self_attn.k_proj.latent._tq_rot.shape[0] for l in trunk.layers if hasattr(l, "self_attn")])
    print("embed", type(trunk.embed_tokens).__name__, "rotary", trunk.rotary_emb.inv_freq.shape,
          trunk.rotary_emb.inv_freq[:4].tolist(), getattr(trunk.rotary_emb, "attention_scaling", None), flush=True)


def head_W(trunk):
    E = trunk.embed_tokens
    return E.dequant_rows(slice(None), torch.bfloat16)


@torch.no_grad()
def snapshot(trunk, cache, last_hidden, ids):
    """state hand-off for the engine (CPU tensors)."""
    from opendecider.mercurius_cache import CODE_ATTRS
    s = {"ids": ids.cpu(), "attn": {}, "lin": {}, "hidden": last_hidden[0].to(torch.bfloat16).cpu()}
    for i, layer in enumerate(trunk.layers):
        cl = cache.layers[i]
        if hasattr(layer, "self_attn"):
            s["attn"][i] = {a: getattr(cl, a)[0].cpu() for a in CODE_ATTRS}
        else:
            s["lin"][i] = (cl.conv_states[0][0].cpu(), cl.recurrent_states[0][0].float().cpu())
    return s


@torch.no_grad()
def reference_gen(trunk, W, V, ids, n, eos, stop_eos=True):
    """greedy through the deployed path's cache, per-token timing; returns tokens, top-2 margins, timings, snapshot."""
    torch.cuda.synchronize()
    t0 = time.time()
    o = trunk(input_ids=ids[None], use_cache=True)
    cache = o.past_key_values
    lg = (o.last_hidden_state[0, -1:].to(torch.bfloat16) @ W.T)[:, :V].float()
    torch.cuda.synchronize()
    t_pf = time.time() - t0
    snap = snapshot(trunk, cache, o.last_hidden_state, ids)
    snap["last_logits"] = lg[0].cpu()
    toks, margins = [], []
    nxt = lg.argmax(-1)
    torch.cuda.synchronize()
    t1 = time.time()
    times = []
    for j in range(n):
        toks.append(nxt)
        top = lg.topk(2, -1).values
        margins.append(top[:, 0] - top[:, 1])
        o = trunk(input_ids=nxt[None], past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        lg = (o.last_hidden_state[0, -1:].to(torch.bfloat16) @ W.T)[:, :V].float()
        nxt = lg.argmax(-1)
        if j % 16 == 15:
            torch.cuda.synchronize()
            times.append(time.time())
    torch.cuda.synchronize()
    t_dec = time.time() - t1
    toks = torch.cat(toks).cpu()
    return {"tokens": toks, "margins": torch.cat(margins).cpu(), "prefill_s": t_pf, "decode_s": t_dec, "n": n,
            "ms_per_token": t_dec / n * 1e3}, snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--gen", type=int, default=320)
    ap.add_argument("--long-gen", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from opendecider.mercurius_port import load_deployed
    t0 = time.time()
    trunk, tok = load_deployed(ROOT)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    inspect(trunk)
    if not a.skip_weights:
        dump_weights(trunk, tok, os.path.join(OUT, "weights.pt"))
    V = len(tok)
    W = head_W(trunk)
    eos = [i for i in [tok.eos_token_id, *tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])] if i is not None]
    json.dump({"vocab_tok": V, "eos": eos}, open(os.path.join(OUT, "tok.json"), "w"))
    res = {}
    if a.only in ("", "chat"):
        for k, p in enumerate(PROMPTS):
            s = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
            ids = torch.tensor(tok(s).input_ids, device="cuda")
            torch.cuda.reset_peak_memory_stats()
            r, snap = reference_gen(trunk, W, V, ids, a.gen, eos)
            r["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            torch.save(snap, os.path.join(OUT, f"snap_chat{k}.pt"))
            torch.save(r, os.path.join(OUT, f"ref_chat{k}.pt"))
            res[f"chat{k}"] = {kk: v for kk, v in r.items() if not torch.is_tensor(v)} | {"T": int(ids.shape[0])}
            print(f"chat{k}: T={ids.shape[0]} {r['ms_per_token']:.1f} ms/token | {tok.decode(r['tokens'][:48])!r}", flush=True)
    if a.only in ("", "long"):
        text = open(os.environ.get("HGRT_WIKITEXT", "/mnt/ext/foveakv/data/wikitext2_test.txt")).read()
        doc_ids = tok(text[:400000]).input_ids
        for L in LONG:
            body = tok.decode(doc_ids[:L - 40])
            s = tok.apply_chat_template([{"role": "user", "content": body + "\n\nSummarize the text above in detail."}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = torch.tensor(tok(s).input_ids, device="cuda")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            r, snap = reference_gen(trunk, W, V, ids, a.long_gen, eos)
            r["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
            torch.save(snap, os.path.join(OUT, f"snap_long{L}.pt"))
            torch.save(r, os.path.join(OUT, f"ref_long{L}.pt"))
            res[f"long{L}"] = {kk: v for kk, v in r.items() if not torch.is_tensor(v)} | {"T": int(ids.shape[0])}
            print(f"long{L}: T={ids.shape[0]} prefill {r['prefill_s']:.1f}s, {r['ms_per_token']:.1f} ms/token, "
                  f"peak {r['peak_gb']:.2f} GB | {tok.decode(r['tokens'][:32])!r}", flush=True)
            del snap
    old = json.load(open(os.path.join(OUT, "ref_timing.json"))) if os.path.exists(os.path.join(OUT, "ref_timing.json")) else {}
    old.update(res)
    json.dump(old, open(os.path.join(OUT, "ref_timing.json"), "w"), indent=1)
    print("PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
