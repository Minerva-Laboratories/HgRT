"""The whole stack on one model: DepthSkip prompt pass -> handoff of its cache -> decode with a compressed / selected cache.

  prefill   "exact" or a DepthSkip depth plan (PREFILLS): the prompt minus its last token goes through
            depthskip.segment_prefill (graded per-token depth, upper-layer caches from ridge maps); the last token runs
            every layer on that cache and gives the first next-token logits.
  handoff   the Hugging Face cache becomes the decode loop's state: post-RoPE K/V of the full-attention layers (T, Hkv,
            D) and the DeltaNet conv / recurrent states (sparsepage.qwen.LinCache).
  decode    any cache by name (hgrt.caches): "stack10" = FoveaKV 2+2-bit pages + SparsePage top 10% + bf16 window,
            "sparsepage10", "bf16", "gs4", "dyn10", ...

Maps: one SkipMaps per cut, fitted on the model itself (depthskip.fit_skip_maps, 200 FineWeb-Edu documents x 256
tokens; 2-5 minutes per cut on the Orin) and cached in HGRT_MAPS_DIR (default ~/.cache/hgrt/maps)."""
import os

import torch

from sparsepage.qwen import DEV, LinCache, Runner
from .caches import factory

PREFILLS = {
    "exact": None,
    "full": [(-1, None)],              # every layer, but through DepthSkip's HF path + handoff (checks the handoff)
    "cut24": [(-1, 24)],
    "cut16": [(-1, 16)],
    "cut16_tail32": [(-1, 16), (32, None)],
    "g512": [(-1, 16), (480, 24), (32, None)],
    "f64_g512": [(64, None), (-1, 16), (480, 24), (32, None)],
}
FIT_DATA = os.environ.get("HGRT_FIT_DATA", "/mnt/ext/mercurius/data/fineweb-edu/sample/10BT/000_00000.parquet")


def fit_texts(tok, n_calib=200, n_held=10, tokens=256):
    """calibration / held-out texts: FineWeb-Edu documents if HGRT_FIT_DATA exists, else WikiText-2 windows."""
    if os.path.exists(FIT_DATA):
        import pyarrow.parquet as pq
        docs = next(pq.ParquetFile(FIT_DATA).iter_batches(batch_size=2000, columns=["text"])).column("text").to_pylist()
    else:
        from sparsepage.qwen import prose_text
        t = prose_text()
        docs = [t[i:i + 4000] for i in range(0, 4000 * (n_calib + n_held), 4000)]
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"][:tokens]
    docs = [d for d in docs if len(enc(d)) >= tokens][: n_calib + n_held]
    return [enc(d) for d in docs[:n_calib]], [enc(d) for d in docs[n_calib:]]


class Pipeline:
    def __init__(self, model="Qwen/Qwen3.5-9B", maps_dir=None, prefill_chunk=4096):
        import depthskip
        self.R = Runner(model)
        self.tm = depthskip.Model.wrap(self.R.m, self.R.tok, prefill_chunk=prefill_chunk)
        self.tag = model.rstrip("/").split("/")[-1].lower()
        self.maps_dir = maps_dir or os.environ.get("HGRT_MAPS_DIR", os.path.expanduser("~/.cache/hgrt/maps"))
        self._maps = {}

    # ---------------------------------------------------------------------------------------------- DepthSkip maps
    def maps(self, cut):
        import depthskip
        if cut not in self._maps:
            p = os.path.join(self.maps_dir, f"{self.tag}_cut{cut}_normed.pt")
            if not os.path.exists(p):
                os.makedirs(self.maps_dir, exist_ok=True)
                calib, held = fit_texts(self.R.tok)
                m = depthskip.fit_skip_maps(self.tm, calib, held, cut, mode="normed")
                m.save(p)
            self._maps[cut] = depthskip.SkipMaps.load(p, DEV)
        return self._maps[cut]

    # ---------------------------------------------------------------------------------------------- prompt pass
    @torch.no_grad()
    def prefill(self, ids, plan="exact"):
        """ids (T,) long on the GPU -> snapshot for sparsepage.qwen.Runner.decode_teacher / generate."""
        spec = PREFILLS[plan] if isinstance(plan, str) else plan
        if spec is None:
            return self.R.prefill(ids)
        import depthskip
        segs = [(n, None if c is None else self.maps(c)) for n, c in spec]
        cache = depthskip.segment_prefill(self.tm, ids[:-1], segs)
        out = self.tm.lm(input_ids=ids[-1:][None], past_key_values=cache, use_cache=True)
        return self.handoff(out.past_key_values, len(ids), self.R.logits(out.last_hidden_state[0, -1:])[0])

    def handoff(self, cache, T, last_logits):
        kv, lin = {}, LinCache(self.R.lin_ids)
        for i in self.R.attn_ids:
            lay = cache.layers[i]
            kv[i] = (lay.keys[0].transpose(0, 1).to(torch.bfloat16), lay.values[0].transpose(0, 1).to(torch.bfloat16))
        for i in self.R.lin_ids:
            lay = cache.layers[i]
            lin.layers[i].conv_states[0] = lay.conv_states[0].clone()
            lin.layers[i].recurrent_states[0] = lay.recurrent_states[0].clone()
        assert all(k.shape[0] == T for k, _ in kv.values()), "cache length"
        return dict(kv=kv, lin=lin, T=T, last_logits=last_logits)

    # ---------------------------------------------------------------------------------------------- decode
    def perplexity(self, snap, targets, decode="stack10"):
        return self.R.decode_teacher(factory(decode), snap, targets)

    @torch.no_grad()
    def generate(self, prompt, max_new=64, plan="g512", decode="stack10"):
        ids = torch.tensor(self.R.tok(prompt)["input_ids"], device=DEV)
        snap = self.prefill(ids, plan)
        caches = self.R.build(factory(decode), snap)
        lin, pos, out = snap["lin"].clone(), snap["T"], []
        nxt = int(snap["last_logits"].argmax())
        for _ in range(max_new):
            out.append(nxt)
            if nxt == self.R.tok.eos_token_id:
                break
            h = self.R.forward(torch.tensor([nxt], device=DEV), pos, caches, lin, prefill=False)
            pos += 1
            nxt = int(self.R.logits(h[-1:])[0].argmax())
        return self.R.tok.decode(out)
