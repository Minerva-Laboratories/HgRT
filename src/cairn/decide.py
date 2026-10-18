"""OpenDecider on the stack: /v1/decide with the state pass through Terrace.

OpenDecider never generates text, so of the stack only the prompt side applies: the state (the long input every
question reads) is encoded once with a Terrace cut, its upper layers' caches written from ridge maps, and every question
then runs all layers on that cache. Maps are fitted on the decider's own backbone (terrace.fit_skip_maps accepts
OpenDecider's Backbone) from state texts: OpenDecider records (JSONL with a "state" field) rendered the way the decider
renders them, or plain texts. Measured on the 9B decider (Terrace README): cut 24 keeps decisions within the model's
own noise (the serving default here); cut 16 loses lettered multiple choice without an exact tail."""
import json
import os

import numpy as np


def state_texts(model, path, n, seed=0):
    from opendecider.formatting import record_state_text
    rows = [json.loads(l) for l in open(path) if l.strip()]
    idx = np.random.default_rng(seed).permutation(len(rows))[:n]
    fmt, cap = getattr(model, "row_format", "list"), model.cfg.v3_list_cap
    return [record_state_text({"state": rows[i]["state"], "questions": []}, fmt, cap) for i in idx]


def fit_decider_maps(ckpt, out, cut=24, calib_jsonl=None, held_jsonl=None, n_calib=200, n_held=10):
    """Fit Terrace maps on an OpenDecider checkpoint's backbone and save them to `out`."""
    import terrace
    from opendecider.checkpoint import load_decider
    dec = load_decider(ckpt)
    model, bb = dec.model, dec.model.backbone
    if calib_jsonl:
        calib = state_texts(model, calib_jsonl, n_calib, 0)
        held = state_texts(model, held_jsonl or calib_jsonl, n_held, 1)
    else:
        from .pipeline import fit_texts
        calib, held = fit_texts(bb.tok, n_calib, n_held)
    maps = terrace.fit_skip_maps(bb, calib, held, cut, mode="normed")
    maps.meta |= {"ckpt": ckpt, "calib": calib_jsonl or "fineweb-edu"}
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    maps.save(out)
    return out


def serve(ckpt, maps=None, host="127.0.0.1", port=8000):
    """OpenDecider's API server (opendecider.api) with the Terrace state pass when `maps` is given."""
    os.environ["OPENDECIDER_CKPT"] = ckpt
    if maps:
        os.environ["OPENDECIDER_SKIPKV"] = maps
    os.environ["HOST"], os.environ["PORT"] = host, str(port)
    from opendecider.api import main
    main()
