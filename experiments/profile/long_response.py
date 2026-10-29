"""Where decode time goes on LONG RESPONSES (user report 2026-10-28: the slowdown shows most with long generations,
not long prompts / contexts). Synthetic caches (BENCH_HKV KV heads, D=256), one attention layer x LAYERS:
  stack10 graph path: per-token graph replay (score + select + page pass + window + combine) vs the eager
           pre_step() block encode that runs every 128 generated tokens (FoveaKV encode + SparsePage summaries);
  dyn10 (FoveaKV dynamic 2+2): eager attend() per token (coarse pass + top-k + delta pass), no graph path.
For prompts P and generated lengths, reports ms per generated token split into parts (CUDA events, median)."""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from hgrt.caches import make_cache                 # noqa: E402
from hgrt.stack import BLK, D, G                   # noqa: E402

HKV = int(os.environ.get("BENCH_HKV", 4))
LAYERS = int(os.environ.get("BENCH_LAYERS", 8))
torch.manual_seed(0)


def kv(T):
    return (torch.randn(T, HKV, D, device="cuda") * 0.8).to(torch.bfloat16), torch.randn(T, HKV, D, device="cuda").to(torch.bfloat16)


def ev():
    return torch.cuda.Event(enable_timing=True)


def stack_run(P, gen, Tmax):
    caches = [make_cache("stack10", HKV, Tmax) for _ in range(LAYERS)]
    k, v = kv(P)
    for c in caches:
        c.prefill(k, v)
        c.graph_setup()
    q = torch.randn(HKV * G, D, device="cuda")
    kn, vn = kv(1)
    pos = torch.full((1,), P, dtype=torch.long, device="cuda")

    def step():
        for c in caches:
            c.graph_attend(q, kn[0][None], vn[0][None], pos)
    for c in caches:                                         # warm-up + capture (as hgrt.engine does)
        c.pre_step()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    t_graph, t_enc, t_host, is_enc = [], [], [], []
    for i in range(gen):
        nc0 = caches[0].n_comp
        h0 = time.perf_counter()
        a = ev(); a.record()
        for c in caches:
            c.pre_step()
        b = ev(); b.record()
        pos.fill_(P + 1 + i)
        g.replay()
        e = ev(); e.record()
        torch.cuda.synchronize()
        t_host.append(1000 * (time.perf_counter() - h0))
        t_enc.append(a.elapsed_time(b)); t_graph.append(b.elapsed_time(e)); is_enc.append(caches[0].n_comp != nc0)
    t_enc, t_graph, t_host = map(np.array, (t_enc, t_graph, t_host))
    encode_steps = np.array(is_enc)
    return {"scheme": "stack10" + ("" if caches[0].encode_graph else " (eager encode)"), "prompt": P, "generated": gen,
            "graph_ms_median": float(np.median(t_graph)),
            "graph_ms_first_256": float(np.median(t_graph[:256])), "graph_ms_last_256": float(np.median(t_graph[-256:])),
            "encode_ms_per_encode_step": float(np.median(t_enc[encode_steps])) if encode_steps.any() else 0.0,
            "encode_steps": int(encode_steps.sum()),
            "encode_ms_amortized_per_token": float(t_enc.sum() / gen),
            "wall_ms_per_token": float(t_host.mean()),
            "wall_ms_per_token_on_encode_steps": float(t_host[encode_steps].mean()) if encode_steps.any() else 0.0}


def dyn_run(P, gen, Tmax):
    caches = [make_cache("dyn10", HKV, Tmax) for _ in range(LAYERS)]
    k, v = kv(P)
    for c in caches:
        c.prefill(k, v)
    q = torch.randn(HKV * G, D, device="cuda")
    kn, vn = kv(1)
    t_att, t_app, t_host = [], [], []
    for i in range(gen):
        h0 = time.perf_counter()
        a, b, e = ev(), ev(), ev()
        a.record()
        for c in caches:
            c.append(kn, vn)
        b.record()
        for c in caches:
            c.attend(q)
        e.record(); torch.cuda.synchronize()
        t_host.append(1000 * (time.perf_counter() - h0))
        t_app.append(a.elapsed_time(b)); t_att.append(b.elapsed_time(e))
    t_att, t_app, t_host = map(np.array, (t_att, t_app, t_host))
    return {"scheme": "dyn10", "prompt": P, "generated": gen,
            "attend_ms_first_256": float(np.median(t_att[:256])), "attend_ms_last_256": float(np.median(t_att[-256:])),
            "append_encode_ms_amortized": float(t_app.sum() / gen),
            "wall_ms_per_token": float(t_host.mean()), "gpu_ms_per_token": float((t_att + t_app).mean())}


def dyn_graph_run(P, gen, Tmax):
    """dyn10 through hgrt.dyn.DynGraph: one captured graph for the LAYERS calls, recaptured when the bucket grows."""
    from hgrt.dyn import DynGraph
    caches = [DynGraph(HKV, Tmax, k_frac=0.1) for _ in range(LAYERS)]
    k, v = kv(P)
    for c in caches:
        c.prefill(k, v)
        c.graph_setup()
    q = torch.randn(HKV * G, D, device="cuda")
    kn, vn = kv(1)
    pos = torch.full((1,), P, dtype=torch.long, device="cuda")
    graph, t_step, t_host, recaps = None, [], [], 0
    for i in range(gen):
        h0 = time.perf_counter()
        a = ev(); a.record()
        for c in caches:
            c.pre_step()
        pos.fill_(P + i)
        if graph is None or any(c.recapture for c in caches):
            for c in caches:
                c.recapture = False
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for c in caches:
                    c.graph_attend(q, kn[0][None], vn[0][None], pos)
            recaps += 1
        graph.replay()
        e = ev(); e.record(); torch.cuda.synchronize()
        t_host.append(1000 * (time.perf_counter() - h0)); t_step.append(a.elapsed_time(e))
    t_step, t_host = np.array(t_step), np.array(t_host)
    return {"scheme": f"dyn10 graph ({caches[0].kernels} kernels)", "prompt": P, "generated": gen, "step_ms_median": float(np.median(t_step)),
            "step_ms_last_256": float(np.median(t_step[-256:])), "wall_ms_per_token": float(t_host.mean()),
            "captures": recaps}


if __name__ == "__main__":
    out = []
    gen = int(os.environ.get("BENCH_GEN", 4096))
    for P in [int(x) for x in os.environ.get("BENCH_PROMPTS", "1024,32768").split(",")]:
        sel = os.environ.get("BENCH_RUNS", "stack,dyn,dyngraph").split(",")
        for fn in [f for n_, f in (("stack", stack_run), ("dyn", dyn_run), ("dyngraph", dyn_graph_run)) if n_ in sel]:
            r = fn(P, gen, P + gen + 2 * BLK)
            print(json.dumps(r), flush=True)
            out.append(r)
            torch.cuda.empty_cache()
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "results", "profile"), exist_ok=True)
    tag = os.environ.get("BENCH_OUT", "long_response")                 # one file per run set (no overwrites)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "profile", f"{tag}.json"), "w"), indent=1)
