"""The fused stack decode path (cairn.stack.StackCache.attend_kernel) against its PyTorch reference and against the other
kernels: correctness (page scores, selected pages, output) and latency per attention call, eager and inside a CUDA
graph (8 calls = the 9B's 8 attention layers, so launch overhead is what a graph-captured decode step pays).
Synthetic caches with drifting topic means (batch 1, G=4, D=256). Env BENCH_HKV (default 4 = Qwen3.5-9B),
BENCH_CTX (default 32768,131072,262144,1048576), BENCH_SCHEMES, BENCH_TAG (output suffix).
Writes results/gpu_stack_bench_h<HKV><TAG>.json."""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cairn.caches import make_cache as make     # noqa: E402
from cairn.stack import BLK, D, G               # noqa: E402

torch.manual_seed(0)
HKV = int(os.environ.get("BENCH_HKV", 4))
CTX = [int(x) for x in os.environ.get("BENCH_CTX", "32768,131072,262144,1048576").split(",")]
SCHEMES = os.environ.get("BENCH_SCHEMES", "bf16-kernel,gs2,dyn10,stack10").split(",")
N = 30
out = []


def synth(T):
    """keys: topic means drifting every 512 tokens + noise; a needle region aligned with the query; values Gaussian."""
    nt = -(-T // 512)
    K = torch.empty(T, HKV, D, device="cuda", dtype=torch.bfloat16)
    for a in range(0, nt, 256):                                   # bounded temporaries
        b = min(nt, a + 256)
        mu = torch.randn(b - a, HKV, D, device="cuda") * 0.8
        x = mu.repeat_interleave(512, 0)[: min(T, b * 512) - a * 512] + torch.randn(min(T, b * 512) - a * 512, HKV, D, device="cuda") * 0.6
        K[a * 512:a * 512 + len(x)] = x.to(torch.bfloat16)
    V = torch.randn(T, HKV, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(HKV * G, D, device="cuda") * 1.5
    s = T // 3
    K[s:s + 100] += (0.25 * q.view(HKV, G, D).mean(1)[None]).to(torch.bfloat16)
    return K, V, q


def timed(f):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    ts = []
    for _ in range(N):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); f(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return float(np.median(ts))


def graph_ms(f, calls=8):
    """median ms of one replay of a graph holding `calls` calls of f (None if capture fails)."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                f()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(calls):
                f()
        return timed(g.replay)
    except Exception as ex:                                       # host sync / unsupported op in the scheme
        print("   graph capture failed:", str(ex).splitlines()[0][:120], flush=True)
        torch.cuda.synchronize()
        return None


for T in CTX:
    K, V, q = synth(T)
    for nm in SCHEMES:
        c = make(nm, HKV, T + 256)
        c.prefill(K, V)
        row = dict(T=T, Hkv=HKV, key=nm, scheme=c.name, read_bits=c.read_bits())
        if nm.startswith("stack"):
            # correctness: scores, selection, output vs the PyTorch reference
            sc_ref = c.page_scores(q)
            o_ref = c.attend_ref(q).float()
            sel_ref = c.last_sel
            o_k = c.attend_kernel(q).float()
            npg = c.n_comp // BLK
            sc_k = c._scores[:, :npg]
            fin = torch.isfinite(sc_ref)
            row["score_rel_diff"] = float((sc_k[fin] - sc_ref[fin]).norm() / sc_ref[fin].norm())
            m = c.n_open()
            jac = []
            for h in range(HKV):
                a_, b_ = set(sel_ref[h].tolist()), set(c._sel[h, :m].tolist())
                jac.append(len(a_ & b_) / len(a_ | b_))
            row["sel_jaccard"] = float(np.mean(jac))
            row["out_rel_diff_vs_ref"] = float((o_k - o_ref).norm() / o_ref.norm())
            f = lambda: c.attend_kernel(q)
        else:
            f = lambda: c.attend(q)
        row["ms_eager"] = timed(f)
        g8 = graph_ms(f)
        row["ms_graph_per_call"] = None if g8 is None else g8 / 8
        out.append(row)
        extra = (f"  scores {row['score_rel_diff']:.1e} sel J {row['sel_jaccard']:.3f} out {row['out_rel_diff_vs_ref']:.1e}"
                 if nm.startswith("stack") else "")
        gtxt = "  n/a " if g8 is None else f"{g8 / 8 * 1e3:7.1f}"
        print(f"T={T:8d} Hkv={HKV} {c.name:28s} eager {row['ms_eager'] * 1e3:8.1f} us  graph/call {gtxt} us"
              f"  bits/step {row['read_bits']:.2f}{extra}", flush=True)
        del c
        torch.cuda.empty_cache()
    del K, V
    torch.cuda.empty_cache()
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", f"stack_bench_h{HKV}{os.environ.get('BENCH_TAG', '')}.json"), "w"), indent=1)
print("STACK_BENCH_DONE")
