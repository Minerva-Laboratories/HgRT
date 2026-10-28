"""Tile choice for the NF4 GEMV per (N, K, M): times every (rows-per-warp, groups-in-flight) variant on the engine's
real weights, writes results/mercurius/gemv_rows.json (loaded by the engine) and prints GB/s."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'src'))
from cairn.mercurius_engine import HID, MercuriusEngine
from cairn.mercurius_kernels import ext

OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")
RES = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'results', 'mercurius')
VARIANTS = (21, 22, 41, 42, 81, 12, 14, 24)


def bench(fn, it=50, warm_s=0.5):
    """graph of `it` calls; replayed for >= warm_s first (the GPU and memory clocks are DVFS-governed and ramp up
    under load), then timed over 3 replays."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it):
            fn()
    t0 = time.time()
    while time.time() - t0 < warm_s:
        g.replay()
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return (time.time() - t) / (3 * it)


@torch.no_grad()
def main():
    E = MercuriusEngine(os.path.join(OUT, "weights.pt"), Tmax=4096)
    lins = {}
    d0, a0 = E.layers[E.lin_ids[0]], E.layers[E.attn_ids[0]]
    lins["mlp gate|up"] = d0["gu"]
    lins["mlp down"] = d0["down"]
    lins["gdn qkv|z"] = d0["qkvz"]
    lins["gdn out"] = d0["out"]
    for i in E.attn_ids:
        lins[f"mla q|down|krms r={E.layers[i]['r']}"] = E.layers[i]["qdk"]
    lins["mla o"] = a0["o"]
    lins["lm head"] = E.head
    from cairn.mercurius_mtp import MTPHead
    H = MTPHead(E, draft_vocab=torch.load(os.path.join(OUT, "draft_vocab.pt"))[:32768])
    lins["mtp fc"] = H.fc
    lins["mtp q|k|v"] = H.qkv
    lins["draft head (32k vocab)"] = H.head
    table, rows_out = [], {}
    X = ext()
    for name, L in lins.items():
        nbytes = L.codes.numel() * 4 + L.scales.numel() * 4
        K = L.K
        f32 = "head" in name
        def run_best(M, codes):
            xq = torch.randint(-127, 128, (M * K,), dtype=torch.int8, device="cuda")
            xd = torch.rand(M * K // 32, device="cuda") * 0.01
            y = torch.empty(M, L.N, dtype=torch.float32 if f32 else torch.bfloat16, device="cuda")
            best = None
            for v in codes:
                t = bench(lambda: X.nf4_gemv(L.codes, L.scales, xq, xd, None, 0, y, L.N, M, v))
                if best is None or t < best[1]:
                    best = (v, t)
            return best
        # the lane mapping (S) is fixed per matrix (bit identity across M); R / U free per M
        bs = {S: run_best(1, [v + (1000 if S == 2 else 0) for v in VARIANTS]) for S in (1, 2)}
        S = min(bs, key=lambda k: bs[k][1])
        fast = run_best(1, [v + 100 + (1000 if S == 2 else 0) for v in VARIANTS])
        for M in (1, 2, 3, 4):
            best = bs[S] if M == 1 else run_best(M, [v + (1000 if S == 2 else 0) for v in VARIANTS])
            rows_out[f"{L.N},{L.K},{M}"] = best[0]
            table.append({"name": name, "N": L.N, "K": L.K, "M": M, "code": best[0], "us": best[1] * 1e6,
                          "GBs": nbytes / best[1] / 1e9, "us_S1_M1": bs[1][1] * 1e6, "us_S2_M1": bs[2][1] * 1e6,
                          "us_approx_M1": fast[1] * 1e6})
            print(f"{name:28s} N={L.N:6d} K={L.K:5d} M={M}: best {best[0]:4d} {best[1]*1e6:8.1f} us "
                  f"{nbytes / best[1] / 1e9:6.1f} GB/s" + (f" | M=1 S=1 {bs[1][1]*1e6:.1f} us, S=2 {bs[2][1]*1e6:.1f} us, "
                  f"one-table approx {fast[1]*1e6:.1f} us ({nbytes / fast[1] / 1e9:.1f} GB/s)" if M == 1 else ""), flush=True)
    os.makedirs(RES, exist_ok=True)
    json.dump(rows_out, open(os.path.join(RES, "gemv_rows.json"), "w"), indent=1)
    json.dump(table, open(os.path.join(RES, "gemv_tune.json"), "w"), indent=1)
    print("TUNE_DONE")


if __name__ == "__main__":
    main()
