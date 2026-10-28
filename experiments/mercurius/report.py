"""Collect the Mercurius benchmark JSONs into results/mercurius/summary.json and print the markdown tables."""
import json
import os
import sys

RES = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'results', 'mercurius')
OUT = os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius")


def load(p):
    p = os.path.join(RES, p) if not p.startswith("/") else p
    return json.load(open(p)) if os.path.exists(p) else None


def llama_results(path=os.path.join(OUT, "llama_bench.log")):
    """parse llama-bench -o json blocks: {model: {depth: tok/s}}"""
    if not os.path.exists(path):
        return {}
    txt = open(path).read()
    out, cur = {}, None
    for block in txt.split("== ")[1:]:
        name = block.split("\n", 1)[0].strip()
        body = block.split("\n", 1)[1]
        s, e = body.find("["), body.rfind("]")
        if s < 0 or e < 0:
            continue
        try:
            rows = json.loads(body[s:e + 1])
        except json.JSONDecodeError:
            continue
        out[name] = {r.get("n_depth", 0): r.get("avg_ts") for r in rows if r.get("n_gen", 0) > 0}
    return out


def main():
    bench = load(sys.argv[1] if len(sys.argv) > 1 else "bench_final.json") or {}
    ref = load(os.path.join(OUT, "ref_timing.json")) or {}
    llama = llama_results()
    rows = []
    print("| context | T | transformers (deployed) | llama-mercurius NF4 | HgRT plain | HgRT+MTP k=1 | k=2 | k=3 |")
    print("|---|---|---|---|---|---|---|---|")
    depth_of = {"long1024": 1024, "long8192": 8192, "long32768": 32768}
    for c, r in bench.get("ctx", {}).items():
        tr = ref.get(c, {}).get("ms_per_token")
        lm = llama.get("mercurius-1-4b-NF4.gguf", {}).get(depth_of.get(c, -1))
        cells = [c, str(r["T"]), f"{1e3 / tr:.2f}" if tr else "-", f"{lm:.1f}" if lm else "-", f"{r['plain']['tok_s']:.1f}"]
        for k in (1, 2, 3):
            m = r.get(f"mtp_k{k}")
            cells.append(f"{m['tok_s']:.1f} (tau {m['tau']:.2f}{'' if m['lossless'] else ', NOT lossless'})" if m else "-")
        print("| " + " | ".join(cells) + " |")
        rows.append(cells)
    json.dump({"bench": bench, "reference": ref, "llama": llama}, open(os.path.join(RES, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
