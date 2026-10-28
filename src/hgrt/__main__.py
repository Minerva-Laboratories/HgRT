"""python -m hgrt <command>

  generate   --prompt TEXT [--model Qwen/Qwen3.5-9B] [--plan g512] [--decode stack10] [--max-new 64]
  fit-maps   --model ... --cuts 16 24                        (DepthSkip maps for generation, cached in HGRT_MAPS_DIR)
  decide-fit --ckpt runs/<run>/model.pt --out maps.pt [--cut 24] [--calib states.jsonl]
  decide     --ckpt runs/<run>/model.pt [--maps maps.pt] [--host 127.0.0.1] [--port 8000]"""
import argparse


def main():
    ap = argparse.ArgumentParser(prog="hgrt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prompt", required=True)
    g.add_argument("--model", default="Qwen/Qwen3.5-9B")
    g.add_argument("--plan", default="g512")
    g.add_argument("--decode", default="stack10")
    g.add_argument("--max-new", type=int, default=64)
    f = sub.add_parser("fit-maps")
    f.add_argument("--model", default="Qwen/Qwen3.5-9B")
    f.add_argument("--cuts", type=int, nargs="+", default=[16, 24])
    df = sub.add_parser("decide-fit")
    df.add_argument("--ckpt", required=True)
    df.add_argument("--out", required=True)
    df.add_argument("--cut", type=int, default=24)
    df.add_argument("--calib", default=None, help="OpenDecider records (JSONL with 'state'); default FineWeb-Edu text")
    d = sub.add_parser("decide")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--maps", default=None)
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.cmd == "generate":
        from .pipeline import Pipeline
        print(Pipeline(a.model).generate(a.prompt, a.max_new, a.plan, a.decode))
    elif a.cmd == "fit-maps":
        from .pipeline import Pipeline
        p = Pipeline(a.model)
        for c in a.cuts:
            p.maps(c)
            print(f"cut {c}: {p.maps_dir}")
    elif a.cmd == "decide-fit":
        from .decide import fit_decider_maps
        print(fit_decider_maps(a.ckpt, a.out, a.cut, a.calib))
    else:
        from .decide import serve
        serve(a.ckpt, a.maps, a.host, a.port)


if __name__ == "__main__":
    main()
