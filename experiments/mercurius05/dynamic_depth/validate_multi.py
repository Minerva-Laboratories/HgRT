"""Accuracy vs the pipeline on several prompts. ref: greedy continuation by the pipeline's FoveaKV forward, then
teacher-forced logits of the FoveaKV (PyTorch tier path) and exact references (fp16, generated positions only).
eng [--prefill]: the engine teacher-forced on the same tokens, per prompt and both caches: top-1 agreement and KL."""
import glob, os, sys, torch
CK = os.path.expanduser("~/Desktop/Models/Mercurius-0.5-4B-Preview")
D = os.path.dirname(os.path.abspath(__file__))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(CK)
chat = lambda s: tok(tok.apply_chat_template([{"role": "user", "content": s}], add_generation_prompt=True, tokenize=False,
                                            enable_thinking=False)).input_ids
src = "".join(open(f).read() for f in sorted(glob.glob(os.path.expanduser(
    "~/Desktop/release-backup-2026-11-18/*.md"))))
PROMPTS = {
    "es": chat("Explica en español cómo funciona la fotosíntesis, en cinco pasos breves."),
    "code": chat("Write a Python function that parses ISO-8601 durations like P3DT4H5M into seconds, with a few tests."),
    "json": chat("Return a JSON object describing three fictional employees: name, role, skills (list), start_date."),
    "math": chat("A train leaves at 3:15 pm at 72 km/h; a second leaves the same station at 4:00 pm at 90 km/h on the same "
                 "track. When does the second catch up? Solve step by step."),
    "doc": tok(src).input_ids[:1400] + chat("List the open items in the notes above."),
}
NGEN = 160
mode = sys.argv[1]
if mode == "ref":
    sys.path.insert(0, D)
    from reference import load_reference
    import mercurius.arch.attention as A
    from mercurius.quant.kv import set_kv_split
    m = load_reference(CK, kv=True)
    R = {}
    with torch.no_grad():
        for name, ids in PROMPTS.items():
            full = m.generate(torch.tensor([ids], device="cuda"), max_new_tokens=NGEN, do_sample=False)[0].tolist()
            A.USE_TRITON = False
            set_kv_split(m, len(ids))
            lg = m(torch.tensor([full], device="cuda")).logits[0, len(ids) - 1:-1].half().cpu()
            A.USE_TRITON = True
            R[name] = {"ids": full, "P": len(ids), "fovea": lg}
            print(name, len(ids), len(full) - len(ids), repr(tok.decode(full[len(ids):len(ids) + 40])), flush=True)
    torch.save(R, os.path.join(D, "ref_multi.pt"))
    print("saved fovea")
elif mode == "refexact":                                   # separate process: two references do not fit in 6 GB
    sys.path.insert(0, D)
    from reference import load_reference
    R = torch.load(os.path.join(D, "ref_multi.pt"))
    m = load_reference(CK, kv=False)
    with torch.no_grad():
        for name, r in R.items():
            r["exact"] = m(torch.tensor([r["ids"]], device="cuda")).logits[0, r["P"] - 1:-1].half().cpu()
    torch.save(R, os.path.join(D, "ref_multi.pt"))
    print("saved")
else:
    sys.path.insert(0, os.environ["HGRT_SRC"])
    from hgrt.mercurius_packed import PackedMercuriusEngine
    R = torch.load(os.path.join(D, "ref_multi.pt"))
    pfl = "--prefill" in sys.argv
    for kv, key in (("foveakv", "fovea"), ("exact", "exact")):
        e = PackedMercuriusEngine(CK, Tmax=max(len(r["ids"]) for r in R.values()) + 64, kv=kv)
        e.capture()
        ta, tn, tk = 0, 0, []
        for name, r in R.items():
            full, P, ref = r["ids"], r["P"], r[key]
            e.reset()
            if kv == "foveakv":
                e.ensure_prefill(P)
            if pfl:
                e.prefill(list(full[:P - 1]))
            agree, kls = 0, []
            for t, x in enumerate(full[:-1]):
                if pfl and t < P - 1:
                    continue
                if t == P:
                    e.calibrate_fovea()
                e._run([x]); e.pos.add_(1)
                if t >= P - 1:
                    lg, rr = e.logits[0].float(), ref[t - (P - 1)].cuda().float()
                    agree += int(lg.argmax() == rr.argmax())
                    kls.append(torch.nn.functional.kl_div(lg.log_softmax(-1), rr.log_softmax(-1), log_target=True,
                                                          reduction="sum").item())
            n = len(kls); ta += agree; tn += n; tk += kls
            print(f"{'prefill ' if pfl else ''}kv={kv:8s} {name:5s} top-1 {agree}/{n} KL mean {sum(kls)/n:.2e}", flush=True)
        print(f"{'prefill ' if pfl else ''}kv={kv:8s} ALL   top-1 {ta}/{tn} ({100 * ta / tn:.1f} %) KL mean {sum(tk)/tn:.2e}")
        del e; torch.cuda.empty_cache()
