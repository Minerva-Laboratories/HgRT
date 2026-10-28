"""GDN-2 gate kernels: CUDA-core path vs tensor-core (staged) path, M = 1..4, plus correctness / M-invariance."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'src')); sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'), 'experiments', 'mercurius'))
from cairn.mercurius_engine import HID, VALD, MercuriusEngine
from cairn.mercurius_kernels import ext
from gemv_tune import bench

@torch.no_grad()
def main():
    E = MercuriusEngine(os.path.join(os.environ.get("HGRT_MERCURIUS_DIR", "/mnt/ext/hgrt-kernels/mercurius"), "weights.pt"), Tmax=4096)
    X, d = ext(), E.layers[0]
    x16 = torch.randn(16, HID, device="cuda").to(torch.bfloat16)
    for M in (1, 2, 3, 4):
        sg = torch.empty(M, E.NSG, device="cuda"); sg2 = torch.empty_like(sg)
        gd, be, bw = (torch.empty(M, VALD, device="cuda") for _ in range(3))
        t_old = bench(lambda: (X.dgemv(d["Ws"], HID, d["Ws"].shape[0], HID, x16, HID, sg, E.NSG, M, True),
                               X.gdn2_gates(sg, E.NSG, 32, E.rank, d["lr"], d["B"], d["Bl"], d["vd"], d["vb"], d["A_log"], d["dt_bias"], gd, be, bw, M, 128)))
        t_a = bench(lambda: X.tc_skinny(d["Ws"], HID, d["Ws"].shape[0], HID, x16, HID, sg2, E.NSG, M, True))
        t_tc = bench(lambda: (X.tc_skinny(d["Ws"], HID, d["Ws"].shape[0], HID, x16, HID, sg2, E.NSG, M, True),
                              X.gdn2_gates_tc(sg2, E.NSG, 32, E.rank, d["lr"], d["B"], d["Bl"], d["vd"], d["vb"], d["A_log"], d["dt_bias"], E.W2, E.T, gd, be, bw, M, 128)))
        ref = (x16[:M].float() @ d["Ws"].float().T).to(torch.bfloat16).float()
        err = float((sg2 - ref).norm() / ref.norm())
        print(f"M={M}: CUDA-core gates {t_old*1e3:.3f} ms | tensor cores: A-part {t_a*1e3:.3f} ms, total {t_tc*1e3:.3f} ms | "
              f"A-part rel err {err:.1e} | bytes A+B {(d['Ws'].numel() + d['B'].numel()) * 2 / 1e6:.1f} MB -> "
              f"{(d['Ws'].numel() + d['B'].numel()) * 2 / t_tc / 1e9:.0f} GB/s", flush=True)
    print("GATES_DONE")

if __name__ == "__main__":
    main()
