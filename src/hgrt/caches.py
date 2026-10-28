"""Cache factory by name: stack* (hgrt.stack), sparsepage* (SparsePage on bf16 pages), anything else a FoveaKV scheme
(bf16, bf16-kernel, gs2, gs4, dyn5, dyn10, quest5, kivi2, ...; see FoveaKV gpu/foveakv_gpu.make)."""
from .deps import foveakv_gpu


def make_cache(name, Hkv, Tmax, device="cuda"):
    if name.startswith("stack"):
        from .stack import make_stack
        return make_stack(name, Hkv, Tmax, device)
    if name.startswith("sparsepage"):
        from sparsepage import SparsePageCache
        base, _, var = name.partition("-")
        cov, bound = {"": (True, True), "mean": (False, False), "bound": (False, True)}[var]
        return SparsePageCache(Hkv, Tmax, head_dim=foveakv_gpu().D, rho=int(base[6:]) / 100, cov=cov, bound=bound, device=device)
    return foveakv_gpu().make(name, Hkv, Tmax)


def factory(name):
    return lambda Hkv, Tmax: make_cache(name, Hkv, Tmax)
