"""Locate the components. SparsePage, DepthSkip and OpenDecider are installed packages (pip install -e third_party/<name>);
FoveaKV's GPU caches are a module of its repository (third_party/foveakv/gpu), put on sys.path here.
HGRT_FOVEAKV overrides the FoveaKV checkout."""
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def foveakv_root():
    return os.environ.get("HGRT_FOVEAKV", os.path.join(ROOT, "third_party", "foveakv"))


def foveakv_gpu():
    r = foveakv_root()
    if not os.path.exists(os.path.join(r, "gpu", "foveakv_gpu.py")):
        raise ImportError(f"FoveaKV not found at {r}: git submodule update --init (or set HGRT_FOVEAKV)")
    if r not in sys.path:
        sys.path.insert(0, r)
    return importlib.import_module("gpu.foveakv_gpu")
