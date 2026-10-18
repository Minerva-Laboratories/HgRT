#!/bin/bash
# One checkout, every component: submodules at the pinned commits, installed editable into the current Python env.
# Needs torch with CUDA and nvcc matching torch's CUDA major (kernels compile on first use for the local GPU).
set -e
cd "$(dirname "$0")/.."
git submodule update --init --recursive
pip install -e third_party/riffle -e third_party/terrace -e third_party/opendecider -e ".[test]"
pip install -r third_party/foveakv/requirements-gpu.txt
python -m pytest -q -m "not gpu"
