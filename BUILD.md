# Building the NUMA-Torch stack

Three components: stock LLVM 19, the NUMA-aware Triton-CPU backend, and the
PyTorch fork carrying the Inductor NUMA-TP passes. Build in that order.

> The in-tree notes inside the forks (`LOCAL_SETUP.md`, `Advisor_setup.md`, the
> `bench_2026-09-25/*.sh` drivers) record absolute paths from the original
> benchmark machine (`/home/shavin/...`). They are kept as provenance. Use the
> instructions here instead and set `$NUMA_ROOT` to wherever you are working.

    export NUMA_ROOT=$HOME/numa-torch-build
    mkdir -p "$NUMA_ROOT" && cd "$NUMA_ROOT"

## Requirements

A multi-socket x86-64 machine with NUMA enabled. These results cannot be
reproduced in a container or on a single-socket host.

    sudo apt install -y build-essential cmake ninja-build python3-dev \
                        libnuma-dev libopenblas-dev git

Check your topology before starting:

    numactl --hardware        # must report 2 or 4 nodes

## 1. LLVM 19 (unmodified upstream)

Pinned to `adba14acea99cc6a17d837763a3248c9d4a2fadf`. This is stock upstream;
there is no NUMA-Torch patch to apply here.

    git clone https://github.com/llvm/llvm-project "$NUMA_ROOT/llvm-project"
    cd "$NUMA_ROOT/llvm-project"
    git checkout adba14acea99cc6a17d837763a3248c9d4a2fadf
    mkdir build && cd build
    cmake -G Ninja ../llvm \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm" \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON
    ninja

    export LLVM_SYSPATH="$NUMA_ROOT/llvm-project/build"
    export LLVM_INCLUDE_DIRS="$LLVM_SYSPATH/include"
    export LLVM_LIBRARY_DIR="$LLVM_SYSPATH/lib"

## 2. Triton-CPU with the NUMA launcher

    git clone https://github.com/LLM-Compiler/triton-cpu "$NUMA_ROOT/triton-cpu"
    cd "$NUMA_ROOT/triton-cpu"
    git checkout v1.0-paper
    pip install -e python

The NUMA work is confined to `third_party/cpu/backend/driver.py`
(`get_numa_node_cpus`, `pthread_setaffinity_np`, `NumaThreadPool`,
`get_or_init_numa_pool`, and the `numa_args` dispatch path).

## 3. PyTorch with the Inductor NUMA-TP passes

One revision covers every platform: `v1.0-paper-four-socket`. It is a strict
superset of the earlier `v1.0-paper-dual-socket` (0 commits behind) and adds the
row-parallel all-reduce correctness fix, so build it for 2-socket and 4-socket
hosts alike.

    git clone https://github.com/LLM-Compiler/pytorch "$NUMA_ROOT/pytorch"
    cd "$NUMA_ROOT/pytorch"
    git checkout v1.0-paper-four-socket
    pip install -r requirements.txt

    export USE_NUMA=1
    export MAX_JOBS=16
    export BLAS=OpenBLAS
    export USE_MKLDNN=0
    export USE_MKL=0
    python setup.py develop

`USE_NUMA=1` is required; the extended `clone` in
`aten/src/ATen/native/TensorFactories.cpp` and the `mbind`/`MPOL_BIND` path in
`c10/util/numa.cpp` are compiled out without it.

## 4. Verify

    python -c "
    import torch
    from torch._inductor import config
    print('torch', torch.__version__)
    print('numa_tp', config._numa_attention_tp_slice_kv)
    "

Then compile a model with the NUMA-TP passes enabled and confirm the Triton-CPU
launcher takes the NUMA path rather than the fallback.

## Reference configuration

| | Dual-socket | Four-socket |
|---|---|---|
| OS | Ubuntu 20.04 LTS | Ubuntu 24.04 LTS |
| Kernel | 5.15 | 7.0 |
| Python | 3.8 | 3.9 |
| PyTorch | 2.6.0a0 | 2.6.0a0 |
| LLVM | 19 (`adba14a`) | 19 (`adba14a`) |
| BLAS | OpenBLAS, MKL/MKLDNN off | OpenBLAS, MKL/MKLDNN off |

Four-socket host: 4 NUMA nodes, 80 logical CPUs (20 per node, 10 physical
cores per node), 62 GB usable.
