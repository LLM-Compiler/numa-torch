# NUMA-Torch

Artifact for **"NUMA-Torch: Compiler-Driven NUMA-Aware LLM Prefill on Multi-Socket CPUs."**

NUMA-Torch makes the PyTorch Inductor compilation path NUMA-aware. It treats each
NUMA socket as a tensor-parallel partition (NUMA-TP), shards weights across sockets
via post-grad FX passes, and dispatches the generated Triton-CPU kernels through a
persistent, socket-local thread pool. Prefill speedups reach 2.63x on dual-socket
Haswell, 1.87x on dual-socket Cascade Lake, and 2.27x on four-socket Haswell-EX.

---

## Components

One revision per component. The PyTorch side is `v1.0-paper-four-socket`
(`84cb280a`), which supersedes the earlier `v1.0-paper-dual-socket` (`54581fbd`):
it is a strict superset, 0 commits behind, and adds the row-parallel all-reduce
correctness fix. Build this revision for every platform.

| Component | Repository | Ref | Modified? |
|---|---|---|---|
| PyTorch (Inductor NUMA-TP passes) | `LLM-Compiler/pytorch` | `v1.0-paper-four-socket` = `84cb280a` | yes |
| Triton-CPU (NUMA launcher) | `LLM-Compiler/triton-cpu` | `v1.0-paper` = `7e394470` | yes |
| LLVM 19 | `llvm/llvm-project` | `adba14acea99cc6a17d837763a3248c9d4a2fadf` | **no** - stock upstream |

LLVM is unmodified. We verified that the commit used for all reported builds is
byte-identical to upstream `adba14a`; it is pinned, not forked.

---

## Getting the modified source

Three paths, cheapest first. Pick by what you actually want.

### 1. Read the contribution only (no clone)

`patches/` holds one diff per modified component, taken against that component's
upstream base. This is the fastest way to see exactly what NUMA-Torch changes,
and it is what we recommend for reviewers.

| Patch | Scope |
|---|---|
| `patches/pytorch-numa-tp.patch` | 36 source files, +5996/-49 under `torch/`, `aten/` |
| `patches/triton-cpu-numa.patch` | 6 files, +822/-18 (`driver.py` +512) |

Each applies to its upstream base with either `git apply` or `patch -p1`.

### 2. Browse the full modified tree (no clone)

`third_party/pytorch` and `third_party/triton-cpu` are submodules pinned to the
exact paper revisions. On GitHub, click either entry in the file list above and
you land on that commit inside the fork, with the whole modified tree browsable
in the web UI. Useful when you want the surrounding code, not just the diff:

- `third_party/pytorch` -> `torch/_inductor/fx_passes/numa_*.py`, `torch/_inductor/codegen/wrapper.py`
- `third_party/triton-cpu` -> `third_party/cpu/backend/driver.py`

### 3. Clone to build

**Do not use `git clone --recurse-submodules`.** PyTorch and LLVM carry very
large histories, and recursing pulls several GB you do not need. Clone the
umbrella repo plainly, then fetch each component shallow at its pinned ref:

    git clone https://github.com/LLM-Compiler/numa-torch

    # Triton-CPU at the paper revision (~6 MB)
    git clone --depth 1 --branch v1.0-paper \
      https://github.com/LLM-Compiler/triton-cpu

    # PyTorch at the paper revision
    git clone --depth 1 --branch v1.0-paper-four-socket \
      https://github.com/LLM-Compiler/pytorch

    # LLVM 19 is pinned to a plain commit, so fetch blobless then check out
    git clone --filter=blob:none https://github.com/llvm/llvm-project
    cd llvm-project && git checkout adba14acea99cc6a17d837763a3248c9d4a2fadf

If you do want the submodules materialised in place, initialise them one at a
time rather than all at once:

    git submodule update --init third_party/triton-cpu

Full build instructions, including the required `USE_NUMA=1`, are in `BUILD.md`.

---

## Measurement provenance

The artifact ships one PyTorch revision, but the published figures were not all
measured on it. This table records what produced each number, so the data can be
audited:

| Paper item | Platform | Measured on | Supported build |
|---|---|---|---|
| Fig. 2 (roofline) | Dual-socket Haswell | `54581fbd` | `v1.0-paper-four-socket` |
| Fig. 7 (time heatmaps) | Dual-socket | `54581fbd` | `v1.0-paper-four-socket` |
| Fig. 8 (speedup bars) | Dual-socket | `54581fbd` | `v1.0-paper-four-socket` |
| Fig. 9 (four-socket geomean) | Four-socket Haswell-EX | `84cb280a` | `v1.0-paper-four-socket` |
| Fig. 10 (four-socket grid) | Four-socket Haswell-EX | `84cb280a` | `v1.0-paper-four-socket` |
| `numactl` policy comparison | Dual-socket | n/a | this repo, `results/numactl/` |

`54581fbd` predates the row-parallel all-reduce fix, which changes numerical
output. It stays reachable in the fork history as the `v1.0-paper-dual-socket`
tag for anyone reproducing the published dual-socket figures exactly as
measured. New runs on any socket count should use `v1.0-paper-four-socket`.

---

## What we actually changed

PyTorch side, all under `torch/_inductor/`:

    fx_passes/numa_mlp_tp.py              1229    column/row-parallel MLP
    fx_passes/numa_attention_tp.py        1228    multi-head attention
    fx_passes/numa_gemma_attention_tp.py   932    MQA + GeGLU (Gemma 3)
    fx_passes/numa_tp_common.py            449    shared partition machinery
    fx_passes/numa_residual_tp.py          198
    fx_passes/numa_lm_head_tp.py           173
    fx_passes/numa_layer_norm_tp.py        158
    post_grad.py                           +43    pass registration
    runtime/triton_heuristics.py           +50    numa_args -> kernel metadata
    codegen/wrapper.py                    +1054   per-socket buffer codegen,
                                                  cross-socket all-reduce
    kernel/mm_common.py, kernel/bmm.py            NUMA-aware output allocation

Plus the extended `clone(..., numa_node=i)` operator under `aten/`, which binds
a tensor's pages to a chosen socket via `mbind`/`MPOL_BIND`.

Triton-CPU side, `third_party/cpu/backend/driver.py` (+512):

    get_numa_node_cpus()          Alg. 4 line 6   OS topology query
    pthread_setaffinity_np()      Alg. 4 line 8   bind worker to core in C_i
    NumaThreadPool                                persistent socket-local pool
    get_or_init_numa_pool()                       process-lifetime init
    launched_numa / numa_args                     Alg. 4 lines 11-12 + fallback

The two sides are coupled by one contract: Inductor emits `numa_args` on kernel
calls and attaches it to kernel metadata; the Triton-CPU launcher reads
`PyObject_HasAttrString(kernel_metadata, "numa_args")`. That attribute name
appears on both sides and nowhere else.

---

## Base revisions (for diffing against upstream)

| Fork | Base commit | In upstream? |
|---|---|---|
| pytorch | `2c1a9b7143eacdd25028c5e1cbabd57c52e65939` | **No** - unmerged ghstack branch, "Add Triton CPU as an Inductor backend" |
| triton-cpu | `e60f448f8f197073b75d6d3e77347414a5db3ee7` | Yes |

The PyTorch fork is based on an upstream pull-request branch that was never merged
into `pytorch/pytorch` main, so that base commit is not reachable from upstream.
It is carried in this fork. Diff with:

    git diff 2c1a9b7143..v1.0-paper-four-socket -- torch aten c10

The pathspec matters: the branch also carries development notes and a
`bench_2026-09-25/` scratch directory that are not part of the contribution.
`patches/pytorch-numa-tp.patch` is that same diff, restricted to the 36 source
files, and has been verified byte-identical to the tagged tree.

---

## Hardware

These results are hardware-bound and cannot be containerized. Reproducing them
requires a multi-socket x86 server with NUMA enabled:

| Platform | Sockets | Topology |
|---|---|---|
| Haswell | 2 | - |
| Cascade Lake | 2 | - |
| Haswell-EX | 4 | 4 NUMA nodes, 80 logical CPUs (20/node, 10 physical cores/node) |

Software: Ubuntu 20.04 (dual-socket) / 24.04 (four-socket), Python 3.8 / 3.9,
PyTorch 2.6.0a0, OpenBLAS, MKL/MKLDNN off, `USE_NUMA=1`.

See `BUILD.md`.

---

## Layout

    patches/      contribution diffs against the upstream bases
    results/      raw per-configuration CSVs behind every figure
    third_party/  submodules pinned to the exact paper revisions
    AUTHORS.md    authors and code contributors
    BUILD.md      build instructions for the three-component stack

## Authors

See `AUTHORS.md` for the paper's author list and the per-component code
contributors.

## Citation

    TODO: add BibTeX entry once the camera-ready is final.

## License

The forks retain their upstream licenses: PyTorch BSD-3-Clause, Triton MIT,
LLVM Apache-2.0 with LLVM exceptions. Material original to this artifact is
released under BSD-3-Clause.
