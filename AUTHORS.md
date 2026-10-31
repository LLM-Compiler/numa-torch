# Authors and contributors

## Paper authors

**NUMA-Torch: Compiler-Driven NUMA-Aware LLM Prefill on Multi-Socket CPUs**

- Thisara Rathnayaka Weerakoon — Dept. of Computer Science & Engineering, University of Moratuwa, Sri Lanka
- Ishan Dewpura — Dept. of Computer Science & Engineering, University of Moratuwa, Sri Lanka
- Niman Ransindu — Dept. of Computer Science & Engineering, University of Moratuwa, Sri Lanka
- Shavin Anjitha — Dept. of Computer Science & Engineering, University of Moratuwa, Sri Lanka
- Tharindu Rusira Patabandi — Qualcomm Technologies, Inc.
- Sanath Jayasena — Dept. of Computer Science & Engineering, University of Moratuwa, Sri Lanka

## Code contributors

Commit counts are from the forks' NUMA-Torch history: `LLM-Compiler/pytorch`
from base `2c1a9b7143` to `v1.0-paper-four-socket` (646 commits), and
`LLM-Compiler/triton-cpu` from base `e60f448f` to `v1.0-paper` (7 commits).

| Contributor | GitHub | pytorch | triton-cpu |
|---|---|---|---|
| Thisara Rathnayaka Weerakoon | [@ThisaraWeerakoon](https://github.com/ThisaraWeerakoon) | 627 | 6 |
| Niman Ransindu | [@NRM10101](https://github.com/NRM10101) | 17 | - |

Because the NUMA-Torch work lives on tagged branches rather than the forks'
default branch, GitHub's contributors graph for `LLM-Compiler/pytorch` and
`LLM-Compiler/triton-cpu` shows the upstream PyTorch and Triton contributors
instead. This table is the authoritative credit for the NUMA-Torch changes.

## Upstream

NUMA-Torch builds on PyTorch, Triton / Triton-CPU and LLVM. Those projects'
contributors are credited in their own repositories; the forks retain their
upstream commit history and licenses unchanged.
