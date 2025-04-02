# Dataset Card for ComputeEval

This dataset is designed for **CUDA code generation and evaluation** tasks, where each data entry provides a self-contained CUDA programming challenge. The data highlights various aspects of CUDA programming, such as kernel launches, thread-block manipulation, shared memory usage, CCCL: Thrust/CUB, and more.

Homepage: [Github](https://github.com/NVIDIA/compute-eval)

## Format

JSON Lines (`.jsonl`), one task per line.

## Data Fields

Each task entry includes:

- `task_id`: Uniquely identifies the task, e.g., `"CUDA/0"`.
- `prompt`: A natural language description or instructions on what CUDA code to write.
- `cc_flags`, `ld_flags`: Suggests compiler/linker arguments that may be used during compilation.
- `declaration`: Preliminary code (e.g., includes, macros, or device stubs) required before the solution kernel.
- `test`: A C++ snippet (often containing a `main()` function) that checks correctness at runtime.
- `example_test`: Ancillary snippet or annotation (may be empty or contain additional tests).
- `cuda_toolkit`: String indicating the CUDA version or requirements, e.g., `"12.0"`.
- `solution`: An example or reference kernel that solves the prompt.

## License

SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0

This work is licensed under a Creative Commons Attribution 4.0 International License. https://creativecommons.org/licenses/by/4.0/
