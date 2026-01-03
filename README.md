# CUDA Kernel Fusion Benchmarks

Performance benchmarks comparing fused and unfused CUDA kernels across four different element-wise operations.

## Overview

This repository contains **systematic benchmarking code and results comparing fused and unfused CUDA kernels** across four distinct element-wise operations. The goal of this work is to **quantify the performance impact of kernel fusion** in memory-bound GPU workloads by measuring execution time, memory transfer counts, and launch overhead across fused versus unfused implementations.

Kernel fusion is a well-known optimization technique in GPU computing that **combines multiple individual CUDA kernels into a single, larger kernel**, thereby reducing memory traffic between global memory and compute units, decreasing kernel launch overhead, and improving arithmetic throughput in memory-bound scenarios. :contentReference[oaicite:0]{index=0}

## Background

In modern GPU applications, a sequence of small kernels is often launched to perform successive operations such as element-wise arithmetic. Each kernel launch incurs overhead, and each kernel performs global memory reads and writes, which can dominate execution time in memory-limited workloads. Kernel fusion aims to reduce these costs by merging multiple operations into a single kernel while ensuring computational correctness and semantic equivalence. :contentReference[oaicite:1]{index=1}

This repository’s benchmarks focus on **four representative CUDA workloads** to explore how fusion affects performance characteristics such as memory transfer counts and total runtime.

## Benchmarked Operations

Each benchmark pair implements the same computation both unfused (multiple kernels) and fused (single kernel):

1. **Element-wise Addition** (`compare_add.cu`)
   - Unfused: Add kernel → Copy kernel
   - Fused: Single add kernel
   - Memory transfers: 5 → 3

2. **Fused Multiply-Add (FMA)** (`compare_fma.cu`)
   - Unfused: Multiply kernel → Add kernel
   - Fused: Single FMA kernel
   - Memory transfers: 6 → 4

3. **Scaled Add + ReLU** (`compare_relu.cu`)
   - Unfused: Scale kernel → Add kernel → ReLU kernel
   - Fused: Single kernel with all operations
   - Memory transfers: 7 → 3

4. **Map-Reduce (Naive)** (`compare_mapreduce.cu`)
   - Unfused: Map kernel → Reduce kernel (with atomics)
   - Fused: Single map-reduce kernel
   - Memory transfers: 4 → 2

5. **Map-Reduce (Block-level)** (`map_reduce_naive_vs_optimized.cu`)
   - Unfused: Map kernel → Block-level reduce kernel (shared memory + atomics)
   - Fused: Single map-reduce kernel with block-level reduction
   - Memory transfers: 4 → 2

These operations were chosen to represent common element-wise and reduction patterns encountered in GPU-accelerated workloads.

## Repository Structure

```
kernel-fusion/
├── plots/ # Visualization of benchmark results
├── scripts/ # Scripts for running experiments & collecting results
├── src/ # Source code for benchmark kernels
├── README.md # Project overview and documentation
└── .gitignore
```

## Building and Running Benchmarks

### Requirements

- CUDA Toolkit (version 11.0 or higher recommended)
- `nvcc` compiler
- Python (optional, for scripting and plotting results)

### Compilation

To build all benchmark programs, use the provided makefile or compile individual `.cu` files:

```bash
nvcc -O3 src/compare_add.cu -o compare_add
nvcc -O3 src/compare_fma.cu -o compare_fma
nvcc -O3 src/compare_relu.cu -o compare_relu
nvcc -O3 src/compare_mapreduce.cu -o compare_mapreduce
nvcc -O3 src/map_reduce_naive_vs_optimized.cu -o map_reduce_naive_vs_optimized
```

### Execution

Run each benchmark binary to collect performance results:

```bash
./compare_add
./compare_fma
./compare_relu
./compare_mapreduce
./map_reduce_naive_vs_optimized
```

Results will be written to stdout or saved via provided scripts in `scripts/`.

### Interpreting Results

Each benchmark outputs execution time and may produce comparison plots between fused and unfused implementations. Attention should be given to:

- Absolute speedup (fused vs. unfused runtime)
- Reduced memory traffic
- Kernel launch overhead impact

While the specific accelerations will depend on hardware (GPU architecture, memory subsystem), kernel fusion consistently reduces total global memory transactions and can yield significant performance improvements in memory-bound workloads.

<!--
## Citation

If this repository or its findings are used as part of academic or industrial research, please cite this paper as:
```
OVDE CEMO DODATI CITAT
```

with the following BibTeX code:
```
OVDE CEMO DODATI BIBTEX KAD IZADJE
```
-->