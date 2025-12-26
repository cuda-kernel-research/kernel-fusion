# CUDA Kernel Fusion Benchmarks

Performance benchmarks comparing fused and unfused CUDA kernels across four different element-wise operations.

## Overview

This repository contains benchmarking code and results for analyzing the performance impact of kernel fusion in CUDA. Kernel fusion combines multiple GPU operations into a single kernel launch, potentially reducing memory traffic and kernel launch overhead.

## Benchmarked Operations

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
