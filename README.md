# CUDA Kernel Fusion Benchmarks

Systematic performance benchmarks comparing fused and unfused CUDA kernels across multiple element-wise and reduction operations, evaluated on **three GPU architectures**: NVIDIA Tesla T4, RTX 3080, and A100.

## Overview

This repository contains benchmarking code and results that **quantify the performance impact of kernel fusion** in memory-bound GPU workloads. Each benchmark measures execution time, effective memory bandwidth, and speedup between fused and unfused kernel implementations. All experiments are run across **both FP32 and FP16 precisions**, with an additional mixed-precision variant for the map-reduce workload.

Kernel fusion is a well-known optimization technique in GPU computing that **combines multiple individual CUDA kernels into a single kernel**, reducing round-trips to global memory, eliminating intermediate buffer allocations, and decreasing kernel launch overhead.

## Benchmarked Operations

Each benchmark implements the same computation in both unfused (multiple kernels) and fused (single kernel) form, in **FP32 and FP16**:

### Element-wise Operations

1. **Element-wise Addition** (`compare_add_fp32.cu` / `compare_add_fp16.cu`)
   - Unfused: Add kernel → Copy kernel
   - Fused: Single add kernel
   - Memory transfers: 5 → 3

2. **Fused Multiply-Add (FMA)** (`compare_fma_fp32.cu` / `compare_fma_fp16.cu`)
   - Unfused: Multiply kernel → Add kernel
   - Fused: Single FMA kernel
   - Memory transfers: 6 → 4

3. **Scaled Add + ReLU** (`relu_fp32.cu` / `relu_fp16.cu`)
   - Unfused: Scale kernel → Add kernel → ReLU kernel
   - Fused: Single kernel with all operations
   - Memory transfers: 7 → 3

### Map-Reduce Operations

4. **Map-Reduce — Naive** (`map_reduce_naive_vs_optimized_fp32.cu` / `map_reduce_naive_vs_optimized_fp16.cu`)
   - Unfused: Map kernel → Reduce kernel (with atomics)
   - Fused: Single map-reduce kernel
   - Memory transfers: 4 → 2

5. **Map-Reduce — Block-level** (same source files as above)
   - Unfused: Map kernel → Block-level reduce kernel (shared memory + atomics)
   - Fused: Single map-reduce kernel with block-level reduction
   - Memory transfers: 4 → 2

6. **Map-Reduce — Mixed Precision** (`map_reduce_mix_precision.cu`)
   - Map in FP16, reduce accumulated in FP32
   - Compares fused vs. unfused mixed-precision pipelines

Input sizes range from **1 000** to **100 000 000** elements to capture both launch-overhead-dominated and bandwidth-dominated regimes.

## Repository Structure

```
kernel-fusion/
├── experiments/          # CUDA source files and orchestration scripts
│   ├── compare_add_fp32.cu / compare_add_fp16.cu
│   ├── compare_fma_fp32.cu / compare_fma_fp16.cu
│   ├── relu_fp32.cu / relu_fp16.cu
│   ├── map_reduce_naive_vs_optimized_fp32.cu / _fp16.cu
│   ├── map_reduce_mix_precision.cu
│   ├── main_a100.py / main_t4.py          # Per-GPU experiment runners
│   ├── main_map_reduce_*.py               # Map-reduce experiment runners
│   └── submit-experiment.sh / submit-mapreduce.sh
├── results/              # Raw benchmark output logs
│   ├── a100.txt / t4.txt / 3080.txt       # Element-wise results
│   ├── a100_map_reduce.txt / t4_map_reduce.txt / 3080_map_reduce.txt
│   └── mapreduce_mixedprec.txt
├── plots/                # Generated visualisation plots
│   ├── A100/             # Per-GPU plots (speedup, bandwidth, time)
│   ├── T4/
│   ├── RTX_3080/
│   └── time plots/       # Cross-GPU comparison plots
├── scripts/              # Data extraction, plotting, and HW info collection
│   ├── data_a100.py / data_rtx3080.py / data_t4.py
│   ├── draw_graphs.py / generate_all_plots.py / generate_comparison_plots.py
│   ├── plot_utils.py
│   ├── device_specifications.py
│   └── hw_dump_a100/ / hw_dump_t4/        # Collected hardware specs
├── requirements_a100.txt / requirements_rtx3080.txt / requirements_t4.txt
└── README.md
```

## Building and Running

### Prerequisites

- CUDA Toolkit ≥ 11.0 (tested with 11.2 and 12.5)
- `nvcc` compiler
- Python 3 with `matplotlib` and `numpy` (for plotting)
- Install Python dependencies for your target GPU:
  ```bash
  pip install -r requirements_a100.txt   # or requirements_t4.txt / requirements_rtx3080.txt
  ```

### Compilation

Compile individual CUDA benchmarks with:

```bash
nvcc -O3 experiments/compare_add_fp32.cu -o compare_add_fp32
nvcc -O3 experiments/compare_add_fp16.cu -o compare_add_fp16
nvcc -O3 experiments/compare_fma_fp32.cu -o compare_fma_fp32
nvcc -O3 experiments/compare_fma_fp16.cu -o compare_fma_fp16
nvcc -O3 experiments/relu_fp32.cu -o relu_fp32
nvcc -O3 experiments/relu_fp16.cu -o relu_fp16
nvcc -O3 experiments/map_reduce_naive_vs_optimized_fp32.cu -o map_reduce_fp32
nvcc -O3 experiments/map_reduce_naive_vs_optimized_fp16.cu -o map_reduce_fp16
nvcc -O3 experiments/map_reduce_mix_precision.cu -o map_reduce_mixed
```

### Running the Full Experiment Suite

Use the provided Python orchestrators or shell scripts:

```bash
# Element-wise benchmarks on A100
python experiments/main_a100.py

# Element-wise benchmarks on T4
python experiments/main_t4.py

# Map-reduce benchmarks (per GPU)
python experiments/main_map_reduce_a100.py
python experiments/main_map_reduce_t4.py
python experiments/main_map_reduce_3080.py

# Or via SLURM
sbatch experiments/submit-experiment.sh
sbatch experiments/submit-mapreduce.sh
```

### Generating Plots

```bash
# Per-GPU plots (speedup, bandwidth, execution time)
python scripts/generate_all_plots.py

# Cross-GPU comparison plots
python scripts/generate_comparison_plots.py
```

Plots are saved to the `plots/` directory, organized by GPU and comparison type.

### Interpreting Results

Each benchmark reports:

- **Unfused / Fused execution time** (μs) — mean ± std over 10 runs
- **Speedup** — ratio of unfused to fused time
- **Effective bandwidth** (GB/s) — data movement rate for each variant

Key takeaways:

- Fusion provides the largest speedup in **memory-bound, element-wise kernels** where the dominant cost is global memory traffic.
- For **reduction workloads**, the block-level variant with shared memory benefits significantly more from fusion than the naive atomic-based variant.
- Speedups are **consistent across all three GPU architectures**, confirming that fusion is a hardware-agnostic optimisation for this class of workloads.
- **FP16 kernels** transfer half the data per element, but fusion still provides comparable relative speedups.

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
