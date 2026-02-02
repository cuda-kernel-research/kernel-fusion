#!/usr/bin/env python3
"""
Generate cross-GPU comparison plots only.

Usage:
    python3 generate_comparison_plots.py
"""

import os
from plot_utils import plot_time_comparison_all_gpus_both_precisions, SIZES

# Import all GPU data modules
import data_rtx3080 as rtx3080
import data_t4 as t4
import data_a100 as a100

# All GPUs to compare
gpu_modules = [rtx3080, t4, a100]
gpu_names = ["rtx3080", "t4", "a100"]
gpu_display_names = [m.GPU_DISPLAY_NAME for m in gpu_modules]

comparison_dir = os.path.join("plots", "gpu_comparison")
os.makedirs(comparison_dir, exist_ok=True)

print("=" * 60)
print("Generating Cross-GPU Comparison Plots")
print("=" * 60)
print(f"Output directory: {comparison_dir}")

# Add (FP32 & FP16 together)
print("\n[Add - All GPUs (FP32 & FP16)]")
plot_time_comparison_all_gpus_both_precisions(
    sizes=SIZES,
    gpu_fp32_data_list=[m.ADD_FP32 for m in gpu_modules],
    gpu_fp16_data_list=[m.ADD_FP16 for m in gpu_modules],
    gpu_names=gpu_names,
    gpu_display_names=gpu_display_names,
    operation_name="Element-wise Addition",
    fig_name="add_gpu_comparison.png",
    output_dir=comparison_dir
)

# FMA (FP32 & FP16 together)
print("[FMA - All GPUs (FP32 & FP16)]")
plot_time_comparison_all_gpus_both_precisions(
    sizes=SIZES,
    gpu_fp32_data_list=[m.FMA_FP32 for m in gpu_modules],
    gpu_fp16_data_list=[m.FMA_FP16 for m in gpu_modules],
    gpu_names=gpu_names,
    gpu_display_names=gpu_display_names,
    operation_name="Fused Multiply-Add",
    fig_name="fma_gpu_comparison.png",
    output_dir=comparison_dir
)

# ReLU (FP32 & FP16 together)
print("[ReLU - All GPUs (FP32 & FP16)]")
plot_time_comparison_all_gpus_both_precisions(
    sizes=SIZES,
    gpu_fp32_data_list=[m.RELU_FP32 for m in gpu_modules],
    gpu_fp16_data_list=[m.RELU_FP16 for m in gpu_modules],
    gpu_names=gpu_names,
    gpu_display_names=gpu_display_names,
    operation_name="ReLU",
    fig_name="relu_gpu_comparison.png",
    output_dir=comparison_dir
)

# Map Reduce Naive (FP32 & FP16 together)
print("[Map Reduce Naive - All GPUs (FP32 & FP16)]")
plot_time_comparison_all_gpus_both_precisions(
    sizes=SIZES,
    gpu_fp32_data_list=[m.MAP_REDUCE_NAIVE_FP32 for m in gpu_modules],
    gpu_fp16_data_list=[m.MAP_REDUCE_NAIVE_FP16 for m in gpu_modules],
    gpu_names=gpu_names,
    gpu_display_names=gpu_display_names,
    operation_name="Map Reduce Naive",
    fig_name="map_reduce_naive_gpu_comparison.png",
    output_dir=comparison_dir
)

# Map Reduce Block (FP32 & FP16 together)
print("[Map Reduce Block - All GPUs (FP32 & FP16)]")
plot_time_comparison_all_gpus_both_precisions(
    sizes=SIZES,
    gpu_fp32_data_list=[m.MAP_REDUCE_BLOCK_FP32 for m in gpu_modules],
    gpu_fp16_data_list=[m.MAP_REDUCE_BLOCK_FP16 for m in gpu_modules],
    gpu_names=gpu_names,
    gpu_display_names=gpu_display_names,
    operation_name="Map Reduce Block",
    fig_name="map_reduce_block_gpu_comparison.png",
    output_dir=comparison_dir
)

print(f"\n{'='*60}")
print("Done!")
print(f"{'='*60}")
print(f"All comparison plots saved to {comparison_dir}/")
print(f"Generated 5 plots (each with FP32 & FP16)")

