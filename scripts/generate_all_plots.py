#!/usr/bin/env python3
"""
Generate benchmark plots for all GPU architectures.

Usage:
    python generate_all_plots.py              # Generate plots for all GPUs
    python generate_all_plots.py rtx3080      # Generate plots for specific GPU
"""

import os
import sys

from plot_utils import generate_operation_plots, SIZES

# Import all GPU data modules
import data_rtx3080 as rtx3080
import data_t4 as t4
import data_a100 as a100  

# List of all available GPU data modules
ALL_GPUS = {
    "rtx3080": rtx3080,
    "t4": t4,
    "a100": a100,  
}


def generate_gpu_plots(gpu_module, output_base_dir="plots"):
    """Generate all plots for a single GPU."""
    gpu_name = gpu_module.GPU_NAME
    display_name = gpu_module.GPU_DISPLAY_NAME
    config = gpu_module.PLOT_CONFIG

    output_dir = os.path.join(output_base_dir, gpu_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating plots for {display_name}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    # Element-wise Addition
    print("\n[Add]")
    generate_operation_plots(
        op_name="add",
        fp32_data=gpu_module.ADD_FP32,
        fp16_data=gpu_module.ADD_FP16,
        output_dir=output_dir,
        sizes=SIZES,
        **config["add"]
    )

    # FMA
    print("\n[FMA]")
    generate_operation_plots(
        op_name="fma",
        fp32_data=gpu_module.FMA_FP32,
        fp16_data=gpu_module.FMA_FP16,
        output_dir=output_dir,
        sizes=SIZES,
        **config["fma"]
    )

    # ReLU
    print("\n[ReLU]")
    generate_operation_plots(
        op_name="relu",
        fp32_data=gpu_module.RELU_FP32,
        fp16_data=gpu_module.RELU_FP16,
        output_dir=output_dir,
        sizes=SIZES,
        **config["relu"]
    )

    # Map Reduce Naive (all 3 precisions)
    print("\n[Map Reduce Naive - All Precisions]")
    from plot_utils import generate_map_reduce_plots_three_precisions
    generate_map_reduce_plots_three_precisions(
        op_name="map_reduce_naive",
        fp32_data=gpu_module.MAP_REDUCE_NAIVE_FP32,
        fp16_data=gpu_module.MAP_REDUCE_NAIVE_FP16,
        mixed_data=gpu_module.MAP_REDUCE_MIXED_NAIVE,
        output_dir=output_dir,
        sizes=SIZES,
        **config["map_reduce_naive"]
    )

    # Map Reduce Block (all 3 precisions)
    print("\n[Map Reduce Block - All Precisions]")
    generate_map_reduce_plots_three_precisions(
        op_name="map_reduce_block",
        fp32_data=gpu_module.MAP_REDUCE_BLOCK_FP32,
        fp16_data=gpu_module.MAP_REDUCE_BLOCK_FP16,
        mixed_data=gpu_module.MAP_REDUCE_MIXED_BLOCK,
        output_dir=output_dir,
        sizes=SIZES,
        **config["map_reduce_block"]
    )
    
    print(f"\n All plots for {display_name} saved to {output_dir}/")
    return output_dir


def main():
    """Main entry point."""
    # Determine which GPUs to process
    if len(sys.argv) > 1:
        # Specific GPU requested
        gpu_key = sys.argv[1].lower()
        if gpu_key not in ALL_GPUS:
            print(f"Error: Unknown GPU '{gpu_key}'")
            print(f"Available GPUs: {', '.join(ALL_GPUS.keys())}")
            sys.exit(1)
        gpus_to_process = {gpu_key: ALL_GPUS[gpu_key]}
    else:
        # Process all GPUs
        gpus_to_process = ALL_GPUS

    print("=" * 60)
    print("Benchmark Plot Generator")
    print("=" * 60)
    print(f"GPUs to process: {', '.join(gpus_to_process.keys())}")

    output_dirs = []
    for gpu_key, gpu_module in gpus_to_process.items():
        output_dir = generate_gpu_plots(gpu_module)
        output_dirs.append(output_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Generated plots for {len(output_dirs)} GPU(s):")
    for d in output_dirs:
        print(f"  - {d}/")
    print("\nDone!")


if __name__ == "__main__":
    main()