import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional
import os

# Global font sizes
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 14
BASE_FONT_SIZE = 15

# PDF-like serif fonts (Times family) with safe fallbacks
PDF_FONT_FAMILY = "serif"
PDF_SERIF_FALLBACKS = [
    "Times New Roman",
    "Times",
    "TeX Gyre Termes",
    "Nimbus Roman No9 L",
    "DejaVu Serif",
]

# Apply global font configuration so figures match the PDF styling
mpl.rcParams.update({
    "font.family": PDF_FONT_FAMILY,
    "font.serif": PDF_SERIF_FALLBACKS,
    "mathtext.fontset": "stix",
    "font.size": BASE_FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

# Standard sizes used in benchmarks
SIZES = [1024, 10240, 102400, 1024000, 10240000, 102400000]
SIZE_LABELS = ["1K", "10K", "100K", "1M", "10M", "100M"]


def _apply_axis_fonts():
    """Apply larger fonts to the current axes and legend."""
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(LEGEND_FONTSIZE)


def plot_time_graph_combined(
    sizes,
    fp32_unfused, fp32_fused,
    fp16_unfused, fp16_fused,
    fp32_unfused_std, fp32_fused_std,
    fp16_unfused_std, fp16_fused_std,
    fig_name,
    output_dir="."
):
    """Plot time comparison with FP32 and FP16 on same graph with error bars."""
    plt.figure(figsize=(16, 7))

    # FP32 lines
    plt.errorbar(sizes, fp32_unfused, yerr=fp32_unfused_std, marker="o", linewidth=2,
                 label="FP32 Unfused", capsize=3, color="tab:blue")
    plt.errorbar(sizes, fp32_fused, yerr=fp32_fused_std, marker="s", linewidth=2,
                 label="FP32 Fused", capsize=3, color="tab:blue", linestyle="--")

    # FP16 lines
    plt.errorbar(sizes, fp16_unfused, yerr=fp16_unfused_std, marker="o", linewidth=2,
                 label="FP16 Unfused", capsize=3, color="tab:orange")
    plt.errorbar(sizes, fp16_fused, yerr=fp16_fused_std, marker="s", linewidth=2,
                 label="FP16 Fused", capsize=3, color="tab:orange", linestyle="--")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Execution time (μs)", fontsize=AXIS_LABEL_FONTSIZE)

    plt.xticks(sizes, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE, rotation=45)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(prop={"size": LEGEND_FONTSIZE})
    _apply_axis_fonts()

    plt.tight_layout()
    filepath = os.path.join(output_dir, fig_name)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_speedup_combined(
    sizes,
    fp32_speedup, fp16_speedup,
    fp32_speedup_std, fp16_speedup_std,
    fig_name,
    output_dir=".",
    y_max: Optional[float] = None,
):
    """Bar chart comparing FP32 and FP16 speedups side by side with error bars."""
    x = list(range(len(sizes)))
    width = 0.3
    gap = 0.06  # gap between FP32 and FP16 bars

    plt.figure(figsize=(16, 7))

    bars_fp32 = plt.bar([i - width/2 - gap/2 for i in x], fp32_speedup, width=width,
                        yerr=fp32_speedup_std, capsize=3,
                        label="FP32", color="tab:blue", edgecolor="black", alpha=0.8)
    bars_fp16 = plt.bar([i + width/2 + gap/2 for i in x], fp16_speedup, width=width,
                        yerr=fp16_speedup_std, capsize=3,
                        label="FP16", color="tab:orange", edgecolor="black", alpha=0.8)

    plt.axhline(y=1.0, linestyle="--", linewidth=1.5, color="red", label="No speedup (1.0×)")

    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Speedup", fontsize=AXIS_LABEL_FONTSIZE)

    plt.xticks(x, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE, rotation=45)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    if y_max is None:
        y_max = max(max(fp32_speedup), max(fp16_speedup)) * 1.15
    plt.ylim(0, y_max)

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # Value labels on bars
    for bar, val in zip(bars_fp32, fp32_speedup):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}×", ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color="tab:blue")
    for bar, val in zip(bars_fp16, fp16_speedup):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}×", ha="center", va="bottom", fontsize=11, fontweight="bold",
                 color="tab:orange")

    plt.legend(prop={"size": LEGEND_FONTSIZE})
    _apply_axis_fonts()
    plt.tight_layout()
    filepath = os.path.join(output_dir, fig_name)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_bandwidth_combined(
    sizes,
    fp32_bw_unfused, fp32_bw_fused,
    fp16_bw_unfused, fp16_bw_fused,
    fp32_bw_unfused_std, fp32_bw_fused_std,
    fp16_bw_unfused_std, fp16_bw_fused_std,
    fig_name,
    output_dir=".",
    y_max: Optional[float] = None,
    use_log_scale: bool = True,
):
    """Grouped bar chart comparing bandwidth for FP32 and FP16."""
    x = [i * 1.6 for i in range(len(sizes))]  # Increase spacing between size categories
    width = 0.18
    gap = 0.08  # gap between FP32 and FP16 groups

    plt.figure(figsize=(18, 7))

    # FP32 bars (shifted left)
    plt.bar([i - width - gap/2 - width/2 for i in x], fp32_bw_unfused, width=width,
            yerr=fp32_bw_unfused_std, capsize=2,
            label="FP32 Unfused", color="lightblue", edgecolor="black", alpha=0.8)
    plt.bar([i - gap/2 - width/2 for i in x], fp32_bw_fused, width=width,
            yerr=fp32_bw_fused_std, capsize=2,
            label="FP32 Fused", color="steelblue", edgecolor="black", alpha=0.8)

    # FP16 bars (shifted right)
    plt.bar([i + gap/2 + width/2 for i in x], fp16_bw_unfused, width=width,
            yerr=fp16_bw_unfused_std, capsize=2,
            label="FP16 Unfused", color="lightsalmon", edgecolor="black", alpha=0.8)
    plt.bar([i + width + gap/2 + width/2 for i in x], fp16_bw_fused, width=width,
            yerr=fp16_bw_fused_std, capsize=2,
            label="FP16 Fused", color="tomato", edgecolor="black", alpha=0.8)

    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Memory bandwidth (GB/s)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(x, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE, rotation=45)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    all_vals = fp32_bw_unfused + fp32_bw_fused + fp16_bw_unfused + fp16_bw_fused
    if use_log_scale:
        plt.yscale("log")
        # Find minimum positive value to set appropriate lower limit
        min_positive = min(v for v in all_vals if v > 0)
        ymin = min_positive * 0.5  # Set lower limit to 50% of minimum value
        ymax = y_max if y_max is not None else max(all_vals) * 1.2
        plt.ylim(ymin, ymax)
    else:
        if y_max is None:
            y_max = max(all_vals) * 1.1
        plt.ylim(0, y_max)

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(prop={"size": LEGEND_FONTSIZE}, loc="upper left")
    _apply_axis_fonts()

    plt.tight_layout()
    filepath = os.path.join(output_dir, fig_name)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def generate_operation_plots(
    op_name: str,
    fp32_data: dict,
    fp16_data: dict,
    output_dir: str,
    sizes=None,
    speedup_y_max: Optional[float] = None,
    bandwidth_y_max: Optional[float] = None,
    use_log_scale: bool = True,
):
    """
    Generate all plots (time, speedup, bandwidth) for a single operation.
    
    Args:
        op_name: Name of operation (e.g., "add", "fma", "relu")
        fp32_data: Dictionary with FP32 benchmark data
        fp16_data: Dictionary with FP16 benchmark data
        output_dir: Directory to save plots
        sizes: Array sizes (uses default SIZES if None)
        speedup_y_max: Maximum y value for speedup plot
        bandwidth_y_max: Maximum y value for bandwidth plot
    """
    if sizes is None:
        sizes = SIZES

    os.makedirs(output_dir, exist_ok=True)

    # Time plot
    plot_time_graph_combined(
        sizes,
        fp32_data["unfused"], fp32_data["fused"],
        fp16_data["unfused"], fp16_data["fused"],
        fp32_data["unfused_std"], fp32_data["fused_std"],
        fp16_data["unfused_std"], fp16_data["fused_std"],
        fig_name=f"{op_name}_time.png",
        output_dir=output_dir
    )

    # Speedup plot
    plot_speedup_combined(
        sizes,
        fp32_data["speedup"], fp16_data["speedup"],
        fp32_data["speedup_std"], fp16_data["speedup_std"],
        fig_name=f"{op_name}_speedup.png",
        output_dir=output_dir,
        y_max=speedup_y_max
    )

    # Bandwidth plot
    plot_bandwidth_combined(
        sizes,
        fp32_data["bw_unfused"], fp32_data["bw_fused"],
        fp16_data["bw_unfused"], fp16_data["bw_fused"],
        fp32_data["bw_unfused_std"], fp32_data["bw_fused_std"],
        fp16_data["bw_unfused_std"], fp16_data["bw_fused_std"],
        fig_name=f"{op_name}_bandwidth.png",
        output_dir=output_dir,
        y_max=bandwidth_y_max,
        use_log_scale=use_log_scale
    )

    print(f" Generated {op_name} plots")


def plot_time_graph_three_precisions(
    sizes,
    fp32_unfused, fp32_fused,
    fp16_unfused, fp16_fused,
    mixed_unfused, mixed_fused,
    fp32_unfused_std, fp32_fused_std,
    fp16_unfused_std, fp16_fused_std,
    mixed_unfused_std, mixed_fused_std,
    fig_name,
    output_dir="."
):
    """Plot time comparison with FP32, FP16, and Mixed precision on same graph with error bars."""
    plt.figure(figsize=(16, 7))

    # FP32 lines
    plt.errorbar(sizes, fp32_unfused, yerr=fp32_unfused_std, marker="o", linewidth=2,
                 label="FP32 Unfused", capsize=3, color="tab:blue")
    plt.errorbar(sizes, fp32_fused, yerr=fp32_fused_std, marker="s", linewidth=2,
                 label="FP32 Fused", capsize=3, color="tab:blue", linestyle="--")

    # FP16 lines
    plt.errorbar(sizes, fp16_unfused, yerr=fp16_unfused_std, marker="o", linewidth=2,
                 label="FP16 Unfused", capsize=3, color="tab:orange")
    plt.errorbar(sizes, fp16_fused, yerr=fp16_fused_std, marker="s", linewidth=2,
                 label="FP16 Fused", capsize=3, color="tab:orange", linestyle="--")

    # Mixed lines
    plt.errorbar(sizes, mixed_unfused, yerr=mixed_unfused_std, marker="o", linewidth=2,
                 label="Mixed Unfused", capsize=3, color="tab:green")
    plt.errorbar(sizes, mixed_fused, yerr=mixed_fused_std, marker="s", linewidth=2,
                 label="Mixed Fused", capsize=3, color="tab:green", linestyle="--")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Execution time (μs)", fontsize=AXIS_LABEL_FONTSIZE)

    plt.xticks(sizes, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE, rotation=45)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(prop={"size": LEGEND_FONTSIZE})
    _apply_axis_fonts()

    plt.tight_layout()
    filepath = os.path.join(output_dir, fig_name)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_speedup_three_precisions(
    sizes,
    fp32_speedup, fp16_speedup, mixed_speedup,
    fp32_speedup_std, fp16_speedup_std, mixed_speedup_std,
    fig_name,
    output_dir=".",
    y_max: Optional[float] = None,
):
    """Bar chart comparing FP32, FP16, and Mixed precision speedups side by side with error bars."""
    x = list(range(len(sizes)))
    width = 0.25
    gap = 0.04

    plt.figure(figsize=(16, 7))

    bars_fp32 = plt.bar([i - width - gap for i in x], fp32_speedup, width=width,
                        yerr=fp32_speedup_std, capsize=3,
                        label="FP32", color="tab:blue", edgecolor="black", alpha=0.8)
    bars_fp16 = plt.bar([i for i in x], fp16_speedup, width=width,
                        yerr=fp16_speedup_std, capsize=3,
                        label="FP16", color="tab:orange", edgecolor="black", alpha=0.8)
    bars_mixed = plt.bar([i + width + gap for i in x], mixed_speedup, width=width,
                        yerr=mixed_speedup_std, capsize=3,
                        label="Mixed (FP16+FP32)", color="tab:green", edgecolor="black", alpha=0.8)

    plt.axhline(y=1.0, linestyle="--", linewidth=1.5, color="red", label="No speedup (1.0×)")

    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Speedup", fontsize=AXIS_LABEL_FONTSIZE)

    plt.xticks(x, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE, rotation=45)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    if y_max is None:
        y_max = max(max(fp32_speedup), max(fp16_speedup), max(mixed_speedup)) * 1.15
    plt.ylim(0, y_max)

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # Value labels on bars
    for bar, val in zip(bars_fp32, fp32_speedup):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:.2f}×", ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color="tab:blue")
    for bar, val in zip(bars_fp16, fp16_speedup):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:.2f}×", ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color="tab:orange")
    for bar, val in zip(bars_mixed, mixed_speedup):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                 f"{val:.2f}×", ha="center", va="bottom", fontsize=8, fontweight="bold",
                 color="tab:green")

    plt.legend(prop={"size": LEGEND_FONTSIZE})
    _apply_axis_fonts()
    plt.tight_layout()
    filepath = os.path.join(output_dir, fig_name)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def plot_bandwidth_three_precisions(
    sizes,
    fp32_bw_unfused, fp32_bw_fused,
    fp16_bw_unfused, fp16_bw_fused,
    mixed_bw_unfused, mixed_bw_fused,
    fp32_bw_unfused_std, fp32_bw_fused_std,
    fp16_bw_unfused_std, fp16_bw_fused_std,
    mixed_bw_unfused_std, mixed_bw_fused_std,
    fig_name,
    output_dir=".",
    y_max: Optional[float] = None,
    use_log_scale: bool = True,
):
    """Grouped bar chart comparing bandwidth for FP32, FP16, and Mixed precision."""
    x = [i * 1.6 for i in range(len(sizes))]  # Increase spacing between size categories
    width = 0.13
    gap = 0.15

    plt.figure(figsize=(18, 7))

    # FP32 bars (left group)
    plt.bar([i - width*2 - gap for i in x], fp32_bw_unfused, width=width,
            yerr=fp32_bw_unfused_std, capsize=2,
            label="FP32 Unfused", color="lightblue", edgecolor="black", alpha=0.8)
    plt.bar([i - width - gap for i in x], fp32_bw_fused, width=width,
            yerr=fp32_bw_fused_std, capsize=2,
            label="FP32 Fused", color="steelblue", edgecolor="black", alpha=0.8)

    # FP16 bars (middle group)
    plt.bar([i for i in x], fp16_bw_unfused, width=width,
            yerr=fp16_bw_unfused_std, capsize=2,
            label="FP16 Unfused", color="lightsalmon", edgecolor="black", alpha=0.8)
    plt.bar([i + width for i in x], fp16_bw_fused, width=width,
            yerr=fp16_bw_fused_std, capsize=2,
            label="FP16 Fused", color="tomato", edgecolor="black", alpha=0.8)

    # Mixed bars (right group)
    plt.bar([i + width*2 + gap for i in x], mixed_bw_unfused, width=width,
            yerr=mixed_bw_unfused_std, capsize=2,
            label="Mixed Unfused", color="lightgreen", edgecolor="black", alpha=0.8)
    plt.bar([i + width*3 + gap for i in x], mixed_bw_fused, width=width,
            yerr=mixed_bw_fused_std, capsize=2,
            label="Mixed Fused", color="forestgreen", edgecolor="black", alpha=0.8)

    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Memory bandwidth (GB/s)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(x, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE, rotation=45)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    all_vals = (fp32_bw_unfused + fp32_bw_fused + fp16_bw_unfused + fp16_bw_fused +
                mixed_bw_unfused + mixed_bw_fused)
    if use_log_scale:
        plt.yscale("log")
        min_positive = min(v for v in all_vals if v > 0)
        ymin = min_positive * 0.5
        ymax = y_max if y_max is not None else max(all_vals) * 1.2
        plt.ylim(ymin, ymax)
    else:
        if y_max is None:
            y_max = max(all_vals) * 1.1
        plt.ylim(0, y_max)

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(prop={"size": LEGEND_FONTSIZE}, loc="upper left")
    _apply_axis_fonts()

    plt.tight_layout()
    filepath = os.path.join(output_dir, fig_name)
    plt.savefig(filepath, dpi=300)
    plt.close()
    return filepath


def generate_map_reduce_plots_three_precisions(
    op_name: str,
    fp32_data: dict,
    fp16_data: dict,
    mixed_data: dict,
    output_dir: str,
    sizes=None,
    speedup_y_max: Optional[float] = None,
    bandwidth_y_max: Optional[float] = None,
    use_log_scale: bool = True,
):
    """
    Generate plots comparing all three precisions (FP32, FP16, Mixed) for map reduce operations.
    
    Args:
        op_name: Name of operation (e.g., "map_reduce_naive", "map_reduce_block")
        fp32_data: Dictionary with FP32 benchmark data
        fp16_data: Dictionary with FP16 benchmark data
        mixed_data: Dictionary with Mixed precision benchmark data
        output_dir: Directory to save plots
        sizes: Array sizes (uses default SIZES if None)
        speedup_y_max: Maximum y value for speedup plot
        bandwidth_y_max: Maximum y value for bandwidth plot
    """
    if sizes is None:
        sizes = SIZES

    os.makedirs(output_dir, exist_ok=True)

    # Time plot
    plot_time_graph_three_precisions(
        sizes,
        fp32_data["unfused"], fp32_data["fused"],
        fp16_data["unfused"], fp16_data["fused"],
        mixed_data["unfused"], mixed_data["fused"],
        fp32_data["unfused_std"], fp32_data["fused_std"],
        fp16_data["unfused_std"], fp16_data["fused_std"],
        mixed_data["unfused_std"], mixed_data["fused_std"],
        fig_name=f"{op_name}_time.png",
        output_dir=output_dir
    )

    # Speedup plot
    plot_speedup_three_precisions(
        sizes,
        fp32_data["speedup"], fp16_data["speedup"], mixed_data["speedup"],
        fp32_data["speedup_std"], fp16_data["speedup_std"], mixed_data["speedup_std"],
        fig_name=f"{op_name}_speedup.png",
        output_dir=output_dir,
        y_max=speedup_y_max
    )

    # Bandwidth plot
    plot_bandwidth_three_precisions(
        sizes,
        fp32_data["bw_unfused"], fp32_data["bw_fused"],
        fp16_data["bw_unfused"], fp16_data["bw_fused"],
        mixed_data["bw_unfused"], mixed_data["bw_fused"],
        fp32_data["bw_unfused_std"], fp32_data["bw_fused_std"],
        fp16_data["bw_unfused_std"], fp16_data["bw_fused_std"],
        mixed_data["bw_unfused_std"], mixed_data["bw_fused_std"],
        fig_name=f"{op_name}_bandwidth.png",
        output_dir=output_dir,
        y_max=bandwidth_y_max,
        use_log_scale=use_log_scale
    )

    print(f" Generated {op_name} plots (3 precisions)")


def plot_time_comparison_all_gpus(
    sizes,
    gpu_data_list,
    gpu_names,
    gpu_display_names,
    operation_name,
    fig_name,
    output_dir,
    precision="fp32",
    y_max=None
):
    """
    Plot execution time comparison across multiple GPUs for a specific operation and precision.
    
    Args:
        sizes: Array sizes
        gpu_data_list: List of data dictionaries for each GPU
        gpu_names: List of GPU short names (e.g., ['rtx3080', 't4', 'a100'])
        gpu_display_names: List of GPU display names (e.g., ['RTX 3080', 'T4', 'A100'])
        operation_name: Name of the operation (e.g., 'Add', 'FMA', 'ReLU')
        fig_name: Output filename
        output_dir: Directory to save plot
        precision: 'fp32' or 'fp16'
        y_max: Maximum y-axis value (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU colors and markers
    gpu_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    gpu_markers = ['o', 's', '^', 'D', 'v', 'p']
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    for idx, (data, gpu_name, display_name) in enumerate(zip(gpu_data_list, gpu_names, gpu_display_names)):
        color = gpu_colors[idx % len(gpu_colors)]
        marker = gpu_markers[idx % len(gpu_markers)]
        
        # Unfused
        ax.errorbar(
            sizes, data["unfused"], yerr=data["unfused_std"],
            label=f"{display_name} Unfused",
            marker=marker, markersize=8, linewidth=2,
            capsize=4, capthick=1.5,
            color=color, linestyle='-', alpha=0.8
        )
        
        # Fused
        ax.errorbar(
            sizes, data["fused"], yerr=data["fused_std"],
            label=f"{display_name} Fused",
            marker=marker, markersize=8, linewidth=2,
            capsize=4, capthick=1.5,
            color=color, linestyle='--', alpha=0.8
        )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Array Size", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Execution Time (μs)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{operation_name} - {precision.upper()} - GPU Comparison", fontsize=AXIS_LABEL_FONTSIZE + 2)
    ax.set_xticks(sizes)
    ax.set_xticklabels(SIZE_LABELS, rotation=45, ha='right')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=LEGEND_FONTSIZE - 2, loc='best', ncol=2)
    
    if y_max is not None:
        ax.set_ylim(top=y_max)
    
    _apply_axis_fonts()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fig_name), dpi=300, bbox_inches="tight")
    plt.close()


def plot_time_comparison_all_gpus_both_precisions(
    sizes,
    gpu_fp32_data_list,
    gpu_fp16_data_list,
    gpu_names,
    gpu_display_names,
    operation_name,
    fig_name,
    output_dir,
    y_max=None
):
    """
    Plot execution time comparison across multiple GPUs showing both FP32 and FP16 on same graph.
    
    Args:
        sizes: Array sizes
        gpu_fp32_data_list: List of FP32 data dictionaries for each GPU
        gpu_fp16_data_list: List of FP16 data dictionaries for each GPU
        gpu_names: List of GPU short names
        gpu_display_names: List of GPU display names
        operation_name: Name of the operation
        fig_name: Output filename
        output_dir: Directory to save plot
        y_max: Maximum y-axis value (optional)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # GPU colors
    gpu_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    gpu_markers = ['o', 's', '^', 'D', 'v', 'p']
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    for idx, (fp32_data, fp16_data, gpu_name, display_name) in enumerate(
        zip(gpu_fp32_data_list, gpu_fp16_data_list, gpu_names, gpu_display_names)
    ):
        color = gpu_colors[idx % len(gpu_colors)]
        marker = gpu_markers[idx % len(gpu_markers)]
        
        # FP32 Unfused
        ax.errorbar(
            sizes, fp32_data["unfused"], yerr=fp32_data["unfused_std"],
            label=f"{display_name} FP32 Unfused",
            marker=marker, markersize=8, linewidth=2.5,
            capsize=4, capthick=1.5,
            color=color, linestyle='-', alpha=0.9
        )
        
        # FP32 Fused
        ax.errorbar(
            sizes, fp32_data["fused"], yerr=fp32_data["fused_std"],
            label=f"{display_name} FP32 Fused",
            marker=marker, markersize=8, linewidth=2.5,
            capsize=4, capthick=1.5,
            color=color, linestyle='--', alpha=0.9
        )
        
        # FP16 Unfused (lighter color)
        ax.errorbar(
            sizes, fp16_data["unfused"], yerr=fp16_data["unfused_std"],
            label=f"{display_name} FP16 Unfused",
            marker=marker, markersize=7, linewidth=2,
            capsize=3, capthick=1.2,
            color=color, linestyle='-', alpha=0.4
        )
        
        # FP16 Fused (lighter color)
        ax.errorbar(
            sizes, fp16_data["fused"], yerr=fp16_data["fused_std"],
            label=f"{display_name} FP16 Fused",
            marker=marker, markersize=7, linewidth=2,
            capsize=3, capthick=1.2,
            color=color, linestyle='--', alpha=0.4
        )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Array Size", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Execution Time (μs)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(f"{operation_name} - GPU Comparison (FP32 & FP16)", fontsize=AXIS_LABEL_FONTSIZE + 2)
    ax.set_xticks(sizes)
    ax.set_xticklabels(SIZE_LABELS, rotation=45, ha='right')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=LEGEND_FONTSIZE - 3, loc='best', ncol=2)
    
    if y_max is not None:
        ax.set_ylim(top=y_max)
    
    _apply_axis_fonts()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fig_name), dpi=300, bbox_inches="tight")
    plt.close()
