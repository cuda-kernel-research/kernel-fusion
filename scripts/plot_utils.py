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
    plt.figure(figsize=(12, 7))

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

    plt.xticks(sizes, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE)
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

    plt.figure(figsize=(12, 7))

    bars_fp32 = plt.bar([i - width/2 - gap/2 for i in x], fp32_speedup, width=width,
                        yerr=fp32_speedup_std, capsize=3,
                        label="FP32", color="tab:blue", edgecolor="black", alpha=0.8)
    bars_fp16 = plt.bar([i + width/2 + gap/2 for i in x], fp16_speedup, width=width,
                        yerr=fp16_speedup_std, capsize=3,
                        label="FP16", color="tab:orange", edgecolor="black", alpha=0.8)

    plt.axhline(y=1.0, linestyle="--", linewidth=1.5, color="red", label="No speedup (1.0×)")

    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Speedup", fontsize=AXIS_LABEL_FONTSIZE)

    plt.xticks(x, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE)
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
    x = list(range(len(sizes)))
    width = 0.18
    gap = 0.08  # gap between FP32 and FP16 groups

    plt.figure(figsize=(14, 7))

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
    plt.xticks(x, SIZE_LABELS, fontsize=TICK_LABEL_FONTSIZE)
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
        y_max=bandwidth_y_max
    )

    print(f" Generated {op_name} plots")