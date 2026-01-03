import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional

# Global font sizes
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 16
LEGEND_FONTSIZE = 16
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


def _apply_axis_fonts():
    """Apply larger fonts to the current axes and legend."""
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(LEGEND_FONTSIZE)

def plot_time_graph(sizes, unfused_time, fused_time, fig_name):
    # X-axis labels
    size_labels = ["1K", "10K", "100K", "1M", "10M"]

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot lines
    plt.plot(
        sizes,
        unfused_time,
        marker="o",
        linewidth=2,
        label="Unfused"
    )

    plt.plot(
        sizes,
        fused_time,
        marker="s",
        linewidth=2,
        label="Fused"
    )

    # Log scale on both axes
    plt.xscale("log")
    plt.yscale("log")

    # Labels and title (Serbian)
    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Execution time (µs)", fontsize=AXIS_LABEL_FONTSIZE)
    # plt.title("Execution Time of Fused and Unfused Kernels")

    # Custom ticks
    plt.xticks(sizes, size_labels, fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Grid styling
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    # Legend
    plt.legend(prop={"size": LEGEND_FONTSIZE})
    _apply_axis_fonts()

    # Layout and show
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.show()


def plot_speedup_bars(
    sizes,
    speedups,
    fig_name,
    y_max_factor: float = 1.05,
    label_pad: float = 0.03,
    y_max: Optional[float] = None,
):
    """
    Bar chart: speedups with a dashed baseline at 1.0, and labels above bars.
    Matches the style of your screenshot.
    """
    size_labels = ["1K", "10K", "100K", "1M", "10M"]
    x = range(len(sizes))

    plt.figure(figsize=(10, 6))

    bars = plt.bar(x, speedups, edgecolor="black", alpha=0.8)

    # Baseline (e.g., 1.0x)
    plt.axhline(y=1.0, linestyle="--", linewidth=1.5, color="red", label="No speedup (1.0×)")

    # plt.title("Kernel Fusion Speedup")
    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Speedup", fontsize=AXIS_LABEL_FONTSIZE)

    plt.xticks(list(x), size_labels, fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)

    # Optionally lift the y-limit so bars don't touch the top
    ymax = y_max if y_max is not None else max(speedups) * y_max_factor
    plt.ylim(0, ymax)

    # Grid similar to screenshot (y-grid helps readability)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # Put value labels on top of bars
    for bar, val in zip(bars, speedups):
        label_color = "green" if val >= 1.0 else "red"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_pad,
            "{:.2f}×".format(val),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=TICK_LABEL_FONTSIZE,
            color=label_color,
        )

    plt.legend(prop={"size": LEGEND_FONTSIZE})
    _apply_axis_fonts()
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.show()


def plot_bandwidth_bars(
    sizes,
    bw_unfused,
    bw_fused,
    fig_name,
    y_max_factor: float = 1.05,
    y_max: Optional[float] = None,
):
    """Grouped bar chart comparing memory bandwidth for unfused vs fused kernels."""

    size_labels = ["1K", "10K", "100K", "1M", "10M"]
    x = list(range(len(sizes)))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - width / 2 for i in x],
        bw_unfused,
        width=width,
        label="Unfused",
        color="lightsalmon",
        edgecolor="black",
        alpha=0.8,
    )

    plt.bar(
        [i + width / 2 for i in x],
        bw_fused,
        width=width,
        label="Fused",
        color="mediumseagreen",
        edgecolor="black",
        alpha=0.8,
    )

    # plt.title("Memory bandwidth: fused vs unfused kernels")
    plt.xlabel("Array size (number of elements)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Memory bandwidth (GB/s)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.xticks(x, size_labels, fontsize=TICK_LABEL_FONTSIZE)
    plt.yticks(fontsize=TICK_LABEL_FONTSIZE)
    
    # Optionally lift the y-limit so bars don't touch the top
    ymax = y_max if y_max is not None else max(max(bw_unfused), max(bw_fused)) * y_max_factor
    plt.ylim(0, ymax)

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.legend(prop={"size": LEGEND_FONTSIZE})
    _apply_axis_fonts()

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.show()


if __name__ == "__main__":

    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    # Element-wise addition
    print("Element-wise addition results:")
    unfused_time = [3.78, 3.87, 4.64, 27.39, 245.73]
    fused_time = [1.90, 1.93, 2.39, 15.71, 146.71]
    speedup = [1.98, 2.01, 1.94, 1.74, 1.67]
    bw_unfused = [5.42, 52.91, 441.50, 747.50, 833.44]
    bw_fused = [6.45, 63.83, 515.02, 782.27, 837.58]

    speedup_calc = [u / f for u, f in zip(unfused_time, fused_time)]
    print("Element-wise addition speedup:", speedup)
    print("Element-wise addition speedup (calculated):", speedup_calc)

    plot_time_graph(sizes, unfused_time, fused_time, fig_name="elem_add_time.png")
    plot_speedup_bars(sizes, speedup, fig_name="elem_add_speedup.png", y_max=2.25)
    plot_bandwidth_bars(sizes, bw_unfused, bw_fused, fig_name="elem_add_bandwidth.png")

    print("Multiply-add results:")

    # Multiply-add (numbers extracted from the provided table screenshot)
    madd_unfused_time = [3.82, 3.83, 4.70, 31.52, 293.54]
    madd_fused_time = [1.95, 1.97, 2.67, 21.82, 190.68]
    madd_speedup = [1.96, 1.94, 1.76, 1.44, 1.54]
    madd_bw_unfused = [6.43, 64.17, 522.88, 779.73, 837.23]
    madd_bw_fused = [8.42, 83.33, 613.03, 750.82, 859.25]

    madd_speedup_calc = [u / f for u, f in zip(madd_unfused_time, madd_fused_time)]
    print("Multiply-add speedup:", madd_speedup)
    print("Multiply-add speedup (calculated):", madd_speedup_calc)

    plot_time_graph(sizes, madd_unfused_time, madd_fused_time, fig_name="fma_time.png")
    plot_speedup_bars(sizes, madd_speedup, fig_name="fma_speedup.png", y_max=2.25)
    plot_bandwidth_bars(sizes, madd_bw_unfused, madd_bw_fused, fig_name="fma_bandwidth.png")

    print("ReLU results:")

    # ReLU (numbers extracted from the provided table screenshot)
    relu_unfused_time = [5.68, 5.71, 6.83, 35.86, 344.94]
    relu_fused_time = [1.97, 1.97, 2.47, 15.94, 146.78]
    relu_speedup = [2.88, 2.90, 2.77, 2.25, 2.35]
    relu_bw_unfused = [5.05, 50.18, 419.79, 799.54, 831.21]
    relu_bw_fused = [6.25, 62.50, 497.93, 770.71, 837.17]

    relu_speedup_calc = [u / f for u, f in zip(relu_unfused_time, relu_fused_time)]
    print("ReLU speedup:", relu_speedup)
    print("ReLU speedup (calculated):", relu_speedup_calc)

    plot_time_graph(sizes, relu_unfused_time, relu_fused_time, fig_name="relu_time.png")
    plot_speedup_bars(sizes, relu_speedup, fig_name="relu_speedup.png", y_max=3.125)
    plot_bandwidth_bars(sizes, relu_bw_unfused, relu_bw_fused, fig_name="relu_bandwidth.png")

    # Map-reduce
    print("Map-reduce results:")

    # Values taken from the provided table
    mr_unfused_time = [5.94, 20.38, 169.73, 1670.38, 16675.51]
    mr_fused_time = [4.12, 19.14, 167.78, 1656.19, 16533.86]
    mr_speedup = [1.44, 1.06, 1.01, 1.01, 1.01]
    mr_bw_unfused = [2.76, 8.04, 9.65, 9.81, 9.83]
    mr_bw_fused = [1.99, 4.28, 4.88, 4.95, 4.95]

    mr_speedup_calc = [u / f for u, f in zip(mr_unfused_time, mr_fused_time)]
    print("Map-reduce speedup:", mr_speedup)
    print("Map-reduce speedup (calculated):", mr_speedup_calc)

    plot_time_graph(sizes, mr_unfused_time, mr_fused_time, fig_name="map_reduce_time.png")
    # Example: set explicit max so labels are comfortably below the top
    plot_speedup_bars(sizes, mr_speedup, fig_name="map_reduce_speedup.png", y_max=1.6)
    plot_bandwidth_bars(sizes, mr_bw_unfused, mr_bw_fused, fig_name="map_reduce_bandwidth.png")
    

    # Map-reduce
    print("Map-reduce block-level reduction results:")
    
    # Values taken from the provided table
    mr_block_unfused_time = [4.84, 4.86, 6.38, 30.50, 271.47]
    mr_block_fused_time = [3.24, 3.37, 4.44, 17.11, 133.99]
    mr_block_speedup = [1.49, 1.44, 1.44, 1.78, 2.03]
    mr_block_bw_unfused = [3.38, 33.68, 256.82, 537.09, 603.52]
    mr_block_bw_fused = [2.53, 24.32, 184.33, 478.76, 611.39]

    mr_bl_speedup_calc = [u / f for u, f in zip(mr_block_unfused_time, mr_block_fused_time)]
    print("Map-reduce block-level speedup:", mr_bl_speedup_calc)
    print("Map-reduce block-level speedup (calculated):", mr_bl_speedup_calc)

    plot_time_graph(sizes, mr_block_unfused_time, mr_block_fused_time, fig_name="map_reduce_block_level_time.png")
    plot_speedup_bars(sizes, mr_block_speedup, fig_name="map_reduce_block_level_speedup.png", y_max=2.5)
    plot_bandwidth_bars(sizes, mr_block_bw_unfused, mr_block_bw_fused, fig_name="map_reduce_block_level_bandwidth.png")
    


