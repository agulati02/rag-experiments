# type: ignore

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.spatial.distance import pdist


def exp_curse_of_dimensionality(save_path: str | None = None, show: bool = True):
    """Demonstrates the curse of dimensionality using min and max pairwise distances.

    Produces a presentation-ready plot:
    - uses a clean seaborn style
    - plots ratios on a log scale (y-axis)
    - uses categorical x positions with dimension labels (no numerical x scale assumed)
    - annotates each point with scientific-notation labels

    Args:
        save_path: if provided, save the generated figure to this path.
        show: whether to call ``plt.show()``. Useful when running headless.

    Returns:
        (fig, ax) tuple with the matplotlib objects.
    """

    dimensions = [2, 3, 192, 384, 768, 1536]
    ratios = []
    min_dists = []
    max_dists = []

    for dim in dimensions:
        # Generate 1000 random vectors of a given dimension
        vectors = np.random.rand(1000, dim)

        # Compute pairwise distances once (avoid double pdist calls)
        dists = pdist(vectors)
        min_dist = float(dists.min())
        max_dist = float(dists.max())
        min_dists.append(min_dist)
        max_dists.append(max_dist)

        # Avoid divide-by-zero; add tiny epsilon if needed
        eps = 1e-12
        ratio = max_dist / (min_dist + eps)
        ratios.append(ratio)

        print(f"Dimension: {dim}, Min Distance: {min_dist:.6g}, Max Distance: {max_dist:.6g}, Ratio (Max/Min): {ratio:.6g}")

    # Presentation-friendly bar chart (no kink)
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

    # Use categorical x positions because the x-axis values are not on a continuous scale
    x = np.arange(len(dimensions))
    bar_colors = ["#2b8cbe"] * len(x)
    bars = ax.bar(x, ratios, color=bar_colors, width=0.6, edgecolor="#204d74", linewidth=0.8)

    # Label ticks with actual dimension values
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in dimensions], ha="right")

    ax.set_yscale("log")
    ax.set_xlabel("Dimension", fontsize=14)
    ax.set_ylabel("Max / Min pairwise distance ratio (log scale)", fontsize=12)
    ax.set_title("Curse of Dimensionality: Max/Min Distance Ratio vs Dimension", fontsize=14, weight="bold")

    # Grid and subtle styling for presentation slides
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate each bar with a compact scientific label above the bar
    for bar, yi in zip(bars, ratios):
        height = bar.get_height()
        ax.annotate(f"{yi:.2e}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color="#034e7b")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()

    plt.close(fig)
    return fig, ax

if __name__ == "__main__":
    exp_curse_of_dimensionality()
