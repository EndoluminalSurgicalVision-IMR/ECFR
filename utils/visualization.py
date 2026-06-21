from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _as_numpy(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


def plot_entropy_consistency_scatter(
    entropies,
    consistencies,
    error_values=None,
    entropy_threshold=None,
    consistency_threshold=None,
    save_path="ECFR_Quadrant_Flow.png",
    x_max=None,
    y_max=None,
    title="Entropy-Consistency Flow",
    colorbar_label="Squared Prediction Error",
    point_size=34,
    dpi=300,
):
    """Plot ECFR samples in the entropy-consistency plane.

    The function is intentionally independent of any dataset path or experiment
    cache so it can be reused for Mendeley, Daffodil, HAM10000, or new streams.
    """
    entropies = _as_numpy(entropies)
    consistencies = _as_numpy(consistencies)

    if entropies.shape != consistencies.shape:
        raise ValueError("entropies and consistencies must have the same shape")

    if error_values is None:
        error_values = np.zeros_like(entropies)
        colorbar_label = "Sample Index"
    else:
        error_values = _as_numpy(error_values)
        if error_values.shape != entropies.shape:
            raise ValueError("error_values must have the same shape as entropies")

    entropy_threshold = float(entropy_threshold) if entropy_threshold is not None else float(np.median(entropies))
    consistency_threshold = (
        float(consistency_threshold)
        if consistency_threshold is not None
        else float(np.median(consistencies))
    )

    x_min = 0.0
    y_min = 0.0
    x_max = float(x_max) if x_max is not None else max(float(np.max(entropies)) * 1.05, entropy_threshold * 1.5, 1e-8)
    y_max = float(y_max) if y_max is not None else max(float(np.max(consistencies)) * 1.05, consistency_threshold * 1.5, 1e-8)

    valid = (
        (entropies >= x_min)
        & (entropies <= x_max)
        & (consistencies >= y_min)
        & (consistencies <= y_max)
    )

    x = entropies[valid]
    y = consistencies[valid]
    c = error_values[valid]

    plt.ioff()
    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    regions = [
        ((x_min, y_min), entropy_threshold - x_min, consistency_threshold - y_min, "#d9f2d9"),
        ((x_min, consistency_threshold), entropy_threshold - x_min, y_max - consistency_threshold, "#dce6ff"),
        ((entropy_threshold, y_min), x_max - entropy_threshold, consistency_threshold - y_min, "#eeeeee"),
        ((entropy_threshold, consistency_threshold), x_max - entropy_threshold, y_max - consistency_threshold, "#f3d7ee"),
    ]
    for xy, width, height, color in regions:
        ax.add_patch(
            plt.Rectangle(
                xy,
                max(width, 0.0),
                max(height, 0.0),
                facecolor=color,
                alpha=0.55,
                edgecolor="none",
                zorder=0,
            )
        )

    ax.axvline(entropy_threshold, color="#555555", linestyle="--", linewidth=1.4, alpha=0.85)
    ax.axhline(consistency_threshold, color="#555555", linestyle="--", linewidth=1.4, alpha=0.85)

    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap="YlOrRd",
        s=point_size,
        alpha=0.82,
        edgecolors="black",
        linewidths=0.25,
        zorder=2,
    )
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.86)
    cbar.set_label(colorbar_label, fontsize=11)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Entropy", fontsize=12)
    ax.set_ylabel("Consistency (Jensen-Shannon divergence)", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(True, alpha=0.22, linewidth=0.7)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ecfr_2d_dynamics(
    entropies,
    cons_losses,
    brier_solo_values,
    entropy_thresh,
    cons_thresh,
    x_max_fixed=0.693,
    y_max_fixed=0.693,
    save_path="ECFR_2D_Dynamics.png",
):
    """Backward-compatible wrapper for the ECFR paper visualization."""
    plot_entropy_consistency_scatter(
        entropies=entropies,
        consistencies=cons_losses,
        error_values=brier_solo_values,
        entropy_threshold=entropy_thresh,
        consistency_threshold=cons_thresh,
        save_path=save_path,
        x_max=x_max_fixed,
        y_max=y_max_fixed,
        title="ECFR Entropy-Consistency Flow",
        colorbar_label="Squared Prediction Error",
    )
