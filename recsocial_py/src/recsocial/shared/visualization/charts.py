"""Reusable chart builders for paper-style figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recsocial.shared.visualization.style import apply_style, color_for_algorithm


def _format_value(value: float, *, decimals: int = 3) -> str:
    if abs(value) >= 100 or (abs(value) < 0.001 and value != 0):
        return f"{value:.2e}"
    return f"{value:.{decimals}f}"


def _annotate_vertical_bars(
    ax: plt.Axes,
    bars,
    *,
    fmt: str | None = None,
    decimals: int = 3,
    offset: float = 0.015,
) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        label = fmt.format(height) if fmt else _format_value(float(height), decimals=decimals)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="medium",
        )
        ymax = max(ymax, float(height) + offset * 4)
    ax.set_ylim(0, min(1.15, ymax * 1.05) if ymax <= 1.1 else ymax * 1.08)


def _annotate_horizontal_bars(
    ax: plt.Axes,
    values: pd.Series | list[float],
    y_positions: np.ndarray,
    *,
    fmt: str | None = None,
    decimals: int = 3,
    pad: float = 0.01,
) -> None:
    xmax = ax.get_xlim()[1]
    for y, value in zip(y_positions, values, strict=False):
        if pd.isna(value):
            continue
        label = fmt.format(value) if fmt else _format_value(float(value), decimals=decimals)
        ax.text(float(value) + pad, y, label, ha="left", va="center", fontsize=8, fontweight="medium")
        xmax = max(xmax, float(value) + pad * 8)
    ax.set_xlim(0, min(1.15, xmax * 1.05) if xmax <= 1.1 else xmax * 1.08)


def _annotate_line_points(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    *,
    decimals: int = 3,
    offset: float = 0.02,
) -> None:
    ymax = ax.get_ylim()[1]
    for x, y in zip(xs, ys, strict=False):
        if pd.isna(y):
            continue
        ax.annotate(
            _format_value(float(y), decimals=decimals),
            (x, y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=7,
            fontweight="medium",
        )
        ymax = max(ymax, float(y) + offset * 3)
    ax.set_ylim(0, min(1.15, ymax * 1.05) if ymax <= 1.1 else ymax * 1.08)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grouped_metrics_bar(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    metrics: tuple[str, ...] = ("mrr", "map", "ndcg"),
    algorithms: list[str] | None = None,
    paper_targets: dict[str, dict[str, float]] | None = None,
    ylabel: str = "Score",
) -> Path:
    apply_style()
    df = summary.copy()
    if algorithms:
        df = df[df["algorithm"].isin(algorithms)]
    df = df.sort_values("algorithm")

    x = np.arange(len(df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.8), 6))

    for i, metric in enumerate(metrics):
        if metric not in df.columns:
            continue
        offset = (i - 1) * width
        bars = ax.bar(x + offset, df[metric], width, label=metric.upper(), alpha=0.9)
        _annotate_vertical_bars(ax, bars, decimals=3)
        if paper_targets and metric in paper_targets:
            for bar, algo in zip(bars, df["algorithm"], strict=False):
                target = paper_targets[metric].get(algo)
                if target is not None:
                    ax.hlines(
                        target,
                        bar.get_x() - width / 2,
                        bar.get_x() + width / 2,
                        colors="black",
                        linestyles="--",
                        linewidth=1,
                    )
                    ax.text(
                        bar.get_x() + width / 2,
                        target - 0.03,
                        f"T={_format_value(target)}",
                        ha="center",
                        va="top",
                        fontsize=7,
                        color="black",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    if ax.get_ylim()[1] <= 1.05:
        ax.set_ylim(0, 1.15)
    return _save(fig, output_path)


def plot_precision_at_k(
    detail: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    k_values: tuple[int, ...] = (1, 2, 3, 4, 5),
    algorithms: list[str] | None = None,
) -> Path:
    apply_style()
    df = detail.copy()
    if algorithms:
        df = df[df["algorithm"].isin(algorithms)]

    cols = [f"precision_at_{k}" for k in k_values if f"precision_at_{k}" in df.columns]
    if not cols:
        raise ValueError("No precision_at_k columns in metrics detail")

    agg = df.groupby("algorithm")[cols].mean().reset_index()
    if algorithms:
        agg = agg.set_index("algorithm").reindex(algorithms).reset_index()
        agg = agg.dropna(subset=cols, how="all")

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = list(k_values[: len(cols)])
    for _, row in agg.iterrows():
        algo = row["algorithm"]
        ys = [row[c] for c in cols]
        ax.plot(xs, ys, marker="o", linewidth=2, label=algo, color=color_for_algorithm(str(algo)))
        _annotate_line_points(ax, xs, ys, decimals=3)

    ax.set_xlabel("K")
    ax.set_ylabel("Precision@K")
    ax.set_title(title)
    ax.set_xticks(xs)
    if ax.get_ylim()[1] <= 1.05:
        ax.set_ylim(0, 1.15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save(fig, output_path)


def plot_paper_targets(
    summary: pd.DataFrame,
    paper_targets: dict[str, dict[str, float]],
    output_path: Path,
    *,
    title: str,
    algorithms: list[str],
    tolerance: float = 0.05,
) -> Path:
    apply_style()
    rows = []
    for algo in algorithms:
        cur = summary[summary["algorithm"] == algo]
        if cur.empty:
            continue
        for metric in ("mrr", "map", "ndcg"):
            targets = paper_targets.get(metric, {})
            target = targets.get(algo)
            if target is None:
                continue
            measured = float(cur[metric].iloc[0])
            rows.append(
                {
                    "algorithm": algo,
                    "metric": metric.upper(),
                    "measured": measured,
                    "target": target,
                    "delta": measured - target,
                    "pass": abs(measured - target) <= tolerance,
                }
            )
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No paper targets configured", ha="center", va="center")
        ax.axis("off")
        return _save(fig, output_path)

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.35)))
    y_pos = np.arange(len(df))
    colors = ["#55A868" if p else "#C44E52" for p in df["pass"]]
    bars = ax.barh(y_pos, df["measured"], color=colors, alpha=0.85, label="Measured")
    ax.scatter(df["target"], y_pos, color="black", marker="|", s=200, zorder=5, label="Paper target")
    _annotate_horizontal_bars(ax, df["measured"], y_pos, decimals=3)
    for target, y in zip(df["target"], y_pos, strict=False):
        ax.text(
            float(target) - 0.02,
            y,
            f"T={_format_value(float(target))}",
            ha="right",
            va="center",
            fontsize=7,
            color="black",
        )
    for measured, target, y in zip(df["measured"], df["target"], y_pos, strict=False):
        delta = measured - target
        sign = "+" if delta >= 0 else ""
        ax.text(
            1.02,
            y,
            f"Δ={sign}{delta:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=7,
            color="#333333",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{a} — {m}" for a, m in zip(df["algorithm"], df["metric"])])
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right")
    return _save(fig, output_path)


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 9))
    data = corr.astype(float)
    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticklabels(data.index)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r")

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    return _save(fig, output_path)


def plot_ttest_heatmap(
    ttests: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    metric: str = "mrr",
) -> Path:
    apply_style()
    df = ttests[ttests["metric"] == metric].copy()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No t-tests for {metric}", ha="center", va="center")
        ax.axis("off")
        return _save(fig, output_path)

    df["label"] = df["algorithm_a"] + "\nvs\n" + df["algorithm_b"]
    df["neg_log_p"] = -np.log10(df["p_value"].clip(lower=1e-12))
    df = df.sort_values("p_value")

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.5)))
    colors = ["#C44E52" if s else "#4C72B0" for s in df["significant"]]
    bars = ax.barh(df["label"], df["neg_log_p"], color=colors, alpha=0.9)
    _annotate_horizontal_bars(ax, df["neg_log_p"], np.arange(len(df)), decimals=3, pad=0.05)
    for y, p_val, sig in zip(range(len(df)), df["p_value"], df["significant"], strict=False):
        marker = "*" if sig else "ns"
        ax.text(
            1.02,
            y,
            f"p={p_val:.4g} ({marker})",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=8,
            fontweight="medium",
        )
    ax.axvline(-np.log10(0.05), color="black", linestyle="--", linewidth=1, label="p = 0.05")
    ax.set_xlabel("-log10(p-value)")
    ax.set_title(f"{title} ({metric.upper()})")
    ax.legend()
    return _save(fig, output_path)


def plot_legacy_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    version_prefix: str,
    metric: str = "mrr",
    limit: int = 12,
) -> Path:
    apply_style()
    col_cur = f"{metric}_{version_prefix}"
    col_leg = f"{metric}_legacy"
    if col_cur not in comparison.columns:
        raise ValueError(f"Missing column {col_cur}")

    df = comparison.dropna(subset=[col_leg]).head(limit).copy()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No legacy comparison data", ha="center", va="center")
        ax.axis("off")
        return _save(fig, output_path)

    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_cur = ax.bar(
        x - width / 2,
        df[col_cur],
        width,
        label=f"Reproduction ({version_prefix.upper()})",
        color="#4C72B0",
    )
    bars_leg = ax.bar(x + width / 2, df[col_leg], width, label="Legacy CSV", color="#937860")
    _annotate_vertical_bars(ax, bars_cur, decimals=3)
    _annotate_vertical_bars(ax, bars_leg, decimals=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"], rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend()
    if ax.get_ylim()[1] <= 1.05:
        ax.set_ylim(0, 1.15)
    return _save(fig, output_path)


def plot_ranking_shifts(
    shifts: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    x_col: str = "pca1_score",
) -> Path:
    apply_style()
    if shifts.empty or x_col not in shifts.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ranking shift data", ha="center", va="center")
        ax.axis("off")
        return _save(fig, output_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for base, grp in shifts.groupby("base_algorithm", sort=False):
        ax.scatter(
            grp[x_col],
            grp["ranking_difference"],
            alpha=0.5,
            s=30,
            label=str(base),
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    summary_lines = []
    for base, grp in shifts.groupby("base_algorithm", sort=False):
        mean_shift = grp["ranking_difference"].mean()
        summary_lines.append(f"{base}: mean Δrank={mean_shift:+.2f} (n={len(grp)})")
    ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel("Ranking difference (positive = moved up)")
    ax.set_title(title)
    ax.legend(title="Base algorithm")
    return _save(fig, output_path)


def plot_oracle_validation(
    oracle: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = oracle["metric"].tolist()
    values = oracle["value"].astype(float).tolist()
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"][: len(labels)]
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    _annotate_vertical_bars(ax, bars, decimals=4)
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, output_path)


def plot_distribution(
    series: pd.Series,
    output_path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 50,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    clean = series.dropna().astype(float)
    counts, edges, patches = ax.hist(clean, bins=bins, color="#4C72B0", alpha=0.85, edgecolor="white")
    label_threshold = max(counts.max() * 0.08, 1) if len(counts) else 1
    for count, patch in zip(counts, patches, strict=False):
        if count < label_threshold:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            count,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    mean_val = clean.mean()
    median_val = clean.median()
    ax.axvline(mean_val, color="#C44E52", linestyle="--", linewidth=1.2, label=f"mean={mean_val:.3f}")
    ax.axvline(median_val, color="#55A868", linestyle=":", linewidth=1.2, label=f"median={median_val:.3f}")
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    return _save(fig, output_path)
