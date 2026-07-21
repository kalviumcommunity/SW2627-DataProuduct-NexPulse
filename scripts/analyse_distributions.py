"""
analyse_distributions.py
------------------------
Distribution Analysis for Business Trends in the NexPulse workflow.

Responsibilities
----------------
1. Compute key distribution statistics for every numeric column:
      mean, median, mode, std, skewness, kurtosis, quartiles.
2. Interpret skewness and kurtosis with business meaning:
      high skewness → median > mean for reporting
      high kurtosis → expect extreme outliers
3. Generate histogram + KDE plots for each numeric column saved to output/.
4. Detect bimodal distributions (suggest hidden segmentation).
5. Compare distributions across business segments (e.g. high-value
   vs low-value customers; B2B vs B2C).
6. Produce a structured JSON report with statistics + business
   interpretations, plus visual outputs.

Usage
-----
    # Built-in demo dataset:
    python scripts/analyse_distributions.py --demo

    # Real CSV:
    python scripts/analyse_distributions.py --input data/processed/features.csv

    # Specify segment column for comparison:
    python scripts/analyse_distributions.py --demo --segment-by segment
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def build_demo_data(n: int = 5000) -> pd.DataFrame:
    """
    Return a customer dataset with intentionally varied distributions:
      - revenue: heavily right-skewed (most small, few large)
      - profit_margin: approximately normal
      - days_since_last_purchase: bimodal (active + churned)
      - support_tickets: left-skewed (most zero, few high)
    """
    rng = np.random.default_rng(seed=99)

    # Right-skewed revenue: exponential with a few whales
    revenue = np.concatenate([
        rng.exponential(scale=200, size=int(n * 0.9)) + 10,  # bulk: $10–$600
        rng.uniform(5000, 50000, size=int(n * 0.1)),         # whales: $5k–$50k
    ])
    rng.shuffle(revenue)

    # Approximately normal profit margin (%)
    profit_margin = rng.normal(loc=25, scale=8, size=n).clip(0, 100)

    # Bimodal recency: active (0–30 days) + churned (300–730 days)
    recency_active = rng.integers(0, 31, size=int(n * 0.6))
    recency_churned = rng.integers(300, 731, size=int(n * 0.4))
    days_since_last_purchase = np.concatenate([recency_active, recency_churned])
    rng.shuffle(days_since_last_purchase)

    # Left-skewed support tickets: most zero, few very high
    support_tickets = np.concatenate([
        np.zeros(int(n * 0.7)),
        rng.integers(1, 5, size=int(n * 0.25)),
        rng.integers(10, 50, size=int(n * 0.05)),
    ])
    rng.shuffle(support_tickets)

    segment = rng.choice(["B2C", "B2B", "Enterprise"], size=n, p=[0.6, 0.3, 0.1])

    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "segment": segment,
            "revenue": revenue.round(2),
            "profit_margin": profit_margin.round(2),
            "days_since_last_purchase": days_since_last_purchase.astype(float),
            "support_tickets": support_tickets.astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Step 1 – Compute distribution statistics
# ---------------------------------------------------------------------------

def compute_distribution_stats(series: pd.Series) -> dict:
    """
    Compute descriptive statistics for a single numeric column.

    Returns
    -------
    dict with keys: mean, median, mode, std, min, max, q25, q50, q75,
                    skewness, kurtosis, count, null_count.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return {"error": "all null"}

    mode_vals = clean.mode()
    mode = float(mode_vals.iloc[0]) if len(mode_vals) > 0 else np.nan

    return {
        "count": int(len(clean)),
        "null_count": int(series.isna().sum()),
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "mode": round(mode, 4) if not np.isnan(mode) else None,
        "std": round(float(clean.std()), 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "q25": round(float(clean.quantile(0.25)), 4),
        "q50": round(float(clean.quantile(0.50)), 4),
        "q75": round(float(clean.quantile(0.75)), 4),
        "skewness": round(float(stats.skew(clean, nan_policy="omit")), 4),
        "kurtosis": round(float(stats.kurtosis(clean, nan_policy="omit")), 4),
    }


# ---------------------------------------------------------------------------
# Step 2 – Interpret statistics with business meaning
# ---------------------------------------------------------------------------

def interpret_distribution(col: str, stats_dict: dict) -> dict:
    """
    Translate raw statistics into business insights.

    Skewness interpretation
    -----------------------
    |skew| < 0.5  : approximately symmetric — mean ≈ median, safe to use mean.
    0.5 ≤ |skew| < 1 : moderately skewed — prefer median for typical value.
    |skew| ≥ 1      : highly skewed — mean is misleading, always use median.

    Kurtosis interpretation
    ------------------------
    kurtosis < 0    : light tails — fewer outliers than normal distribution.
    0 ≤ kurtosis < 3: normal-like tails.
    kurtosis ≥ 3    : heavy tails — expect extreme outliers.

    Mean vs median gap
    ------------------
    Large gap → distribution is pulled by a tail; median more representative.
    """
    skew = stats_dict["skewness"]
    kurt = stats_dict["kurtosis"]
    mean = stats_dict["mean"]
    median = stats_dict["median"]

    # Skewness interpretation
    if abs(skew) < 0.5:
        skew_interp = "approximately symmetric"
        central_rec = "mean"
    elif abs(skew) < 1.0:
        skew_interp = "moderately skewed"
        central_rec = "median preferred"
    else:
        skew_interp = "highly skewed"
        central_rec = "median strongly preferred (mean is misleading)"

    skew_direction = "right" if skew > 0 else "left" if skew < 0 else "none"

    # Kurtosis interpretation
    if kurt < 0:
        kurt_interp = "light tails — fewer extreme outliers than normal"
    elif kurt < 3:
        kurt_interp = "normal-like tails"
    else:
        kurt_interp = "heavy tails — expect extreme outliers"

    # Mean vs median gap
    gap = abs(mean - median)
    gap_pct = (gap / abs(mean)) * 100 if mean != 0 else 0
    if gap_pct > 20:
        gap_msg = f"Mean ({mean:.2f}) differs {gap_pct:.1f}% from median ({median:.2f}). Distribution is pulled by a tail."
    elif gap_pct > 10:
        gap_msg = f"Mean ({mean:.2f}) and median ({median:.2f}) differ moderately ({gap_pct:.1f}%)."
    else:
        gap_msg = f"Mean ({mean:.2f}) ≈ median ({median:.2f}) — symmetric center."

    return {
        "column": col,
        "shape": skew_interp,
        "skew_direction": skew_direction,
        "recommended_central_tendency": central_rec,
        "tail_behavior": kurt_interp,
        "mean_vs_median": gap_msg,
        "business_insight": _business_insight(col, skew, kurt, mean, median),
    }


def _business_insight(col: str, skew: float, kurt: float, mean: float, median: float) -> str:
    """Generate a plain-English business interpretation."""
    insights = []

    if abs(skew) >= 1:
        if skew > 0:
            insights.append(
                f"Most {col} values are low, but a few extremely high values "
                f"pull the average up. Median ({median:.2f}) better represents "
                f"the typical value than mean ({mean:.2f})."
            )
        else:
            insights.append(
                f"Most {col} values are high, but a few extremely low values "
                f"pull the average down. Median ({median:.2f}) is more representative."
            )

    if kurt >= 3:
        insights.append(
            f"Expect extreme outliers in {col}. Consider capping or flagging "
            f"values beyond the 99th percentile for robustness."
        )

    if not insights:
        insights.append(
            f"{col} is relatively well-behaved: symmetric and no extreme tails. "
            f"Mean ({mean:.2f}) is a trustworthy summary."
        )

    return " ".join(insights)


# ---------------------------------------------------------------------------
# Step 3 – Detect bimodal distributions
# ---------------------------------------------------------------------------

def detect_bimodal(series: pd.Series, col: str) -> dict | None:
    """
    Heuristic bimodal detection: if the distribution has two clear peaks
    separated by a valley, flag it as potentially bimodal.

    Simple method: compute histogram, look for multiple local maxima with
    significant separation. Not foolproof but catches obvious cases like
    "active + churned" customer recency.

    Returns
    -------
    dict with "bimodal": True, "interpretation": str if detected, else None.
    """
    clean = series.dropna()
    if len(clean) < 100:
        return None

    hist, edges = np.histogram(clean, bins=30)
    # Find local maxima
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
            peaks.append((i, hist[i]))

    # Bimodal if we have 2+ peaks with significant height and separation
    if len(peaks) >= 2:
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
        idx1, h1 = peaks[0]
        idx2, h2 = peaks[1]
        separation = abs(idx1 - idx2)

        if separation >= 5 and min(h1, h2) / max(h1, h2) > 0.3:
            return {
                "bimodal": True,
                "interpretation": (
                    f"{col} shows a bimodal distribution with two distinct clusters. "
                    f"This suggests two different underlying populations or behaviors "
                    f"(e.g. active vs churned customers, small vs enterprise accounts). "
                    f"Consider segmenting the analysis."
                ),
            }

    return None


# ---------------------------------------------------------------------------
# Step 4 – Visualize: histogram + KDE
# ---------------------------------------------------------------------------

def plot_distribution(
    series: pd.Series,
    col: str,
    stats_dict: dict,
    output_dir: Path,
) -> Path:
    """
    Generate histogram + KDE overlay plot, annotate with mean/median/skew/kurtosis.

    Returns
    -------
    Path to the saved PNG file.
    """
    clean = series.dropna()
    if len(clean) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(clean, bins=50, alpha=0.6, color="steelblue", edgecolor="black", density=True, label="Histogram")

    # KDE overlay
    try:
        clean.plot(kind="density", ax=ax, color="darkorange", linewidth=2, label="KDE")
    except Exception:
        pass  # skip KDE if it fails (e.g. constant column)

    # Vertical lines for mean and median
    mean_val = stats_dict["mean"]
    median_val = stats_dict["median"]
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle="--", linewidth=2, label=f"Median = {median_val:.2f}")

    # Annotations
    skew = stats_dict["skewness"]
    kurt = stats_dict["kurtosis"]
    ax.text(
        0.98, 0.95,
        f"Skewness: {skew:.2f}\nKurtosis: {kurt:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Distribution: {col}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plot_path = output_dir / f"distribution_{col}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


# ---------------------------------------------------------------------------
# Step 5 – Segment comparison
# ---------------------------------------------------------------------------

def compare_segments(
    df: pd.DataFrame,
    segment_col: str,
    numeric_cols: list[str],
    output_dir: Path,
) -> dict:
    """
    Compare distributions of numeric columns across segments.

    For each numeric column, plot overlapping histograms per segment and
    report mean/median per segment.

    Returns
    -------
    dict keyed by numeric column, containing per-segment stats.
    """
    if segment_col not in df.columns:
        return {"error": f"Segment column '{segment_col}' not found."}

    segments = df[segment_col].dropna().unique()
    comparison: dict = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue

        segment_stats = {}
        fig, ax = plt.subplots(figsize=(10, 6))

        for seg in segments:
            subset = df[df[segment_col] == seg][col].dropna()
            if len(subset) == 0:
                continue

            segment_stats[str(seg)] = {
                "mean": round(float(subset.mean()), 4),
                "median": round(float(subset.median()), 4),
                "count": int(len(subset)),
            }

            ax.hist(
                subset,
                bins=30,
                alpha=0.5,
                label=f"{seg} (n={len(subset)})",
                edgecolor="black",
            )

        ax.set_xlabel(col, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{col} by {segment_col}", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        plot_path = output_dir / f"segment_comparison_{col}_by_{segment_col}.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        comparison[col] = {
            "segment_stats": segment_stats,
            "plot": str(plot_path.name),
        }

    return comparison


# ---------------------------------------------------------------------------
# Step 6 – Orchestrate full analysis
# ---------------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
    segment_col: str | None = None,
) -> tuple[dict, Path]:
    """
    Run the complete distribution analysis pipeline.

    Returns
    -------
    (report_dict, output_dir)
    """
    output_dir = Path("output/distribution_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n[run_analysis] Analyzing {len(numeric_cols)} numeric columns: {numeric_cols}\n")

    report: dict = {"columns": {}}

    for col in numeric_cols:
        print(f"  [{col}]")

        # Statistics
        stats_dict = compute_distribution_stats(df[col])
        if "error" in stats_dict:
            print(f"    skipped: {stats_dict['error']}")
            continue

        # Interpretation
        interp = interpret_distribution(col, stats_dict)

        # Bimodal detection
        bimodal = detect_bimodal(df[col], col)

        # Plot
        plot_path = plot_distribution(df[col], col, stats_dict, output_dir)

        report["columns"][col] = {
            "statistics": stats_dict,
            "interpretation": interp,
            "bimodal_detection": bimodal,
            "plot": str(plot_path.name) if plot_path else None,
        }

        print(f"    shape: {interp['shape']}  skew={stats_dict['skewness']:.2f}  "
              f"kurt={stats_dict['kurtosis']:.2f}")
        if bimodal:
            print(f"    ⚠ bimodal detected")

    # Segment comparison
    if segment_col and segment_col in df.columns:
        print(f"\n[segment comparison] by '{segment_col}' …")
        comparison = compare_segments(df, segment_col, numeric_cols, output_dir)
        report["segment_comparison"] = {
            "segment_column": segment_col,
            "comparisons": comparison,
        }
    else:
        report["segment_comparison"] = None

    return report, output_dir


# ---------------------------------------------------------------------------
# Step 7 – Console report
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    sep = "=" * 65

    print(f"\n{sep}")
    print("DISTRIBUTION ANALYSIS — BUSINESS TRENDS REPORT")
    print(sep)

    cols_analyzed = len(report["columns"])
    print(f"\n  Columns analyzed : {cols_analyzed}")

    for col, data in report["columns"].items():
        stats = data["statistics"]
        interp = data["interpretation"]
        bimodal = data["bimodal_detection"]

        print(f"\n  ── {col.upper()} ──")
        print(f"    Count       : {stats['count']:,}   Nulls: {stats['null_count']}")
        print(f"    Mean        : {stats['mean']:.2f}")
        print(f"    Median      : {stats['median']:.2f}   (Q25={stats['q25']:.2f}, Q75={stats['q75']:.2f})")
        print(f"    Std         : {stats['std']:.2f}")
        print(f"    Range       : [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"    Skewness    : {stats['skewness']:.2f}  → {interp['shape']}")
        print(f"    Kurtosis    : {stats['kurtosis']:.2f}  → {interp['tail_behavior']}")
        print(f"    Recommend   : Use {interp['recommended_central_tendency']} for central tendency")
        print(f"    Insight     : {interp['business_insight']}")
        if bimodal:
            print(f"    ⚠ Bimodal   : {bimodal['interpretation']}")
        print(f"    Plot saved  : {data['plot']}")

    if report.get("segment_comparison"):
        sc = report["segment_comparison"]
        print(f"\n  ── SEGMENT COMPARISON BY {sc['segment_column'].upper()} ──")
        for col, comp in sc["comparisons"].items():
            print(f"\n    {col}:")
            for seg, seg_stats in comp["segment_stats"].items():
                print(f"      {seg:<15} mean={seg_stats['mean']:>10.2f}  "
                      f"median={seg_stats['median']:>10.2f}  n={seg_stats['count']:>6,}")
            print(f"      plot: {comp['plot']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Step 8 – Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(report: dict, output_dir: Path) -> None:
    """
    Write JSON report to output/distribution_analysis/.

    Files created
    -------------
    output/distribution_analysis/report.json            — full analysis report
    output/distribution_analysis/distribution_*.png     — per-column plots
    output/distribution_analysis/segment_comparison_*.png — segment plots
    """
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)
    print(f"[save_outputs] Report saved → {report_path}")
    print(f"[save_outputs] Plots saved  → {output_dir}/*.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distribution Analysis for Business Trends — NexPulse"
    )
    parser.add_argument(
        "--input", dest="input_path",
        help="Path to a CSV file. Omit to use the built-in demo dataset.",
    )
    parser.add_argument(
        "--segment-by", dest="segment_col",
        help="Column name to use for segment comparison (e.g. 'segment', 'country').",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Force the built-in demo dataset.",
    )
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.demo or not args.input_path:
        print("Using built-in demo dataset (5,000 rows).")
        return build_demo_data(), "demo dataset"

    path = Path(args.input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded CSV: {path}  ({len(df):,} rows, {len(df.columns)} cols)")
    return df, str(path)


def main() -> None:
    args = parse_args()

    # 1. Load
    df, source_label = load_input(args)
    print(f"\nSource : {source_label}")
    print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Cols   : {list(df.columns)}")

    # 2. Run analysis
    report, output_dir = run_analysis(df, segment_col=args.segment_col)

    # 3. Console report
    print_report(report)

    # 4. Persist
    save_outputs(report, output_dir)

    print(f"\nDistribution analysis complete.")
    print(f"  Columns analyzed     : {len(report['columns'])}")
    print(f"  Plots generated      : {len([v for v in report['columns'].values() if v.get('plot')])}")
    if report.get("segment_comparison"):
        sc_count = len(report["segment_comparison"]["comparisons"])
        print(f"  Segment comparisons  : {sc_count}")


if __name__ == "__main__":
    main()
