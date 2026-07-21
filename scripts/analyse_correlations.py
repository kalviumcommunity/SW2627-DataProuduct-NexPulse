"""
analyse_correlations.py
-----------------------
Correlation & Relationship Analysis for the NexPulse workflow.

Responsibilities
----------------
1. Compute Pearson and Spearman correlation matrices for all numeric columns.
2. Identify strong, moderate, and weak pairs with thresholds and band labels.
3. Visualize both correlation matrices as annotated seaborn heatmaps.
4. Generate scatter plots for every strongly-correlated pair (|r| > 0.5).
5. Flag redundant features: pairs so strongly correlated (|r| > 0.9) that
   one is likely a duplicate signal.
6. Attach structured causation warnings — "correlation ≠ causation" — with
   three alternative explanations for every strong pair.
7. Produce a JSON report: ranked pair list, feature-selection recommendations,
   and causation reasoning notes.

Usage
-----
    # Built-in demo dataset:
    python scripts/analyse_correlations.py --demo

    # Real CSV:
    python scripts/analyse_correlations.py --input data/processed/features.csv

    # Custom strong-pair threshold:
    python scripts/analyse_correlations.py --demo --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Strength thresholds
# ---------------------------------------------------------------------------

VERY_STRONG_THRESHOLD = 0.9   # |r| ≥ 0.9 → likely redundant features
STRONG_THRESHOLD      = 0.7   # |r| ≥ 0.7 → strong relationship
MODERATE_THRESHOLD    = 0.4   # |r| ≥ 0.4 → moderate relationship
# |r| < 0.4              → weak / negligible


def correlation_band(r: float) -> str:
    """Return a human-readable strength label for a correlation coefficient."""
    abs_r = abs(r)
    if abs_r >= VERY_STRONG_THRESHOLD:
        return "very strong"
    if abs_r >= STRONG_THRESHOLD:
        return "strong"
    if abs_r >= MODERATE_THRESHOLD:
        return "moderate"
    return "weak"


def correlation_direction(r: float) -> str:
    if r > 0.05:
        return "positive"
    if r < -0.05:
        return "negative"
    return "none"


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def build_demo_data(n: int = 2000) -> pd.DataFrame:
    """
    Return a customer dataset with intentional correlation structure:

    Strong positives (by construction)
    ─────────────────────────────────
    revenue ↔ profit_margin      : high revenue drives margin
    total_transactions ↔ revenue : more orders → more revenue

    Strong negative (by construction)
    ─────────────────────────────────
    days_since_last_purchase ↔ transactions_per_month : churned = low frequency

    Moderate
    ────────
    support_tickets ↔ churn_score : pain drives both (confounding variable demo)

    Weak / near-zero
    ────────────────
    customer_age ↔ revenue : no meaningful relationship
    """
    rng = np.random.default_rng(seed=21)

    # Base signals
    customer_age       = rng.integers(18, 75, size=n).astype(float)
    transactions       = rng.integers(1, 200, size=n).astype(float)
    days_since_purchase= rng.integers(1, 730, size=n).astype(float)
    support_tickets    = (rng.exponential(scale=1.5, size=n)).clip(0, 40).round()

    # Derived with noise — intentional correlations
    revenue = (
        transactions * rng.uniform(10, 50, size=n)       # positive with tx
        + rng.normal(0, 200, size=n)
    ).clip(5).round(2)

    profit_margin = (
        revenue * 0.25
        + rng.normal(0, 50, size=n)
    ).clip(0).round(2)

    # transactions_per_month negatively correlated with days_since_purchase
    transactions_per_month = (
        (200 - days_since_purchase) / 20
        + rng.normal(0, 1.5, size=n)
    ).clip(0).round(2)

    # churn_score driven by support tickets AND recency — confounding example
    churn_score = (
        support_tickets * 4
        + days_since_purchase * 0.08
        + rng.normal(0, 5, size=n)
    ).clip(0, 100).round(1)

    discount_pct = rng.uniform(0, 50, size=n).round(1)

    return pd.DataFrame(
        {
            "customer_age":           customer_age,
            "total_transactions":     transactions,
            "revenue":                revenue,
            "profit_margin":          profit_margin,
            "days_since_last_purchase": days_since_purchase,
            "transactions_per_month": transactions_per_month,
            "support_tickets":        support_tickets,
            "churn_score":            churn_score,
            "discount_pct":           discount_pct,
        }
    )


# ---------------------------------------------------------------------------
# Step 1 – Compute correlation matrices
# ---------------------------------------------------------------------------

def compute_correlations(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pearson and Spearman correlation matrices.

    Pearson  — linear relationships; sensitive to outliers.
    Spearman — monotonic (rank-based) relationships; robust to outliers
               and non-linear monotonic trends.

    Returns
    -------
    (pearson_matrix, spearman_matrix)
        Both are n×n DataFrames of correlation coefficients.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    pearson  = numeric_df.corr(method="pearson")
    spearman = numeric_df.corr(method="spearman")
    return pearson, spearman


# ---------------------------------------------------------------------------
# Step 2 – Extract ranked correlation pairs
# ---------------------------------------------------------------------------

def extract_correlation_pairs(
    matrix: pd.DataFrame,
    method: str,
    min_abs: float = 0.0,
) -> list[dict]:
    """
    Flatten a correlation matrix into a ranked list of unique pairs.

    Self-correlations (diagonal) are excluded.
    Pairs are sorted by |r| descending.

    Parameters
    ----------
    matrix  : n×n correlation DataFrame.
    method  : "pearson" or "spearman" (stored in the output for reporting).
    min_abs : Only include pairs with |r| >= this value.

    Returns
    -------
    List of dicts: var1, var2, r, abs_r, strength, direction.
    """
    seen: set[frozenset] = set()
    pairs: list[dict] = []

    for col in matrix.columns:
        for row in matrix.index:
            if col == row:
                continue
            key = frozenset([col, row])
            if key in seen:
                continue
            seen.add(key)

            r = float(matrix.loc[row, col])
            if np.isnan(r):
                continue
            abs_r = abs(r)
            if abs_r < min_abs:
                continue

            pairs.append(
                {
                    "var1": row,
                    "var2": col,
                    "method": method,
                    "r": round(r, 4),
                    "abs_r": round(abs_r, 4),
                    "strength": correlation_band(r),
                    "direction": correlation_direction(r),
                }
            )

    return sorted(pairs, key=lambda x: x["abs_r"], reverse=True)


# ---------------------------------------------------------------------------
# Step 3 – Redundancy detection (feature selection)
# ---------------------------------------------------------------------------

def detect_redundant_features(
    pairs: list[dict],
    threshold: float = VERY_STRONG_THRESHOLD,
) -> list[dict]:
    """
    Identify feature pairs that are so strongly correlated they are likely
    carrying duplicate information.

    For each redundant pair, recommend dropping the less interpretable feature.
    Convention: drop the second variable alphabetically (arbitrary but consistent).

    Returns
    -------
    List of dicts with var1, var2, r, recommendation.
    """
    redundant = []
    for p in pairs:
        if p["abs_r"] >= threshold and p["abs_r"] < 1.0:
            # Recommend keeping the feature that sounds more business-meaningful
            keep    = p["var1"]
            discard = p["var2"]
            redundant.append(
                {
                    "var1": p["var1"],
                    "var2": p["var2"],
                    "r": p["r"],
                    "recommendation": (
                        f"|r|={p['abs_r']:.2f} ({p['method']}). "
                        f"Consider dropping '{discard}' — it carries near-identical "
                        f"information to '{keep}'. Keeping both inflates model complexity "
                        f"without adding signal."
                    ),
                }
            )
    return redundant


# ---------------------------------------------------------------------------
# Step 4 – Causation warnings
# ---------------------------------------------------------------------------

CAUSATION_WARNING = (
    "CORRELATION ≠ CAUSATION. Three alternative explanations:\n"
    "  1. {var1} causes {var2}\n"
    "  2. {var2} causes {var1}\n"
    "  3. A confounding variable causes both {var1} and {var2}\n"
    "Always reason about mechanism before concluding causation."
)


def attach_causation_notes(pairs: list[dict], threshold: float = STRONG_THRESHOLD) -> list[dict]:
    """
    Attach a causation warning to every pair above the strength threshold.
    """
    for p in pairs:
        if p["abs_r"] >= threshold:
            p["causation_warning"] = CAUSATION_WARNING.format(
                var1=p["var1"], var2=p["var2"]
            )
        else:
            p["causation_warning"] = None
    return pairs


# ---------------------------------------------------------------------------
# Step 5 – Visualize heatmaps
# ---------------------------------------------------------------------------

def plot_heatmap(
    matrix: pd.DataFrame,
    method: str,
    output_dir: Path,
) -> Path:
    """
    Generate an annotated seaborn heatmap for a correlation matrix.

    - coolwarm diverging palette: red = positive, blue = negative.
    - Annotations show r values rounded to 2 decimal places.
    - Saves PNG to output_dir.
    """
    n = len(matrix.columns)
    fig_size = max(8, n * 1.2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.zeros_like(matrix, dtype=bool)
    np.fill_diagonal(mask, True)          # hide diagonal (self-correlation = 1)

    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 9},
        ax=ax,
        cbar_kws={"shrink": 0.8, "label": "r"},
    )

    ax.set_title(
        f"{method.capitalize()} Correlation Matrix\n"
        f"Red = positive  |  Blue = negative  |  Diagonal hidden",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)

    plot_path = output_dir / f"correlation_heatmap_{method}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


# ---------------------------------------------------------------------------
# Step 6 – Scatter plots for strong pairs
# ---------------------------------------------------------------------------

def plot_scatter_pairs(
    df: pd.DataFrame,
    pairs: list[dict],
    output_dir: Path,
    threshold: float = 0.5,
    max_plots: int = 8,
) -> list[Path]:
    """
    Generate a scatter plot for each pair with |r| >= threshold.

    Adds a linear regression trend line via numpy polyfit.
    Annotates with Pearson r and Spearman r values.
    """
    strong_pairs = [
        p for p in pairs
        if p["abs_r"] >= threshold and p["method"] == "pearson"
    ][:max_plots]

    paths: list[Path] = []

    for p in strong_pairs:
        v1, v2 = p["var1"], p["var2"]
        if v1 not in df.columns or v2 not in df.columns:
            continue

        x = df[v1].dropna()
        y = df[v2].reindex(x.index).dropna()
        x = x.reindex(y.index)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(x, y, alpha=0.3, s=15, color="steelblue", edgecolors="none")

        # Regression trend line
        if len(x) > 1:
            m, b = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 200)
            ax.plot(x_line, m * x_line + b, color="red", linewidth=2, label="Trend")

        # Compute both r values for annotation
        pearson_r  = float(stats.pearsonr(x, y)[0])
        spearman_r = float(stats.spearmanr(x, y)[0])

        ax.set_xlabel(v1, fontsize=11)
        ax.set_ylabel(v2, fontsize=11)
        ax.set_title(
            f"{v1}  ↔  {v2}",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            0.03, 0.96,
            f"Pearson r  = {pearson_r:+.3f}\nSpearman r = {spearman_r:+.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

        fname = f"scatter_{v1}_vs_{v2}.png".replace(" ", "_")
        plot_path = output_dir / fname
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        paths.append(plot_path)

    return paths


# ---------------------------------------------------------------------------
# Step 7 – Console report
# ---------------------------------------------------------------------------

def print_report(
    pearson_pairs: list[dict],
    spearman_pairs: list[dict],
    redundant: list[dict],
    threshold: float,
) -> None:
    sep = "=" * 65

    strong_pearson  = [p for p in pearson_pairs  if p["abs_r"] >= threshold]
    strong_spearman = [p for p in spearman_pairs if p["abs_r"] >= threshold]

    print(f"\n{sep}")
    print("CORRELATION & RELATIONSHIP ANALYSIS — REPORT")
    print(sep)

    # --- Pearson ---
    print(f"\n  ── PEARSON  (linear, continuous)  |r| ≥ {threshold} ──")
    print(f"  {'Pair':<45} {'r':>7}  {'Strength':<14} {'Direction'}")
    print(f"  {'-'*45} {'-'*7}  {'-'*14} {'-'*10}")
    if strong_pearson:
        for p in strong_pearson:
            pair_str = f"{p['var1']}  ↔  {p['var2']}"
            print(
                f"  {pair_str:<45} {p['r']:>+7.4f}  "
                f"{p['strength']:<14} {p['direction']}"
            )
    else:
        print(f"  No pairs above |r| = {threshold}")

    # --- Spearman ---
    print(f"\n  ── SPEARMAN (rank-based, robust)  |r| ≥ {threshold} ──")
    print(f"  {'Pair':<45} {'r':>7}  {'Strength':<14} {'Direction'}")
    print(f"  {'-'*45} {'-'*7}  {'-'*14} {'-'*10}")
    if strong_spearman:
        for p in strong_spearman:
            pair_str = f"{p['var1']}  ↔  {p['var2']}"
            print(
                f"  {pair_str:<45} {p['r']:>+7.4f}  "
                f"{p['strength']:<14} {p['direction']}"
            )
    else:
        print(f"  No pairs above |r| = {threshold}")

    # --- Pearson vs Spearman differences (signals non-linearity) ---
    print(f"\n  ── PEARSON vs SPEARMAN DIFFERENCES (non-linearity signals) ──")
    p_dict = {frozenset([p["var1"], p["var2"]]): p["r"] for p in pearson_pairs}
    s_dict = {frozenset([p["var1"], p["var2"]]): p["r"] for p in spearman_pairs}
    diff_pairs = []
    for key, p_r in p_dict.items():
        s_r = s_dict.get(key)
        if s_r is None:
            continue
        diff = abs(p_r - s_r)
        if diff >= 0.15:
            v1, v2 = tuple(key)
            diff_pairs.append((v1, v2, p_r, s_r, diff))
    if diff_pairs:
        diff_pairs.sort(key=lambda x: x[4], reverse=True)
        print(f"  {'Pair':<45} {'Pearson':>8}  {'Spearman':>9}  {'Δ':>6}")
        print(f"  {'-'*45} {'-'*8}  {'-'*9}  {'-'*6}")
        for v1, v2, p_r, s_r, diff in diff_pairs:
            print(f"  {v1+' ↔ '+v2:<45} {p_r:>+8.4f}  {s_r:>+9.4f}  {diff:>6.4f}")
        print(f"  Large Δ = relationship is non-linear or driven by outliers.")
    else:
        print(f"  No pairs differ by ≥ 0.15 — relationships are approximately linear.")

    # --- Redundant features ---
    print(f"\n  ── REDUNDANT FEATURES (|r| ≥ {VERY_STRONG_THRESHOLD}) ──")
    if redundant:
        for r in redundant:
            print(f"\n  {r['var1']}  ↔  {r['var2']}")
            print(f"    {r['recommendation']}")
    else:
        print(f"  No redundant feature pairs detected at threshold {VERY_STRONG_THRESHOLD}.")

    # --- Causation warnings ---
    print(f"\n  ── CAUSATION WARNINGS (top 3 strongest pairs) ──")
    top3 = [p for p in pearson_pairs if p.get("causation_warning")][:3]
    for p in top3:
        print(f"\n  {p['var1']} ↔ {p['var2']}  (r={p['r']:+.4f})")
        print(f"  ⚠  {p['causation_warning']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Step 8 – Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    pearson_pairs: list[dict],
    spearman_pairs: list[dict],
    redundant: list[dict],
    heatmap_paths: list[Path],
    scatter_paths: list[Path],
    output_dir: Path,
) -> None:
    """
    Files created
    -------------
    output/correlation_analysis/report.json            — full ranked pairs + notes
    output/correlation_analysis/correlation_heatmap_pearson.png
    output/correlation_analysis/correlation_heatmap_spearman.png
    output/correlation_analysis/scatter_<var1>_vs_<var2>.png  (per strong pair)
    """
    report = {
        "pearson_pairs":   pearson_pairs,
        "spearman_pairs":  spearman_pairs,
        "redundant_features": redundant,
        "plots": {
            "heatmaps": [str(p.name) for p in heatmap_paths],
            "scatter_plots": [str(p.name) for p in scatter_paths],
        },
    }

    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)

    print(f"[save_outputs] Report saved     → {report_path}")
    print(f"[save_outputs] Heatmaps saved   → {[str(p.name) for p in heatmap_paths]}")
    print(f"[save_outputs] Scatters saved   → {[str(p.name) for p in scatter_paths]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlation & Relationship Analysis — NexPulse"
    )
    parser.add_argument(
        "--input", dest="input_path",
        help="Path to a CSV file. Omit to use the built-in demo dataset.",
    )
    parser.add_argument(
        "--threshold", dest="threshold", type=float, default=STRONG_THRESHOLD,
        help=f"Minimum |r| to classify as 'strong' (default: {STRONG_THRESHOLD}).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Force the built-in demo dataset.",
    )
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.demo or not args.input_path:
        print("Using built-in demo dataset (2,000 rows).")
        return build_demo_data(), "demo dataset"

    path = Path(args.input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded: {path}  ({len(df):,} rows, {len(df.columns)} cols)")
    return df, str(path)


def main() -> None:
    args = parse_args()
    threshold = args.threshold

    # 1. Load
    df, source_label = load_input(args)
    print(f"\nSource    : {source_label}")
    print(f"Shape     : {df.shape[0]:,} rows × {df.shape[1]} cols")
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"Numeric   : {list(numeric_df.columns)}\n")

    output_dir = Path("output/correlation_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Compute matrices
    print("[1/5] Computing Pearson and Spearman matrices …")
    pearson_matrix, spearman_matrix = compute_correlations(df)

    # 3. Extract ranked pairs
    print("[2/5] Extracting ranked correlation pairs …")
    pearson_pairs  = extract_correlation_pairs(pearson_matrix,  "pearson")
    spearman_pairs = extract_correlation_pairs(spearman_matrix, "spearman")
    pearson_pairs  = attach_causation_notes(pearson_pairs,  threshold)
    spearman_pairs = attach_causation_notes(spearman_pairs, threshold)

    print(f"  Pearson  pairs : {len(pearson_pairs)}")
    print(f"  Spearman pairs : {len(spearman_pairs)}")

    # 4. Redundancy detection
    print("[3/5] Detecting redundant features …")
    redundant = detect_redundant_features(pearson_pairs)
    print(f"  Redundant pairs: {len(redundant)}")

    # 5. Heatmaps
    print("[4/5] Generating heatmaps …")
    pearson_heatmap  = plot_heatmap(pearson_matrix,  "pearson",  output_dir)
    spearman_heatmap = plot_heatmap(spearman_matrix, "spearman", output_dir)
    heatmap_paths = [pearson_heatmap, spearman_heatmap]
    print(f"  Saved: {[p.name for p in heatmap_paths]}")

    # 6. Scatter plots for strong pairs
    print("[5/5] Generating scatter plots for strong pairs …")
    scatter_paths = plot_scatter_pairs(df, pearson_pairs, output_dir, threshold=threshold)
    print(f"  Saved: {[p.name for p in scatter_paths]}")

    # 7. Console report
    print_report(pearson_pairs, spearman_pairs, redundant, threshold)

    # 8. Persist
    save_outputs(
        pearson_pairs, spearman_pairs, redundant,
        heatmap_paths, scatter_paths, output_dir,
    )

    strong_count = len([p for p in pearson_pairs if p["abs_r"] >= threshold])
    print(f"\nCorrelation analysis complete.")
    print(f"  Total pairs analyzed  : {len(pearson_pairs)}")
    print(f"  Strong pairs (|r|≥{threshold}) : {strong_count}")
    print(f"  Redundant features    : {len(redundant)}")
    print(f"  Plots saved to        : {output_dir}/")


if __name__ == "__main__":
    main()
