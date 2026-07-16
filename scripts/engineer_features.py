"""
engineer_features.py
--------------------
Feature Engineering & Derived Business Columns for the NexPulse workflow.

Responsibilities
----------------
1. Ratio features  — normalise raw counts/totals by time or volume so
                     values carry context (transactions per month, spend
                     per transaction, lifetime value per month).
2. Binned features — convert continuous ratios into labelled tiers using
                     pd.cut  (equal-width, business-defined boundaries) and
                     pd.qcut (equal-frequency / quantile-based).
3. Composite scores — combine multiple signals into a single interpretable
                      metric (RFM score: Recency × Frequency × Monetary).
4. Distribution validation — verify every engineered column has no
                      unexpected nulls and that numeric ranges are sane.
5. Feature report  — structured JSON documenting each derived column:
                     formula, type, value distribution, business meaning.

Usage
-----
    # Built-in demo dataset:
    python scripts/engineer_features.py --demo

    # Real CSV:
    python scripts/engineer_features.py --input data/processed/merged.csv

    # Custom output:
    python scripts/engineer_features.py --demo --output output/features.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def build_demo_data() -> pd.DataFrame:
    """
    Return a synthetic customer dataset designed to exercise all three
    feature-engineering patterns (ratio, binned, composite).

    Intentional edge cases
    ----------------------
    - One customer with days_as_customer = 0  → division-guard needed.
    - One customer with total_transactions = 0 → division-guard needed.
    - Wide spread in recency, frequency, and monetary to exercise all
      RFM quantile bins.
    """
    rng = np.random.default_rng(seed=7)
    n = 50

    days_as_customer = rng.integers(30, 1500, size=n).astype(float)
    days_as_customer[4] = 0          # edge case: brand-new customer

    total_transactions = rng.integers(1, 200, size=n).astype(float)
    total_transactions[11] = 0       # edge case: customer with no completed orders

    total_spent = (rng.exponential(scale=300, size=n) + 20).round(2)
    total_spent[11] = 0.0            # consistent with zero transactions

    days_since_last_purchase = rng.integers(1, 730, size=n).astype(float)
    days_since_last_purchase[4] = 730  # churned new customer

    return pd.DataFrame(
        {
            "customer_id": range(1001, 1001 + n),
            "segment": rng.choice(["B2C", "B2B", "Enterprise"], size=n),
            "days_as_customer": days_as_customer,
            "total_transactions": total_transactions,
            "total_spent": total_spent,
            "days_since_last_purchase": days_since_last_purchase,
            "total_returns": rng.integers(0, 15, size=n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    fill_value: float = 0.0,
) -> pd.Series:
    """
    Divide two Series element-wise, replacing division-by-zero with
    *fill_value* rather than raising or producing inf/NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(denominator == 0, fill_value, numerator / denominator)
    return pd.Series(result, index=numerator.index)


# ---------------------------------------------------------------------------
# Step 1 – Ratio features
# ---------------------------------------------------------------------------

def add_ratio_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Derive normalised ratio features that add context to raw counts.

    New columns
    -----------
    transactions_per_month    : total_transactions / (days_as_customer / 30)
                                Captures engagement rate, not just volume.
    avg_spend_per_transaction : total_spent / total_transactions
                                Captures average basket size.
    lifetime_value_per_month  : total_spent / (days_as_customer / 30)
                                Revenue rate per month of tenure.
    return_rate               : total_returns / total_transactions
                                Proportion of orders that were returned.
    """
    result = df.copy()
    log: list[dict] = []

    months_active = _safe_divide(result["days_as_customer"], pd.Series(30.0, index=result.index))

    # transactions per month
    result["transactions_per_month"] = _safe_divide(
        result["total_transactions"], months_active
    ).round(4)
    log.append({
        "column": "transactions_per_month",
        "type": "ratio",
        "formula": "total_transactions / (days_as_customer / 30)",
        "business_meaning": (
            "Engagement rate normalised by tenure. "
            "50 transactions over 5 years differs greatly from 50 in one month."
        ),
    })

    # avg spend per transaction
    result["avg_spend_per_transaction"] = _safe_divide(
        result["total_spent"], result["total_transactions"]
    ).round(4)
    log.append({
        "column": "avg_spend_per_transaction",
        "type": "ratio",
        "formula": "total_spent / total_transactions",
        "business_meaning": (
            "Average basket size. Distinguishes a high-frequency low-value "
            "buyer from a low-frequency high-value buyer."
        ),
    })

    # lifetime value per month
    result["lifetime_value_per_month"] = _safe_divide(
        result["total_spent"], months_active
    ).round(4)
    log.append({
        "column": "lifetime_value_per_month",
        "type": "ratio",
        "formula": "total_spent / (days_as_customer / 30)",
        "business_meaning": (
            "Revenue rate per active month. Comparable across customers "
            "with different tenure lengths."
        ),
    })

    # return rate
    result["return_rate"] = _safe_divide(
        result["total_returns"], result["total_transactions"]
    ).round(4)
    log.append({
        "column": "return_rate",
        "type": "ratio",
        "formula": "total_returns / total_transactions",
        "business_meaning": (
            "Proportion of orders returned. High return rate signals "
            "dissatisfaction or mis-matched product expectations."
        ),
    })

    print(f"[ratio features]  added {len(log)} columns: "
          f"{[e['column'] for e in log]}")

    return result, log


# ---------------------------------------------------------------------------
# Step 2 – Binned / tiered features
# ---------------------------------------------------------------------------

def add_binned_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Convert continuous ratio features into labelled categorical tiers.

    Equal-width bins (pd.cut) — business-defined fixed boundaries
    ---------------------------------------------------------------
    engagement_tier        : based on transactions_per_month
        low    = 0 – 2 tx/month
        medium = 2 – 10 tx/month
        high   = > 10 tx/month

    return_risk_tier       : based on return_rate
        low    = 0 – 10 %
        medium = 10 – 25 %
        high   = > 25 %

    Equal-frequency bins (pd.qcut) — quartile-based
    ------------------------------------------------
    spend_quartile         : total_spent split into four equal-frequency buckets
        tier_1 (lowest 25 %) … tier_4 (top 25 %)

    ltv_per_month_quartile : lifetime_value_per_month split into four quartiles
    """
    result = df.copy()
    log: list[dict] = []

    # ── engagement_tier (pd.cut — fixed business boundaries) ─────────────
    result["engagement_tier"] = pd.cut(
        result["transactions_per_month"],
        bins=[0, 2, 10, float("inf")],
        labels=["low", "medium", "high"],
        right=True,
        include_lowest=True,
    )
    log.append({
        "column": "engagement_tier",
        "type": "binned_equal_width",
        "source": "transactions_per_month",
        "boundaries": [0, 2, 10, "inf"],
        "labels": ["low", "medium", "high"],
        "business_meaning": (
            "Actionable engagement segment. 'high' (>10 tx/month) are power users. "
            "'low' (<2 tx/month) are at risk of churn. "
            "Drives different retention strategies per tier."
        ),
    })

    # ── return_risk_tier (pd.cut — fixed boundaries) ─────────────────────
    result["return_risk_tier"] = pd.cut(
        result["return_rate"],
        bins=[-0.001, 0.10, 0.25, float("inf")],
        labels=["low", "medium", "high"],
        right=True,
    )
    log.append({
        "column": "return_risk_tier",
        "type": "binned_equal_width",
        "source": "return_rate",
        "boundaries": [0, 0.10, 0.25, "inf"],
        "labels": ["low", "medium", "high"],
        "business_meaning": (
            "Return risk segment. 'high' (>25 % return rate) customers are "
            "costly to serve and may indicate product-fit issues."
        ),
    })

    # ── spend_quartile (pd.qcut — equal-frequency) ───────────────────────
    # duplicates="drop" handles ties at quantile edges gracefully
    result["spend_quartile"] = pd.qcut(
        result["total_spent"],
        q=4,
        labels=["tier_1", "tier_2", "tier_3", "tier_4"],
        duplicates="drop",
    )
    log.append({
        "column": "spend_quartile",
        "type": "binned_quantile",
        "source": "total_spent",
        "q": 4,
        "labels": ["tier_1", "tier_2", "tier_3", "tier_4"],
        "business_meaning": (
            "Equal-frequency spend segments. tier_4 is the top 25 % spenders. "
            "Quartile-based so each tier has roughly the same customer count, "
            "enabling balanced segment analysis."
        ),
    })

    # ── ltv_per_month_quartile (pd.qcut) ─────────────────────────────────
    result["ltv_per_month_quartile"] = pd.qcut(
        result["lifetime_value_per_month"],
        q=4,
        labels=["tier_1", "tier_2", "tier_3", "tier_4"],
        duplicates="drop",
    )
    log.append({
        "column": "ltv_per_month_quartile",
        "type": "binned_quantile",
        "source": "lifetime_value_per_month",
        "q": 4,
        "labels": ["tier_1", "tier_2", "tier_3", "tier_4"],
        "business_meaning": (
            "Revenue-rate quartile. Normalises by tenure so a short-tenure "
            "high-spender ranks fairly against a long-tenure moderate spender."
        ),
    })

    print(f"[binned features]  added {len(log)} columns: "
          f"{[e['column'] for e in log]}")

    return result, log


# ---------------------------------------------------------------------------
# Step 3 – RFM composite score
# ---------------------------------------------------------------------------

def add_rfm_score(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """
    Build a classic RFM (Recency–Frequency–Monetary) composite score.

    Individual component scores (1–5 scale)
    ----------------------------------------
    recency_score    : quantile rank of days_since_last_purchase.
                       INVERTED — lower recency days = higher score (more recent).
    frequency_score  : quantile rank of total_transactions.
                       Higher transaction count = higher score.
    monetary_score   : quantile rank of total_spent.
                       Higher spend = higher score.

    Composite
    ---------
    rfm_score        : sum of the three component scores (range 3–15).
    rfm_segment      : labelled tier based on rfm_score total.
        champions    : 13–15
        loyal        : 10–12
        at_risk      : 7–9
        dormant      : 3–6
    """
    result = df.copy()
    log: list[dict] = []

    # Helper: robust qcut that falls back to rank-based scoring if too many ties
    def _score_quantile(series: pd.Series, q: int, ascending: bool = True) -> pd.Series:
        """
        Assign 1–q quantile score.  ascending=True means higher value → higher score.
        Returns Int64 Series.
        """
        labels = list(range(1, q + 1))
        try:
            scored = pd.qcut(series, q=q, labels=labels, duplicates="drop")
            # If drop reduced bins, fall back to rank
            if scored.isna().any():
                raise ValueError("NaN after qcut — using rank fallback")
            return scored.astype("Int64")
        except ValueError:
            pct = series.rank(pct=True)
            bins = np.linspace(0, 1, q + 1)
            ranked = pd.cut(pct, bins=bins, labels=labels, include_lowest=True)
            return ranked.astype("Int64")

    # recency_score: LOWER days = MORE recent = HIGHER score → ascending=False
    result["recency_score"] = _score_quantile(
        result["days_since_last_purchase"], q=5, ascending=True
    )
    # Invert: score 5 should be most recent (fewest days)
    result["recency_score"] = (6 - result["recency_score"]).astype("Int64")

    log.append({
        "column": "recency_score",
        "type": "rfm_component",
        "source": "days_since_last_purchase",
        "scale": "1 (least recent) – 5 (most recent)",
        "business_meaning": (
            "How recently the customer purchased. Score 5 = purchased in "
            "the last few days. Score 1 = has not purchased in a long time."
        ),
    })

    # frequency_score: MORE transactions = HIGHER score
    result["frequency_score"] = _score_quantile(
        result["total_transactions"], q=5, ascending=True
    )
    log.append({
        "column": "frequency_score",
        "type": "rfm_component",
        "source": "total_transactions",
        "scale": "1 (fewest) – 5 (most transactions)",
        "business_meaning": (
            "How often the customer purchases. Score 5 = top 20 %% "
            "most frequent buyers."
        ),
    })

    # monetary_score: HIGHER spend = HIGHER score
    result["monetary_score"] = _score_quantile(
        result["total_spent"], q=5, ascending=True
    )
    log.append({
        "column": "monetary_score",
        "type": "rfm_component",
        "source": "total_spent",
        "scale": "1 (lowest spend) – 5 (highest spend)",
        "business_meaning": (
            "How much the customer has spent in total. Score 5 = top 20 %% "
            "highest-value customers."
        ),
    })

    # Composite RFM score (3–15)
    result["rfm_score"] = (
        result["recency_score"].astype(float)
        + result["frequency_score"].astype(float)
        + result["monetary_score"].astype(float)
    ).astype("Int64")

    log.append({
        "column": "rfm_score",
        "type": "composite",
        "formula": "recency_score + frequency_score + monetary_score",
        "range": "3 (lowest) – 15 (highest)",
        "business_meaning": (
            "Single interpretable customer health score combining recency, "
            "frequency, and monetary signals. Drives segmentation and "
            "targeted retention/upsell campaigns."
        ),
    })

    # RFM segment label
    result["rfm_segment"] = pd.cut(
        result["rfm_score"].astype(float),
        bins=[2, 6, 9, 12, 15],
        labels=["dormant", "at_risk", "loyal", "champions"],
        right=True,
        include_lowest=True,
    )
    log.append({
        "column": "rfm_segment",
        "type": "composite_tier",
        "source": "rfm_score",
        "boundaries": {"dormant": "3–6", "at_risk": "7–9",
                       "loyal": "10–12", "champions": "13–15"},
        "business_meaning": (
            "Actionable customer segment derived from RFM score. "
            "'champions' receive VIP treatment; 'dormant' receive win-back campaigns."
        ),
    })

    print(f"[RFM score]        added {len(log)} columns: "
          f"{[e['column'] for e in log]}")

    return result, log


# ---------------------------------------------------------------------------
# Step 4 – Distribution validation
# ---------------------------------------------------------------------------

def validate_feature_distributions(
    df: pd.DataFrame,
    feature_log: list[dict],
) -> dict[str, dict]:
    """
    Verify every engineered column is internally consistent.

    Checks per column
    -----------------
    - Null count and null rate (any unexpected NaN?).
    - For numeric: min, max, mean, median, std.
    - For categorical/binned: value_counts with percentages.
    """
    validation: dict[str, dict] = {}

    for entry in feature_log:
        col = entry["column"]
        if col not in df.columns:
            continue

        s = df[col]
        null_count = int(s.isna().sum())
        null_rate = round(null_count / len(s) * 100, 2) if len(s) else 0.0

        col_stats: dict = {
            "null_count": null_count,
            "null_rate_pct": null_rate,
            "feature_type": entry.get("type", "unknown"),
        }

        if pd.api.types.is_numeric_dtype(s):
            numeric = s.dropna()
            col_stats.update({
                "min": round(float(numeric.min()), 4),
                "max": round(float(numeric.max()), 4),
                "mean": round(float(numeric.mean()), 4),
                "median": round(float(numeric.median()), 4),
                "std": round(float(numeric.std()), 4),
            })
        else:
            vc = s.value_counts(dropna=False)
            col_stats["value_distribution"] = {
                str(k): {
                    "count": int(v),
                    "pct": round(int(v) / len(s) * 100, 1),
                }
                for k, v in vc.items()
            }

        validation[col] = col_stats

    return validation


# ---------------------------------------------------------------------------
# Step 5 – Console report
# ---------------------------------------------------------------------------

def print_report(
    df: pd.DataFrame,
    ratio_log: list[dict],
    bin_log: list[dict],
    rfm_log: list[dict],
    validation: dict[str, dict],
) -> None:
    """Print a structured feature engineering summary to stdout."""

    sep = "=" * 65
    all_new_cols = [e["column"] for e in ratio_log + bin_log + rfm_log]

    print(f"\n{sep}")
    print("FEATURE ENGINEERING — SUMMARY REPORT")
    print(sep)

    print(f"\n  Total rows       : {len(df)}")
    print(f"  New columns added: {len(all_new_cols)}")
    print(f"  Final shape      : {df.shape[0]} rows × {df.shape[1]} cols")

    # --- Ratio features ---
    print(f"\n  ── RATIO FEATURES ──")
    for entry in ratio_log:
        col = entry["column"]
        v = validation.get(col, {})
        print(f"\n  {col}")
        print(f"    Formula : {entry['formula']}")
        print(f"    Meaning : {entry['business_meaning']}")
        print(f"    Nulls   : {v.get('null_count', '?')}  ({v.get('null_rate_pct', '?')} %)")
        print(f"    Range   : [{v.get('min', '?')}, {v.get('max', '?')}]  "
              f"mean={v.get('mean', '?')}  median={v.get('median', '?')}")

    # --- Binned features ---
    print(f"\n  ── BINNED / TIERED FEATURES ──")
    for entry in bin_log:
        col = entry["column"]
        v = validation.get(col, {})
        print(f"\n  {col}  ({entry['type']}  source={entry['source']})")
        print(f"    Meaning : {entry['business_meaning']}")
        print(f"    Nulls   : {v.get('null_count', '?')}  ({v.get('null_rate_pct', '?')} %)")
        dist = v.get("value_distribution", {})
        for label, stats in dist.items():
            bar = "█" * max(1, int(stats["pct"] / 5))
            print(f"    {str(label):<12} {bar:<20} {stats['count']:>4} rows  "
                  f"({stats['pct']} %)")

    # --- RFM score ---
    print(f"\n  ── RFM COMPOSITE SCORE ──")
    for entry in rfm_log:
        col = entry["column"]
        v = validation.get(col, {})
        print(f"\n  {col}")
        print(f"    Meaning : {entry['business_meaning']}")
        print(f"    Nulls   : {v.get('null_count', '?')}  ({v.get('null_rate_pct', '?')} %)")
        if "min" in v:
            print(f"    Range   : [{v['min']}, {v['max']}]  "
                  f"mean={v['mean']}  median={v['median']}")
        dist = v.get("value_distribution", {})
        for label, stats in dist.items():
            bar = "█" * max(1, int(stats["pct"] / 5))
            print(f"    {str(label):<12} {bar:<20} {stats['count']:>4} rows  "
                  f"({stats['pct']} %)")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Step 6 – Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    feature_log: list[dict],
    validation: dict[str, dict],
    output_csv: str = "output/features.csv",
) -> None:
    """
    Write enriched dataset and JSON feature report to output/.

    Files created
    -------------
    output/features.csv               — dataset with all engineered columns
    output/feature_engineering_report.json — feature log + distribution validation
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_csv)
    df.to_csv(csv_path, index=False)
    print(f"[save_outputs] Features CSV saved        → {csv_path}  ({len(df)} rows, "
          f"{len(df.columns)} cols)")

    report = {
        "feature_log": feature_log,
        "distribution_validation": validation,
    }

    report_path = output_dir / "feature_engineering_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)
    print(f"[save_outputs] Feature report saved      → {report_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature Engineering & Derived Business Columns — NexPulse"
    )
    parser.add_argument(
        "--input", dest="input_path",
        help="Path to a CSV file. Omit to use the built-in demo dataset.",
    )
    parser.add_argument(
        "--output", dest="output_path", default="output/features.csv",
        help="Destination path for the feature-enriched CSV.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Force the built-in demo dataset.",
    )
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.demo or not args.input_path:
        print("Using built-in demo dataset.")
        return build_demo_data(), "demo dataset"

    path = Path(args.input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    print(f"Loaded CSV: {path}  ({len(df)} rows, {len(df.columns)} cols)")
    return df, str(path)


def main() -> None:
    args = parse_args()

    # 1. Load
    df_raw, source_label = load_input(args)
    print(f"\nSource : {source_label}")
    print(f"Shape  : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
    print(f"Cols   : {list(df_raw.columns)}\n")

    # 2. Ratio features
    df, ratio_log = add_ratio_features(df_raw)

    # 3. Binned / tiered features
    df, bin_log = add_binned_features(df)

    # 4. RFM composite score
    df, rfm_log = add_rfm_score(df)

    # 5. Distribution validation
    all_log = ratio_log + bin_log + rfm_log
    validation = validate_feature_distributions(df, all_log)

    # 6. Console report
    print_report(df, ratio_log, bin_log, rfm_log, validation)

    # 7. Persist
    save_outputs(df, all_log, validation, output_csv=args.output_path)

    new_cols = [e["column"] for e in all_log]
    print(f"\nFeature engineering complete.")
    print(f"  Rows          : {len(df)}")
    print(f"  New columns   : {len(new_cols)}  → {new_cols}")
    print(f"  Total columns : {len(df.columns)}")


if __name__ == "__main__":
    main()
