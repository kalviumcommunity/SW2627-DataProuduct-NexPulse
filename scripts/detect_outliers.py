"""
detect_outliers.py
------------------
Outlier Detection with Statistical Methods for the NexPulse workflow.

Responsibilities
----------------
1. Detect outliers using Z-score (scipy.stats) and IQR methods.
2. Apply a configurable handling strategy per column:
      cap    — clip values to the detected boundary; preserves all rows.
      remove — drop rows where any flagged column is an outlier.
      flag   — add a boolean indicator column; keep all data intact.
3. Produce a structured cleaning log: column, method, strategy, count,
   bounds, and business reasoning.
4. Persist the cleaned dataset and JSON audit report to output/.

Usage
-----
    # Built-in demo dataset:
    python scripts/detect_outliers.py --demo

    # Real CSV:
    python scripts/detect_outliers.py --input data/raw/sample.csv

    # Real CSV with custom output path:
    python scripts/detect_outliers.py --input data/raw/sample.csv \
        --output output/outlier_cleaned.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

DetectionMethod = Literal["zscore", "iqr"]
HandlingStrategy = Literal["cap", "remove", "flag"]


# ---------------------------------------------------------------------------
# Column-level outlier configuration
# ---------------------------------------------------------------------------

# Each entry defines how to treat a specific column:
#   method   : "zscore" or "iqr"
#   strategy : "cap", "remove", or "flag"
#   threshold: z-score cutoff (zscore method) or IQR multiplier (iqr method)
#   reason   : business justification logged in the audit report

OUTLIER_CONFIG: dict[str, dict] = {
    "salary": {
        "method": "zscore",
        "strategy": "cap",
        "threshold": 3.0,
        "reason": (
            "Executive compensation outliers skew mean salary. "
            "Cap at 3-sigma boundary to preserve the row while bounding influence."
        ),
    },
    "transaction_amount": {
        "method": "iqr",
        "strategy": "cap",
        "threshold": 1.5,
        "reason": (
            "Transaction amounts follow a right-skewed distribution. "
            "IQR is more robust than Z-score for skewed data. "
            "Cap prevents extreme spend from distorting average order value."
        ),
    },
    "vacation_days": {
        "method": "iqr",
        "strategy": "flag",
        "threshold": 1.5,
        "reason": (
            "Vacation days above company max (30) may reflect data-entry errors "
            "or special leave. Flag for HR review rather than silently removing."
        ),
    },
    "return_rate": {
        "method": "zscore",
        "strategy": "remove",
        "threshold": 3.0,
        "reason": (
            "Return rates above 90 % are physically implausible for this product "
            "category and indicate data collection errors. Remove the row."
        ),
    },
    "quantity": {
        "method": "iqr",
        "strategy": "flag",
        "threshold": 1.5,
        "reason": (
            "Unusually high order quantities may be bulk/wholesale orders. "
            "Flag for separate analysis; do not remove or cap."
        ),
    },
}


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def build_demo_data() -> pd.DataFrame:
    """
    Return a realistic employee / transaction dataset containing intentional
    outliers that exercise all three handling strategies.
    """
    rng = np.random.default_rng(seed=42)

    n = 40

    # Base salary: ~50 000, one executive outlier at 650 000
    salary = rng.normal(loc=50_000, scale=8_000, size=n).round(2)
    salary[5] = 650_000   # extreme high — Z-score outlier
    salary[12] = 3_000    # extreme low  — Z-score outlier

    # Transaction amounts: right-skewed, one whale spend at 50 000
    tx_amount = rng.exponential(scale=200, size=n).round(2) + 20
    tx_amount[8] = 50_000   # whale — IQR outlier
    tx_amount[25] = 0.01    # near-zero — IQR outlier

    # Vacation days: mostly 0–25, two impossible values
    vacation = rng.integers(0, 26, size=n).astype(float)
    vacation[3] = 150   # data-entry error
    vacation[18] = 95   # data-entry error

    # Return rate 0–20 %, two implausible values near 100 %
    return_rate = rng.uniform(0, 20, size=n).round(2)
    return_rate[7] = 97.5   # implausible — will be removed
    return_rate[33] = 99.0  # implausible — will be removed

    # Order quantity: mostly 1–15, two bulk orders
    quantity = rng.integers(1, 16, size=n).astype(float)
    quantity[14] = 980   # bulk order — IQR outlier
    quantity[29] = 750   # bulk order — IQR outlier

    return pd.DataFrame(
        {
            "employee_id": range(1, n + 1),
            "salary": salary,
            "transaction_amount": tx_amount,
            "vacation_days": vacation,
            "return_rate": return_rate,
            "quantity": quantity,
            "department": rng.choice(
                ["Engineering", "Sales", "Marketing", "HR"], size=n
            ),
        }
    )


# ---------------------------------------------------------------------------
# Step 1 – Detection: Z-score
# ---------------------------------------------------------------------------

def detect_zscore_outliers(
    series: pd.Series,
    threshold: float = 3.0,
) -> tuple[pd.Series, float, float]:
    """
    Identify outliers using the Z-score method.

    A value is an outlier when |z| > threshold (default 3.0 standard deviations
    from the mean).  Assumes the column has an approximately normal distribution.

    Parameters
    ----------
    series    : numeric pd.Series (NaN values are ignored).
    threshold : Z-score cutoff; default 3.0.

    Returns
    -------
    (outlier_mask, lower_bound, upper_bound)
        outlier_mask : boolean Series, True where value is an outlier.
        lower_bound  : mean − threshold × std
        upper_bound  : mean + threshold × std
    """
    clean = series.dropna()
    z_scores = np.abs(stats.zscore(clean, ddof=0))

    mean = clean.mean()
    std = clean.std(ddof=0)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    # Re-index back to the full series index (NaN → False)
    outlier_mask = pd.Series(False, index=series.index)
    outlier_mask.loc[clean.index] = z_scores > threshold

    return outlier_mask, round(float(lower_bound), 4), round(float(upper_bound), 4)


# ---------------------------------------------------------------------------
# Step 2 – Detection: IQR
# ---------------------------------------------------------------------------

def detect_iqr_outliers(
    series: pd.Series,
    multiplier: float = 1.5,
) -> tuple[pd.Series, float, float]:
    """
    Identify outliers using the Interquartile Range (IQR) method.

    Boundaries: Q1 − multiplier × IQR  and  Q3 + multiplier × IQR.
    Robust to skewed distributions; does not assume normality.

    Parameters
    ----------
    series     : numeric pd.Series (NaN values are ignored).
    multiplier : IQR scaling factor; default 1.5 (Tukey fences).

    Returns
    -------
    (outlier_mask, lower_bound, upper_bound)
        outlier_mask : boolean Series, True where value is an outlier.
        lower_bound  : Q1 − multiplier × IQR
        upper_bound  : Q3 + multiplier × IQR
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    # Keep NaN as False
    outlier_mask = outlier_mask.fillna(False)

    return outlier_mask, round(float(lower_bound), 4), round(float(upper_bound), 4)


# ---------------------------------------------------------------------------
# Step 3 – Handling: cap
# ---------------------------------------------------------------------------

def apply_cap(
    df: pd.DataFrame,
    column: str,
    lower_bound: float,
    upper_bound: float,
) -> pd.DataFrame:
    """
    Clip *column* values to [lower_bound, upper_bound].

    Values below lower_bound become lower_bound.
    Values above upper_bound become upper_bound.
    All rows are preserved.

    Returns a copy of *df* with the column clipped in-place.
    """
    result = df.copy()
    result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
    return result


# ---------------------------------------------------------------------------
# Step 4 – Handling: remove
# ---------------------------------------------------------------------------

def apply_remove(
    df: pd.DataFrame,
    outlier_mask: pd.Series,
) -> pd.DataFrame:
    """
    Drop rows where *outlier_mask* is True.

    Returns a copy of *df* with outlier rows removed and index reset.
    """
    return df[~outlier_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5 – Handling: flag
# ---------------------------------------------------------------------------

def apply_flag(
    df: pd.DataFrame,
    column: str,
    outlier_mask: pd.Series,
) -> pd.DataFrame:
    """
    Add a boolean indicator column ``is_<column>_outlier``.

    All rows are kept.  Downstream analysis can filter or weight them
    independently using the indicator column.

    Returns a copy of *df* with the new flag column appended.
    """
    result = df.copy()
    flag_column = f"is_{column}_outlier"
    result[flag_column] = outlier_mask.astype("boolean")
    return result


# ---------------------------------------------------------------------------
# Step 6 – Orchestration: run the full pipeline
# ---------------------------------------------------------------------------

def run_outlier_pipeline(
    df: pd.DataFrame,
    config: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Apply outlier detection and handling for every configured column.

    For each column in *config* that exists in *df*:
      1. Detect outliers using the specified method (zscore or iqr).
      2. Apply the specified strategy (cap / remove / flag).
      3. Append a cleaning-log entry documenting the decision.

    Parameters
    ----------
    df     : Input DataFrame.
    config : Column-level outlier configuration dict.
             Defaults to the module-level OUTLIER_CONFIG.

    Returns
    -------
    (cleaned_df, cleaning_log)
        cleaned_df   : DataFrame after all strategies applied.
        cleaning_log : List of dicts; one entry per processed column.
    """
    if config is None:
        config = OUTLIER_CONFIG

    cleaned = df.copy()
    cleaning_log: list[dict] = []
    rows_removed_total = 0

    for column, cfg in config.items():
        if column not in cleaned.columns:
            print(f"[outlier_pipeline] Column '{column}' not found — skipping.")
            continue

        method: DetectionMethod = cfg["method"]
        strategy: HandlingStrategy = cfg["strategy"]
        threshold: float = cfg.get("threshold", 3.0 if method == "zscore" else 1.5)
        reason: str = cfg.get("reason", "No reason provided.")

        # --- Detect ---
        if method == "zscore":
            outlier_mask, lower_bound, upper_bound = detect_zscore_outliers(
                cleaned[column], threshold=threshold
            )
        elif method == "iqr":
            outlier_mask, lower_bound, upper_bound = detect_iqr_outliers(
                cleaned[column], multiplier=threshold
            )
        else:
            raise ValueError(f"Unknown detection method: '{method}'. Use 'zscore' or 'iqr'.")

        outlier_count = int(outlier_mask.sum())

        print(
            f"\n[{column}]  method={method}  strategy={strategy}  "
            f"outliers={outlier_count}  bounds=[{lower_bound}, {upper_bound}]"
        )

        # --- Handle ---
        rows_before = len(cleaned)

        if strategy == "cap":
            cleaned = apply_cap(cleaned, column, lower_bound, upper_bound)
            rows_after = len(cleaned)
            action_detail = f"Values clipped to [{lower_bound}, {upper_bound}]"

        elif strategy == "remove":
            cleaned = apply_remove(cleaned, outlier_mask)
            rows_after = len(cleaned)
            rows_removed = rows_before - rows_after
            rows_removed_total += rows_removed
            action_detail = f"{rows_removed} row(s) removed"

        elif strategy == "flag":
            cleaned = apply_flag(cleaned, column, outlier_mask)
            rows_after = len(cleaned)
            flag_col = f"is_{column}_outlier"
            action_detail = f"Flag column '{flag_col}' added; {outlier_count} row(s) marked True"

        else:
            raise ValueError(f"Unknown strategy: '{strategy}'. Use 'cap', 'remove', or 'flag'.")

        # --- Log ---
        log_entry = {
            "column": column,
            "detection_method": method,
            "handling_strategy": strategy,
            "threshold_used": threshold,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers_detected": outlier_count,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "action_detail": action_detail,
            "business_reason": reason,
        }
        cleaning_log.append(log_entry)

        if outlier_count:
            print(f"  → {action_detail}")
        else:
            print(f"  → No outliers detected.")

    print(f"\n[outlier_pipeline] Total rows removed across all 'remove' strategies: {rows_removed_total}")
    return cleaned, cleaning_log


# ---------------------------------------------------------------------------
# Step 7 – Summary statistics: before vs after
# ---------------------------------------------------------------------------

def compute_summary_stats(df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    """
    Compute mean, median, std, min, max for each numeric column.

    Used to compare statistics before and after outlier handling.
    """
    summary: dict[str, dict] = {}
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        summary[col] = {
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "count": int(s.count()),
        }
    return summary


# ---------------------------------------------------------------------------
# Step 8 – Console report
# ---------------------------------------------------------------------------

def print_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    cleaning_log: list[dict],
    numeric_columns: list[str],
) -> None:
    """Print a readable audit report comparing before/after statistics."""

    sep = "=" * 65

    print(f"\n{sep}")
    print("OUTLIER DETECTION — AUDIT REPORT")
    print(sep)

    print(f"\nRows before : {len(df_before)}")
    print(f"Rows after  : {len(df_after)}")
    print(f"Rows removed: {len(df_before) - len(df_after)}")

    print("\n--- Cleaning log ---")
    for entry in cleaning_log:
        print(
            f"\n  Column   : {entry['column']}\n"
            f"  Method   : {entry['detection_method']}  "
            f"(threshold={entry['threshold_used']})\n"
            f"  Strategy : {entry['handling_strategy']}\n"
            f"  Bounds   : [{entry['lower_bound']}, {entry['upper_bound']}]\n"
            f"  Detected : {entry['outliers_detected']} outlier(s)\n"
            f"  Action   : {entry['action_detail']}\n"
            f"  Reason   : {entry['business_reason']}"
        )

    stats_before = compute_summary_stats(df_before, numeric_columns)
    stats_after = compute_summary_stats(df_after, numeric_columns)

    print(f"\n--- Statistics: before vs after ---")
    print(f"  {'Column':<22} {'Stat':<8} {'Before':>14} {'After':>14}  {'Delta':>12}")
    print(f"  {'-'*22} {'-'*8} {'-'*14} {'-'*14}  {'-'*12}")

    for col in numeric_columns:
        if col not in stats_before:
            continue
        b = stats_before[col]
        a = stats_after.get(col, {})
        for stat in ("mean", "median", "std", "min", "max"):
            b_val = b.get(stat, float("nan"))
            a_val = a.get(stat, float("nan"))
            try:
                delta = round(a_val - b_val, 4)
            except TypeError:
                delta = "n/a"
            print(f"  {col:<22} {stat:<8} {b_val:>14.4f} {a_val:>14.4f}  {str(delta):>12}")
        print()

    print(sep)


# ---------------------------------------------------------------------------
# Step 9 – Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    cleaning_log: list[dict],
    stats_before: dict,
    stats_after: dict,
    output_csv: str = "output/outlier_cleaned.csv",
) -> None:
    """
    Write cleaned CSV and JSON audit report to output/.

    Files created
    -------------
    output/outlier_cleaned.csv       — cleaned dataset
    output/outlier_audit_report.json — cleaning log + before/after stats
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_csv)
    df.to_csv(csv_path, index=False)
    print(f"\n[save_outputs] Cleaned CSV saved → {csv_path}")

    report = {
        "cleaning_log": cleaning_log,
        "statistics_before": stats_before,
        "statistics_after": stats_after,
    }

    report_path = output_dir / "outlier_audit_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)
    print(f"[save_outputs] Audit report saved  → {report_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Outlier Detection with Statistical Methods — NexPulse"
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to a CSV file. Omit to use the built-in demo dataset.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="output/outlier_cleaned.csv",
        help="Destination path for the cleaned CSV (default: output/outlier_cleaned.csv).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force the built-in demo dataset even when --input is provided.",
    )
    return parser.parse_args()


def load_input(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    """Load the dataset according to CLI arguments."""
    if args.demo or not args.input_path:
        print("Using built-in demo dataset.")
        return build_demo_data(), "demo dataset"

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded CSV: {input_path}  ({len(df)} rows, {len(df.columns)} columns)")
    return df, str(input_path)


def main() -> None:
    args = parse_args()

    # 1. Load data
    df_raw, source_label = load_input(args)

    print(f"\nSource  : {source_label}")
    print(f"Shape   : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    print(f"Columns : {list(df_raw.columns)}")

    # Numeric columns that have a config entry and exist in the dataframe
    numeric_columns = [c for c in OUTLIER_CONFIG if c in df_raw.columns]

    # 2. Snapshot statistics before any changes
    stats_before = compute_summary_stats(df_raw, numeric_columns)

    # 3. Run the outlier detection + handling pipeline
    df_cleaned, cleaning_log = run_outlier_pipeline(df_raw)

    # 4. Snapshot statistics after all changes
    stats_after = compute_summary_stats(df_cleaned, numeric_columns)

    # 5. Console audit report
    print_report(df_raw, df_cleaned, cleaning_log, numeric_columns)

    # 6. Persist outputs
    save_outputs(df_cleaned, cleaning_log, stats_before, stats_after, args.output_path)

    print(f"\nPipeline complete.")
    print(f"  Rows in  : {len(df_raw)}")
    print(f"  Rows out : {len(df_cleaned)}")
    print(f"  Columns  : {len(df_cleaned.columns)}  (flag columns added where strategy=flag)")


if __name__ == "__main__":
    main()
