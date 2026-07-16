"""
datetime_transform.py
---------------------
Date & Time Transformation Pipeline for the NexPulse workflow.

Responsibilities
----------------
1. Parse transaction_date strings into proper datetime objects.
2. Extract calendar features: day-of-week, hour-of-day, week number,
   month, quarter.
3. Compute recency metric: days_since_transaction.
4. Enable time-series aggregation via .resample() on a datetime index.
5. Persist enriched data and a JSON feature report to output/.

Usage
-----
    # Demo dataset (built-in):
    python scripts/datetime_transform.py --demo

    # Real CSV:
    python scripts/datetime_transform.py --input data/raw/sample.csv

    # Real JSON (transactions):
    python scripts/datetime_transform.py --input data/raw/transactions.json

    # Explicit output path:
    python scripts/datetime_transform.py --demo --output output/datetime_enriched.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
RAW_DATE_FORMAT = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def build_demo_data() -> pd.DataFrame:
    """Return a small transaction dataset with string timestamps."""
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "customer_id": [101, 102, 101, 103, 104, 102, 105, 103, 101, 104],
            "amount": [150.50, 200.00, 75.25, 320.00, 490.00, 210.00, 130.00, 600.00, 45.00, 88.75],
            "status": [
                "completed", "pending", "completed", "completed",
                "completed", "pending", "completed", "failed",
                "completed", "completed",
            ],
            # Stored as plain strings — the problem this pipeline solves
            "transaction_date": [
                "2025-01-15 14:30:45",
                "2025-01-20 09:15:00",
                "2025-02-01 18:05:30",
                "2025-02-10 11:45:00",
                "2025-03-05 08:00:00",
                "2025-03-12 20:22:10",
                "2025-04-03 16:10:55",
                "2025-04-18 07:55:30",
                "2025-05-22 13:40:00",
                "2025-06-01 23:59:00",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Step 1 – Parse timestamps
# ---------------------------------------------------------------------------

def parse_timestamps(df: pd.DataFrame, column: str = "transaction_date") -> pd.DataFrame:
    """
    Convert a string timestamp column to datetime64.

    Supports both full timestamps ("%Y-%m-%d %H:%M:%S") and date-only
    strings ("%Y-%m-%d").  Raises ValueError if neither format matches.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe — column must contain string timestamps.
    column : str
        Name of the date column to parse.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the column converted to datetime64.
    """
    result = df.copy()

    if column not in result.columns:
        raise KeyError(f"Column '{column}' not found. Available: {list(result.columns)}")

    # Already parsed — nothing to do
    if pd.api.types.is_datetime64_any_dtype(result[column]):
        print(f"[parse_timestamps] '{column}' is already datetime — skipping parse.")
        return result

    print(f"[parse_timestamps] Parsing '{column}' from string → datetime64 …")
    before_dtype = str(result[column].dtype)

    # Try full timestamp format first, fall back to date-only
    try:
        result[column] = pd.to_datetime(result[column], format=RAW_TIMESTAMP_FORMAT, errors="raise")
    except ValueError:
        result[column] = pd.to_datetime(result[column], format=RAW_DATE_FORMAT, errors="raise")

    after_dtype = str(result[column].dtype)
    print(f"  dtype: {before_dtype} → {after_dtype}")
    print(f"  rows parsed: {result[column].notna().sum()} / {len(result)}")
    return result


# ---------------------------------------------------------------------------
# Step 2 – Extract calendar features
# ---------------------------------------------------------------------------

def extract_time_features(df: pd.DataFrame, column: str = "transaction_date") -> tuple[pd.DataFrame, dict]:
    """
    Use the .dt accessor to derive calendar and recency features.

    New columns added
    -----------------
    transaction_day_of_week      : str  — "Monday", "Tuesday", …
    transaction_day_of_week_num  : Int64 — 0 (Monday) … 6 (Sunday)
    transaction_hour_of_day      : Int64 — 0–23
    transaction_week_number      : Int64 — ISO week 1–53
    transaction_month            : Int64 — 1–12
    transaction_quarter          : Int64 — 1–4
    days_since_transaction       : Int64 — days from transaction date to today

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe where *column* is already datetime64.
    column : str
        Name of the parsed datetime column.

    Returns
    -------
    (enriched_df, feature_summary)
        enriched_df      : DataFrame with the new feature columns appended.
        feature_summary  : dict summarising distributions for reporting.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found.")

    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        raise TypeError(
            f"Column '{column}' must be datetime64 before extracting features. "
            "Call parse_timestamps() first."
        )

    enriched = df.copy()
    date_series = enriched[column]

    # Timezone-aware reference timestamp
    tz = getattr(date_series.dt, "tz", None)
    today = pd.Timestamp.now(tz=tz) if tz else pd.Timestamp.now()

    print(f"\n[extract_time_features] Extracting features from '{column}' …")

    # --- Day of week (name + numeric) ---
    enriched["transaction_day_of_week"] = date_series.dt.day_name().astype("string")
    enriched["transaction_day_of_week_num"] = date_series.dt.dayofweek.astype("Int64")

    # --- Hour of day ---
    enriched["transaction_hour_of_day"] = date_series.dt.hour.astype("Int64")

    # --- ISO week number ---
    enriched["transaction_week_number"] = date_series.dt.isocalendar().week.astype("Int64")

    # --- Month and quarter ---
    enriched["transaction_month"] = date_series.dt.month.astype("Int64")
    enriched["transaction_quarter"] = date_series.dt.quarter.astype("Int64")

    # --- Recency: days since transaction ---
    enriched["days_since_transaction"] = (today - date_series).dt.days.astype("Int64")

    feature_columns = [
        "transaction_day_of_week",
        "transaction_day_of_week_num",
        "transaction_hour_of_day",
        "transaction_week_number",
        "transaction_month",
        "transaction_quarter",
        "days_since_transaction",
    ]

    print(f"  Features added: {feature_columns}")

    # Build summary distributions for the report
    feature_summary: dict = {
        "reference_date": today.isoformat(),
        "feature_columns": feature_columns,
        "hour_of_day_distribution": {
            int(k): int(v)
            for k, v in enriched["transaction_hour_of_day"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
            if pd.notna(k)
        },
        "day_of_week_distribution": {
            str(k): int(v)
            for k, v in enriched["transaction_day_of_week"]
            .value_counts(dropna=False)
            .items()
            if pd.notna(k)
        },
        "week_number_distribution": {
            int(k): int(v)
            for k, v in enriched["transaction_week_number"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
            if pd.notna(k)
        },
        "month_distribution": {
            int(k): int(v)
            for k, v in enriched["transaction_month"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
            if pd.notna(k)
        },
        "recency_stats": {
            "min_days": int(enriched["days_since_transaction"].min()),
            "max_days": int(enriched["days_since_transaction"].max()),
            "mean_days": round(float(enriched["days_since_transaction"].mean()), 1),
        },
    }

    return enriched, feature_summary


# ---------------------------------------------------------------------------
# Step 3 – Time-series aggregation via resample
# ---------------------------------------------------------------------------

def aggregate_by_time(
    df: pd.DataFrame,
    date_column: str = "transaction_date",
    value_column: str = "amount",
) -> dict:
    """
    Perform .resample() aggregations on a datetime-indexed DataFrame.

    Frequencies computed
    --------------------
    D (daily), W (weekly), ME (month-end), QE (quarter-end)

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe; *date_column* must be datetime64.
    date_column : str
        Column to use as the time index.
    value_column : str
        Numeric column to aggregate (sum and count).

    Returns
    -------
    dict
        Nested dict with aggregation results keyed by frequency label.
    """
    if value_column not in df.columns:
        print(f"[aggregate_by_time] '{value_column}' not found — skipping aggregation.")
        return {}

    print(f"\n[aggregate_by_time] Building time-series aggregations on '{value_column}' …")

    # Set datetime as index — required for .resample()
    time_indexed = df.set_index(date_column).sort_index()

    aggregations: dict = {}

    freq_map = {
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
    }

    for label, freq in freq_map.items():
        amount_sum = time_indexed[value_column].resample(freq).sum(min_count=1)
        amount_count = time_indexed[value_column].resample(freq).count()

        records = [
            {
                "period_end": ts.isoformat(),
                "total_amount": round(float(v), 2),
                "transaction_count": int(amount_count[ts]),
            }
            for ts, v in amount_sum.items()
            if pd.notna(v)
        ]

        aggregations[label] = records
        print(f"  {label} ({freq}): {len(records)} period(s)")

    return aggregations


# ---------------------------------------------------------------------------
# Step 4 – Print a formatted console report
# ---------------------------------------------------------------------------

def print_report(df: pd.DataFrame, feature_summary: dict, aggregations: dict) -> None:
    """Print a readable summary of the datetime transformation results."""

    separator = "=" * 60

    print(f"\n{separator}")
    print("DATE & TIME TRANSFORMATION — SUMMARY")
    print(separator)

    print(f"\nRows enriched : {len(df)}")
    print(f"Reference date: {feature_summary.get('reference_date', 'N/A')}")

    print("\n--- Feature columns added ---")
    for col in feature_summary.get("feature_columns", []):
        print(f"  • {col}")

    print("\n--- Hour-of-day distribution ---")
    for hour, count in feature_summary.get("hour_of_day_distribution", {}).items():
        bar = "█" * count
        print(f"  {int(hour):02d}:00  {bar}  ({count})")

    print("\n--- Day-of-week distribution ---")
    for day, count in feature_summary.get("day_of_week_distribution", {}).items():
        bar = "█" * count
        print(f"  {day:<12} {bar}  ({count})")

    print("\n--- Recency (days since transaction) ---")
    recency = feature_summary.get("recency_stats", {})
    print(f"  Min  : {recency.get('min_days')} days")
    print(f"  Max  : {recency.get('max_days')} days")
    print(f"  Mean : {recency.get('mean_days')} days")

    print("\n--- Time-series aggregations ---")
    for freq, records in aggregations.items():
        print(f"\n  [{freq.upper()}]")
        for r in records:
            print(f"    {r['period_end'][:10]}  amount={r['total_amount']:>10.2f}  transactions={r['transaction_count']}")

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# Step 5 – Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    feature_summary: dict,
    aggregations: dict,
    output_csv: str = "output/datetime_enriched.csv",
) -> None:
    """
    Write enriched CSV and JSON report to output/.

    Files created
    -------------
    output/datetime_enriched.csv       — full enriched dataset
    output/datetime_feature_report.json — feature + aggregation summary
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_csv)
    df.to_csv(csv_path, index=False)
    print(f"[save_outputs] Enriched CSV saved → {csv_path}")

    report = {
        "feature_summary": feature_summary,
        "time_series_aggregations": aggregations,
    }

    report_path = output_dir / "datetime_feature_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4, default=str)
    print(f"[save_outputs] Feature report saved → {report_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Date & Time Transformation Pipeline — NexPulse"
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to a CSV or JSON file. Omit to use the built-in demo dataset.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="output/datetime_enriched.csv",
        help="Destination path for the enriched CSV (default: output/datetime_enriched.csv).",
    )
    parser.add_argument(
        "--date-column",
        dest="date_column",
        default="transaction_date",
        help="Name of the timestamp column to parse (default: transaction_date).",
    )
    parser.add_argument(
        "--value-column",
        dest="value_column",
        default="amount",
        help="Numeric column to use for time-series aggregation (default: amount).",
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
    suffix = input_path.suffix.lower()

    if suffix == ".json":
        import json as _json
        with input_path.open("r", encoding="utf-8") as fh:
            raw = _json.load(fh)
        df = pd.json_normalize(raw) if isinstance(raw, list) else pd.DataFrame([raw])
        print(f"Loaded JSON: {input_path}  ({len(df)} rows)")
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
        print(f"Loaded CSV: {input_path}  ({len(df)} rows)")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .json.")

    return df, str(input_path)


def main() -> None:
    args = parse_args()

    # 1. Load data
    df, source_label = load_input(args)

    print(f"\nSource     : {source_label}")
    print(f"Shape      : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns    : {list(df.columns)}")
    print(f"Date dtype : {df[args.date_column].dtype if args.date_column in df.columns else 'column not found'}")

    # 2. Parse timestamps → datetime64
    df = parse_timestamps(df, column=args.date_column)

    # 3. Extract calendar + recency features
    df, feature_summary = extract_time_features(df, column=args.date_column)

    # 4. Time-series aggregation
    aggregations = aggregate_by_time(df, date_column=args.date_column, value_column=args.value_column)

    # 5. Console report
    print_report(df, feature_summary, aggregations)

    # 6. Persist outputs
    save_outputs(df, feature_summary, aggregations, output_csv=args.output_path)

    print(f"Pipeline complete. Rows processed: {len(df)}")


if __name__ == "__main__":
    main()
