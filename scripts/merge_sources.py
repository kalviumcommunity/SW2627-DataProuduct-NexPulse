"""
merge_sources.py
----------------
Multi-Source Merging & Join Validation for the NexPulse workflow.

Responsibilities
----------------
1. Merge two datasets using an explicitly chosen join type
   (inner / left / right / outer).
2. Validate the merge: compare row counts before and after, explain
   why the count changed, detect key multiplicity.
3. Isolate unmatched keys from both sides into separate audit files.
4. Document the join decision — type chosen, business reasoning,
   expected vs actual row counts.
5. Persist the merged dataset, unmatched records, and a full JSON
   merge report to output/.

Usage
-----
    # Built-in demo (customers + orders):
    python scripts/merge_sources.py --demo

    # Real CSV files:
    python scripts/merge_sources.py \
        --left  data/raw/customers.csv \
        --right data/raw/orders.csv \
        --key   customer_id \
        --how   left

    # Custom output path:
    python scripts/merge_sources.py --demo --output output/merged.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

JoinType = Literal["inner", "left", "right", "outer"]


# ---------------------------------------------------------------------------
# Join-type reference: semantics + when-to-use documentation
# ---------------------------------------------------------------------------

JOIN_TYPE_DOCS: dict[str, dict] = {
    "inner": {
        "keeps": "Only rows where the key exists in BOTH tables.",
        "result_size": "Smaller than or equal to both inputs.",
        "use_when": (
            "You want complete records only. Records that cannot be matched "
            "on either side are excluded. Use when both tables are required "
            "for the analysis and partial records are not acceptable."
        ),
    },
    "left": {
        "keeps": "All rows from the LEFT table; matched rows from RIGHT; "
                 "NaN where RIGHT has no match.",
        "result_size": "Equal to the left table row count (before multiplicity).",
        "use_when": (
            "The LEFT table is your source of truth (e.g. customers). "
            "The RIGHT table is enrichment (e.g. orders). "
            "Customers with no orders are kept; their order columns are NaN."
        ),
    },
    "right": {
        "keeps": "All rows from the RIGHT table; matched rows from LEFT; "
                 "NaN where LEFT has no match.",
        "result_size": "Equal to the right table row count (before multiplicity).",
        "use_when": (
            "The RIGHT table is authoritative (e.g. all orders must be kept). "
            "Mirror of left join — use when the right dataset drives the output."
        ),
    },
    "outer": {
        "keeps": "All rows from BOTH tables. NaN where either side has no match.",
        "result_size": "Greater than or equal to the larger of the two inputs.",
        "use_when": (
            "You need full coverage — no record from either side should be lost. "
            "Useful for gap analysis: which customers never ordered, "
            "which orders have no customer profile."
        ),
    },
}


# ---------------------------------------------------------------------------
# Demo datasets
# ---------------------------------------------------------------------------

def build_demo_customers() -> pd.DataFrame:
    """
    Return a customer master table.
    Customers 9 and 10 have no orders → will appear as unmatched on a left/outer join.
    """
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "customer_name": [
                "Alice", "Bob", "Carol", "Diana", "Eve",
                "Frank", "Grace", "Heidi", "Ivan", "Judy",
            ],
            "segment":  ["B2C", "B2B", "B2C", "B2B", "B2C",
                         "B2B", "B2C", "B2C", "B2B", "B2C"],
            "country": ["US", "GB", "FR", "DE", "IN",
                        "AU", "CA", "JP", "US", "GB"],
            "signup_date": [
                "2023-01-10", "2023-02-14", "2023-03-05", "2023-04-20",
                "2023-05-18", "2023-06-01", "2023-07-22", "2023-08-30",
                "2023-09-11", "2023-10-03",
            ],
        }
    )


def build_demo_orders() -> pd.DataFrame:
    """
    Return an orders table.
    Order with customer_id=99 is an orphan → will appear as unmatched on a left/outer join.
    Multiple orders per customer (1, 2, 3) → demonstrates multiplicity.
    """
    return pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
            "customer_id": [  1,   1,   2,   2,   3,   4,   5,   6,   7,   8,   8,   8,  99],
            #                                                                 ↑ three orders for customer 8
            #                                                                                   ↑ orphan
            "order_date": [
                "2024-01-05", "2024-02-10", "2024-01-20", "2024-03-15",
                "2024-02-28", "2024-04-01", "2024-01-12", "2024-05-06",
                "2024-03-22", "2024-02-14", "2024-04-18", "2024-06-01",
                "2024-03-10",
            ],
            "amount":     [150.00, 89.50, 320.00, 45.00, 210.75, 500.00,
                           75.25, 430.00, 190.00, 60.00, 320.00, 85.50, 999.00],
            "status": [
                "completed", "completed", "completed", "pending",
                "completed", "completed", "completed", "completed",
                "pending",   "completed", "completed", "completed", "completed",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Step 1 – Pre-merge profile
# ---------------------------------------------------------------------------

def profile_keys(
    df: pd.DataFrame,
    key: str,
    label: str,
) -> dict:
    """
    Analyse the join key in a single DataFrame.

    Returns a dict with total rows, unique key count, duplicate key count,
    null key count, and the top-5 most frequent keys (multiplicity check).
    """
    total = len(df)
    null_count = int(df[key].isna().sum())
    non_null = df[key].dropna()
    unique_count = int(non_null.nunique())
    duplicate_count = int((non_null.value_counts() > 1).sum())

    top_keys = (
        non_null.value_counts()
        .head(5)
        .reset_index()
        .rename(columns={key: "key_value", "count": "occurrences"})
        .to_dict(orient="records")
    )

    profile = {
        "table": label,
        "total_rows": total,
        "null_keys": null_count,
        "unique_keys": unique_count,
        "keys_with_duplicates": duplicate_count,
        "top_5_key_frequencies": top_keys,
    }

    print(f"\n  [{label}]")
    print(f"    rows          : {total}")
    print(f"    null keys     : {null_count}")
    print(f"    unique keys   : {unique_count}")
    print(f"    keys with dups: {duplicate_count}")
    if duplicate_count:
        print(f"    note          : {duplicate_count} key(s) appear more than once — "
              "expect row multiplication in inner/left/right joins.")

    return profile


# ---------------------------------------------------------------------------
# Step 2 – Execute and validate the merge
# ---------------------------------------------------------------------------

def validated_merge(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    key: str,
    how: JoinType,
    left_label: str = "left",
    right_label: str = "right",
    business_reason: str = "",
) -> tuple[pd.DataFrame, dict]:
    """
    Merge two DataFrames and validate the result exhaustively.

    Parameters
    ----------
    df_left         : Left DataFrame.
    df_right        : Right DataFrame.
    key             : Column name to join on (must exist in both).
    how             : Join type — "inner", "left", "right", or "outer".
    left_label      : Descriptive name for the left table (used in reports).
    right_label     : Descriptive name for the right table (used in reports).
    business_reason : Why this join type was chosen (logged in the report).

    Returns
    -------
    (merged_df, merge_report)
    """
    rows_left = len(df_left)
    rows_right = len(df_right)

    print(f"\n[validated_merge] Merging on '{key}' with how='{how}' …")
    print(f"  {left_label:<20} : {rows_left:>6} rows")
    print(f"  {right_label:<20} : {rows_right:>6} rows")

    # --- Execute ---
    merged = pd.merge(df_left, df_right, on=key, how=how)
    rows_merged = len(merged)
    delta = rows_merged - rows_left

    print(f"  merged result        : {rows_merged:>6} rows  (Δ {delta:+d} vs left)")

    # --- Explain the row count change ---
    if how == "inner":
        explanation = (
            f"Inner join keeps only matched rows. "
            f"Result ({rows_merged}) ≤ min({rows_left}, {rows_right}={min(rows_left, rows_right)})."
        )
    elif how == "left":
        explanation = (
            f"Left join keeps all {rows_left} left rows plus row multiplication "
            f"where right has multiple matches per key. "
            f"Result ({rows_merged}) {'>' if rows_merged > rows_left else '='} left ({rows_left})."
        )
    elif how == "right":
        explanation = (
            f"Right join keeps all {rows_right} right rows plus row multiplication "
            f"where left has multiple matches per key. "
            f"Result ({rows_merged}) {'>' if rows_merged > rows_right else '='} right ({rows_right})."
        )
    else:  # outer
        explanation = (
            f"Outer join keeps all rows from both tables. "
            f"Result ({rows_merged}) ≥ max({rows_left}, {rows_right}={max(rows_left, rows_right)})."
        )

    print(f"  explanation: {explanation}")

    # --- Multiplicity: keys that appear more than once in the result ---
    result_key_counts = merged[key].value_counts()
    multiplied_keys = result_key_counts[result_key_counts > 1]
    multiplicity_count = int(len(multiplied_keys))
    extra_rows_from_multiplicity = int((multiplied_keys - 1).sum()) if not multiplied_keys.empty else 0

    if multiplicity_count:
        print(f"  multiplicity: {multiplicity_count} key(s) appear >1 time "
              f"→ {extra_rows_from_multiplicity} extra row(s) created")

    merge_report = {
        "join_key": key,
        "join_type": how,
        "join_semantics": JOIN_TYPE_DOCS[how],
        "business_reason": business_reason or "Not specified.",
        "row_counts": {
            left_label: rows_left,
            right_label: rows_right,
            "merged": rows_merged,
            "delta_vs_left": delta,
        },
        "count_explanation": explanation,
        "multiplicity": {
            "keys_with_multiple_rows": multiplicity_count,
            "extra_rows_created": extra_rows_from_multiplicity,
        },
    }

    return merged, merge_report


# ---------------------------------------------------------------------------
# Step 3 – Identify unmatched keys
# ---------------------------------------------------------------------------

def find_unmatched_keys(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    key: str,
    left_label: str = "left",
    right_label: str = "right",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Identify keys that exist in one table but not the other.

    Returns
    -------
    (unmatched_left, unmatched_right, unmatched_summary)

    unmatched_left  : Rows in left whose key is absent from right.
    unmatched_right : Rows in right whose key is absent from left.
    unmatched_summary : Counts and sample key values for reporting.
    """
    left_keys = set(df_left[key].dropna())
    right_keys = set(df_right[key].dropna())

    keys_only_in_left = left_keys - right_keys
    keys_only_in_right = right_keys - left_keys
    matched_keys = left_keys & right_keys

    unmatched_left = df_left[df_left[key].isin(keys_only_in_left)].copy().reset_index(drop=True)
    unmatched_right = df_right[df_right[key].isin(keys_only_in_right)].copy().reset_index(drop=True)

    print(f"\n[find_unmatched_keys]")
    print(f"  matched keys              : {len(matched_keys)}")
    print(f"  keys only in {left_label:<12}: {len(keys_only_in_left)}  "
          f"→ {len(unmatched_left)} unmatched row(s)")
    print(f"  keys only in {right_label:<12}: {len(keys_only_in_right)}  "
          f"→ {len(unmatched_right)} unmatched row(s)")

    summary = {
        "matched_keys": len(matched_keys),
        f"keys_only_in_{left_label}": len(keys_only_in_left),
        f"unmatched_rows_in_{left_label}": len(unmatched_left),
        f"sample_unmatched_{left_label}_keys": sorted(
            [str(k) for k in list(keys_only_in_left)[:10]]
        ),
        f"keys_only_in_{right_label}": len(keys_only_in_right),
        f"unmatched_rows_in_{right_label}": len(unmatched_right),
        f"sample_unmatched_{right_label}_keys": sorted(
            [str(k) for k in list(keys_only_in_right)[:10]]
        ),
    }

    return unmatched_left, unmatched_right, summary


# ---------------------------------------------------------------------------
# Step 4 – Post-merge quality checks
# ---------------------------------------------------------------------------

def post_merge_quality(
    merged: pd.DataFrame,
    key: str,
    left_cols: list[str],
    right_cols: list[str],
) -> dict:
    """
    Run quality checks on the merged DataFrame.

    Checks
    ------
    - Null rate per column introduced from the right table (NaN = unmatched).
    - Overall null rate across merged DataFrame.
    - Duplicate row count.
    """
    right_only_cols = [c for c in right_cols if c != key and c in merged.columns]

    null_from_right: dict[str, dict] = {}
    for col in right_only_cols:
        null_count = int(merged[col].isna().sum())
        null_rate = round(null_count / len(merged) * 100, 2) if len(merged) else 0.0
        null_from_right[col] = {"null_count": null_count, "null_rate_pct": null_rate}

    duplicate_count = int(merged.duplicated().sum())
    overall_null_rate = round(
        merged.isna().sum().sum() / (len(merged) * len(merged.columns)) * 100, 2
    ) if len(merged) else 0.0

    quality = {
        "total_merged_rows": len(merged),
        "total_merged_columns": len(merged.columns),
        "duplicate_rows": duplicate_count,
        "overall_null_rate_pct": overall_null_rate,
        "null_from_right_table": null_from_right,
    }

    print(f"\n[post_merge_quality]")
    print(f"  merged rows       : {len(merged)}")
    print(f"  merged columns    : {len(merged.columns)}")
    print(f"  duplicate rows    : {duplicate_count}")
    print(f"  overall null rate : {overall_null_rate} %")
    if null_from_right:
        print(f"  nulls from right table (indicate unmatched rows):")
        for col, stats in null_from_right.items():
            print(f"    {col:<25} {stats['null_count']:>4} null(s)  ({stats['null_rate_pct']} %)")

    return quality


# ---------------------------------------------------------------------------
# Step 5 – Console report
# ---------------------------------------------------------------------------

def print_report(
    merge_report: dict,
    unmatched_summary: dict,
    quality: dict,
    left_label: str,
    right_label: str,
) -> None:
    """Print a comprehensive merge audit report to stdout."""

    sep = "=" * 65

    print(f"\n{sep}")
    print("MULTI-SOURCE MERGE — AUDIT REPORT")
    print(sep)

    rc = merge_report["row_counts"]
    print(f"\n  Join key  : {merge_report['join_key']}")
    print(f"  Join type : {merge_report['join_type'].upper()}")
    print(f"  Reason    : {merge_report['business_reason']}")

    print(f"\n  ── Row counts ──")
    print(f"  {left_label:<25}: {rc[left_label]:>6} rows")
    print(f"  {right_label:<25}: {rc[right_label]:>6} rows")
    print(f"  {'Merged result':<25}: {rc['merged']:>6} rows  (Δ {rc['delta_vs_left']:+d} vs left)")
    print(f"  Explanation: {merge_report['count_explanation']}")

    mult = merge_report["multiplicity"]
    if mult["keys_with_multiple_rows"]:
        print(f"\n  ── Multiplicity ──")
        print(f"  {mult['keys_with_multiple_rows']} key(s) appear >1 time in result "
              f"→ {mult['extra_rows_created']} extra row(s) from fan-out")
        print(f"  This is EXPECTED for one-to-many relationships (e.g. one customer, many orders).")
        print(f"  Verify that downstream aggregations use the correct grain (customer vs order).")

    print(f"\n  ── Unmatched keys ──")
    print(f"  Matched keys                  : {unmatched_summary['matched_keys']}")
    left_uk = unmatched_summary.get(f"keys_only_in_{left_label}", 0)
    right_uk = unmatched_summary.get(f"keys_only_in_{right_label}", 0)
    left_ur = unmatched_summary.get(f"unmatched_rows_in_{left_label}", 0)
    right_ur = unmatched_summary.get(f"unmatched_rows_in_{right_label}", 0)
    print(f"  {left_label} keys with no match : {left_uk}  ({left_ur} row(s)) "
          f"→ output/unmatched_{left_label}.csv")
    print(f"  {right_label} keys with no match: {right_uk}  ({right_ur} row(s)) "
          f"→ output/unmatched_{right_label}.csv")

    samp_l = unmatched_summary.get(f"sample_unmatched_{left_label}_keys", [])
    samp_r = unmatched_summary.get(f"sample_unmatched_{right_label}_keys", [])
    if samp_l:
        print(f"  Sample unmatched {left_label} keys  : {samp_l}")
    if samp_r:
        print(f"  Sample unmatched {right_label} keys : {samp_r}")

    print(f"\n  ── Post-merge quality ──")
    print(f"  Duplicate rows      : {quality['duplicate_rows']}")
    print(f"  Overall null rate   : {quality['overall_null_rate_pct']} %")
    if quality["null_from_right_table"]:
        print(f"  Nulls from right (unmatched indicators):")
        for col, s in quality["null_from_right_table"].items():
            print(f"    {col:<28}: {s['null_count']:>4} ({s['null_rate_pct']} %)")

    print(f"\n  ── Join semantics reference ──")
    jt = merge_report["join_type"]
    doc = JOIN_TYPE_DOCS[jt]
    print(f"  Keeps   : {doc['keeps']}")
    print(f"  Size    : {doc['result_size']}")
    print(f"  Use when: {doc['use_when']}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Step 6 – Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    merged: pd.DataFrame,
    unmatched_left: pd.DataFrame,
    unmatched_right: pd.DataFrame,
    merge_report: dict,
    unmatched_summary: dict,
    quality: dict,
    left_label: str,
    right_label: str,
    output_csv: str = "output/merged.csv",
) -> None:
    """
    Write all merge outputs to output/.

    Files created
    -------------
    output/merged.csv                  — full merged dataset
    output/unmatched_<left>.csv        — left rows with no match in right
    output/unmatched_<right>.csv       — right rows with no match in left
    output/merge_audit_report.json     — complete join decision + validation log
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_path = Path(output_csv)
    merged.to_csv(merged_path, index=False)
    print(f"[save_outputs] Merged dataset saved      → {merged_path}  ({len(merged)} rows)")

    ul_path = output_dir / f"unmatched_{left_label}.csv"
    unmatched_left.to_csv(ul_path, index=False)
    print(f"[save_outputs] Unmatched {left_label:<14} → {ul_path}  ({len(unmatched_left)} rows)")

    ur_path = output_dir / f"unmatched_{right_label}.csv"
    unmatched_right.to_csv(ur_path, index=False)
    print(f"[save_outputs] Unmatched {right_label:<14} → {ur_path}  ({len(unmatched_right)} rows)")

    full_report = {
        "merge_report": merge_report,
        "unmatched_summary": unmatched_summary,
        "post_merge_quality": quality,
    }

    report_path = output_dir / "merge_audit_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(full_report, fh, indent=4, default=str)
    print(f"[save_outputs] Merge audit report saved  → {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Source Merging & Join Validation — NexPulse"
    )
    parser.add_argument("--left",  dest="left_path",  help="Path to the left CSV.")
    parser.add_argument("--right", dest="right_path", help="Path to the right CSV.")
    parser.add_argument(
        "--key", dest="key", default="customer_id",
        help="Column name to join on (default: customer_id).",
    )
    parser.add_argument(
        "--how", dest="how", default="left",
        choices=["inner", "left", "right", "outer"],
        help="Join type (default: left).",
    )
    parser.add_argument(
        "--left-label",  dest="left_label",  default="customers",
        help="Human-readable name for the left table.",
    )
    parser.add_argument(
        "--right-label", dest="right_label", default="orders",
        help="Human-readable name for the right table.",
    )
    parser.add_argument(
        "--reason", dest="reason", default="",
        help="Business reason for the chosen join type (logged in the report).",
    )
    parser.add_argument(
        "--output", dest="output_path", default="output/merged.csv",
        help="Destination path for the merged CSV.",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use the built-in demo dataset (customers + orders).",
    )
    return parser.parse_args()


def load_inputs(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Load both tables according to CLI arguments."""
    if args.demo or not (args.left_path and args.right_path):
        print("Using built-in demo dataset (customers + orders).")
        return (
            build_demo_customers(),
            build_demo_orders(),
            "customers",
            "orders",
        )

    df_left = pd.read_csv(args.left_path)
    df_right = pd.read_csv(args.right_path)
    print(f"Left  loaded: {args.left_path}  ({len(df_left)} rows)")
    print(f"Right loaded: {args.right_path}  ({len(df_right)} rows)")
    return df_left, df_right, args.left_label, args.right_label


def main() -> None:
    args = parse_args()

    # 1. Load
    df_left, df_right, left_label, right_label = load_inputs(args)
    key = args.key
    how: JoinType = args.how  # type: ignore[assignment]
    business_reason = args.reason or (
        f"Left join chosen: '{left_label}' is the source of truth. "
        f"'{right_label}' enriches each record. "
        f"Unmatched {left_label} rows are retained with NaN for {right_label} columns."
    )

    print(f"\nLeft  : {left_label}  ({len(df_left)} rows × {len(df_left.columns)} cols)")
    print(f"Right : {right_label}  ({len(df_right)} rows × {len(df_right.columns)} cols)")
    print(f"Key   : {key}")
    print(f"How   : {how}")

    # 2. Pre-merge key profiling
    print("\n[pre-merge key profile]")
    left_profile  = profile_keys(df_left,  key, left_label)
    right_profile = profile_keys(df_right, key, right_label)

    # 3. Merge + validation
    merged, merge_report = validated_merge(
        df_left, df_right,
        key=key, how=how,
        left_label=left_label,
        right_label=right_label,
        business_reason=business_reason,
    )

    # 4. Unmatched keys
    unmatched_left, unmatched_right, unmatched_summary = find_unmatched_keys(
        df_left, df_right,
        key=key,
        left_label=left_label,
        right_label=right_label,
    )

    # 5. Post-merge quality
    quality = post_merge_quality(
        merged, key,
        left_cols=list(df_left.columns),
        right_cols=list(df_right.columns),
    )

    # 6. Console report
    print_report(merge_report, unmatched_summary, quality, left_label, right_label)

    # 7. Persist
    save_outputs(
        merged,
        unmatched_left,
        unmatched_right,
        merge_report,
        unmatched_summary,
        quality,
        left_label,
        right_label,
        output_csv=args.output_path,
    )

    print(f"\nMerge pipeline complete.")
    print(f"  {left_label:<12}: {len(df_left)} rows in")
    print(f"  {right_label:<12}: {len(df_right)} rows in")
    print(f"  Merged       : {len(merged)} rows out")
    print(f"  Unmatched {left_label:<4}: {len(unmatched_left)} rows")
    print(f"  Unmatched {right_label:<4}: {len(unmatched_right)} rows")


if __name__ == "__main__":
    main()
