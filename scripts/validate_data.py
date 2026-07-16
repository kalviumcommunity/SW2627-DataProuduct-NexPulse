"""
validate_data.py
----------------
Data Consistency & Validation Rules for the NexPulse workflow.

Responsibilities
----------------
1. Define a rule registry covering all five validation categories:
      range_check       — values must fall within expected numeric/date bounds.
      null_constraint   — critical columns must never be null.
      format_pattern    — values must match a regex pattern (email, phone, etc.).
      referential       — a column value must exist in a reference set.
      business_rule     — domain-specific cross-column logic checks.
2. Run every rule against each row and record pass/fail per rule.
3. Isolate failing records into output/validation_failures.csv.
4. Produce a structured validation report: per-rule pass/fail counts,
   failure reasons per row, and a full audit trail.
5. Return only records that pass all rules for downstream analysis.

Usage
-----
    # Built-in demo dataset:
    python scripts/validate_data.py --demo

    # Real CSV:
    python scripts/validate_data.py --input data/raw/customers.csv

    # Custom output path:
    python scripts/validate_data.py --demo --output output/validated_clean.csv
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RuleCategory = Literal[
    "range_check",
    "null_constraint",
    "format_pattern",
    "referential",
    "business_rule",
]


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

@dataclass
class ValidationRule:
    """
    Describes a single validation check.

    Attributes
    ----------
    rule_id   : Unique identifier used in the report and flag columns.
    category  : One of the five validation categories.
    description: Human-readable description for the audit report.
    check_fn  : Callable(df) → boolean Series.
                True  = row PASSES the rule.
                False = row FAILS the rule.
    """
    rule_id: str
    category: RuleCategory
    description: str
    check_fn: Callable[[pd.DataFrame], pd.Series]
    columns_checked: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Demo dataset
# ---------------------------------------------------------------------------

def build_demo_data() -> pd.DataFrame:
    """
    Return a customer / campaign dataset with intentional violations that
    exercise all five validation categories.
    """
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    return pd.DataFrame(
        {
            "customer_id": [1, 2, None, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "email": [
                "alice@example.com",
                "bob_no_at_sign",          # bad format — missing @
                "carol@example.com",
                None,                       # null constraint violation
                "diana@example.com",
                "eve@example.com",
                "frank@example.com",
                "grace@example.com",
                "heidi@",                  # bad format — no domain
                "ivan@example.com",
                "judy@example.com",
                "mallory@example.com",
            ],
            "phone": [
                "5551234567",
                "555-123-4567",            # bad format — contains dashes
                "5559876543",
                "5554561234",
                "NOT_A_PHONE",             # bad format — non-numeric
                "5550001111",
                "5552223333",
                "5554445555",
                "5556667777",
                "5558889999",
                "5550123456",
                "5557654321",
            ],
            "birth_date": [
                "1990-06-15",
                "2050-01-01",              # future date — range violation
                "1985-03-22",
                "1978-11-05",
                "1800-07-04",              # too old — range violation
                "1995-12-31",
                "2000-08-19",
                "1970-04-10",
                "1988-09-27",
                "2002-02-14",
                "1993-07-07",
                "1967-10-30",
            ],
            "age": [34, 25, 39, 46, 224, 29, 24, 54, 36, 22, 31, 57],
            # age=224 is impossible                                    ↑
            "price": [19.99, 5.00, -3.50, 120.00, 45.00,              # -3.50 negative
                      0.00, 89.99, 12.50, 55.00, 7.75, 33.00, 199.99],
            "campaign_start_date": [
                "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01",
                "2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01",
                "2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01",
            ],
            "campaign_end_date": [
                "2025-01-31", "2025-01-15",   # end BEFORE start — business rule
                "2025-03-31", "2025-04-30",
                "2025-05-31", "2025-06-30",
                "2025-07-31", "2025-07-25",   # end BEFORE start — business rule
                "2025-09-30", "2025-10-31",
                "2025-11-30", "2025-12-31",
            ],
            "country_code": [
                "US", "GB", "FR", "ZZ",       # ZZ is not a valid ISO code
                "DE", "IN", "US", "GB",
                "XQ",                          # XQ is not a valid ISO code
                "AU", "CA", "JP",
            ],
            "discount_pct": [10, 5, 0, 20, 15, 55,  # 55 % > 50 % max
                             30, 10, 5, 0, 25, 60],  # 60 % > 50 % max
        }
    )


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

VALID_COUNTRY_CODES: frozenset[str] = frozenset(
    [
        "AF", "AL", "DZ", "AD", "AO", "AR", "AM", "AU", "AT", "AZ",
        "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ", "BT", "BO",
        "BA", "BW", "BR", "BN", "BG", "BF", "BI", "CV", "KH", "CM",
        "CA", "CF", "TD", "CL", "CN", "CO", "KM", "CG", "CR", "HR",
        "CU", "CY", "CZ", "DK", "DJ", "DM", "DO", "EC", "EG", "SV",
        "GQ", "ER", "EE", "SZ", "ET", "FJ", "FI", "FR", "GA", "GM",
        "GE", "DE", "GH", "GR", "GD", "GT", "GN", "GW", "GY", "HT",
        "HN", "HU", "IS", "IN", "ID", "IR", "IQ", "IE", "IL", "IT",
        "JM", "JP", "JO", "KZ", "KE", "KI", "KP", "KR", "KW", "KG",
        "LA", "LV", "LB", "LS", "LR", "LY", "LI", "LT", "LU", "MG",
        "MW", "MY", "MV", "ML", "MT", "MH", "MR", "MU", "MX", "FM",
        "MD", "MC", "MN", "ME", "MA", "MZ", "MM", "NA", "NR", "NP",
        "NL", "NZ", "NI", "NE", "NG", "NO", "OM", "PK", "PW", "PA",
        "PG", "PY", "PE", "PH", "PL", "PT", "QA", "RO", "RU", "RW",
        "KN", "LC", "VC", "WS", "SM", "ST", "SA", "SN", "RS", "SC",
        "SL", "SG", "SK", "SI", "SB", "SO", "ZA", "SS", "ES", "LK",
        "SD", "SR", "SE", "CH", "SY", "TW", "TJ", "TZ", "TH", "TL",
        "TG", "TO", "TT", "TN", "TR", "TM", "TV", "UG", "UA", "AE",
        "GB", "US", "UY", "UZ", "VU", "VE", "VN", "YE", "ZM", "ZW",
    ]
)


# ---------------------------------------------------------------------------
# Rule registry factory
# ---------------------------------------------------------------------------

def build_rule_registry() -> list[ValidationRule]:
    """
    Construct and return the full list of validation rules.

    Each rule is self-contained: the check_fn receives the full DataFrame
    and returns a boolean Series (True = pass).
    """
    today = pd.Timestamp.now().normalize()
    min_birth = pd.Timestamp("1920-01-01")

    rules: list[ValidationRule] = [

        # ── NULL CONSTRAINTS ──────────────────────────────────────────────

        ValidationRule(
            rule_id="null_customer_id",
            category="null_constraint",
            description="customer_id must never be null — it is the primary key.",
            columns_checked=["customer_id"],
            check_fn=lambda df: df["customer_id"].notna()
            if "customer_id" in df.columns
            else pd.Series(True, index=df.index),
        ),

        ValidationRule(
            rule_id="null_email",
            category="null_constraint",
            description="email must never be null — required for all communications.",
            columns_checked=["email"],
            check_fn=lambda df: df["email"].notna()
            if "email" in df.columns
            else pd.Series(True, index=df.index),
        ),

        # ── FORMAT PATTERNS ───────────────────────────────────────────────

        ValidationRule(
            rule_id="format_email",
            category="format_pattern",
            description="email must contain '@' and at least one '.' after it.",
            columns_checked=["email"],
            check_fn=lambda df: df["email"].astype(str).str.match(
                r"^[^@\s]+@[^@\s]+\.[^@\s]+$", na=False
            )
            if "email" in df.columns
            else pd.Series(True, index=df.index),
        ),

        ValidationRule(
            rule_id="format_phone",
            category="format_pattern",
            description="phone must be exactly 10 digits with no separators.",
            columns_checked=["phone"],
            check_fn=lambda df: df["phone"].astype(str).str.match(
                r"^\d{10}$", na=False
            )
            if "phone" in df.columns
            else pd.Series(True, index=df.index),
        ),

        # ── RANGE CHECKS ─────────────────────────────────────────────────

        ValidationRule(
            rule_id="range_birth_date",
            category="range_check",
            description=(
                f"birth_date must be between {min_birth.date()} and today "
                f"({today.date()}). Future dates and dates before 1920 are invalid."
            ),
            columns_checked=["birth_date"],
            check_fn=lambda df, _min=min_birth, _max=today: (
                pd.to_datetime(df["birth_date"], errors="coerce")
                .between(_min, _max)
                .fillna(False)
            )
            if "birth_date" in df.columns
            else pd.Series(True, index=df.index),
        ),

        ValidationRule(
            rule_id="range_age",
            category="range_check",
            description="age must be between 0 and 150 — values outside this range are impossible.",
            columns_checked=["age"],
            check_fn=lambda df: (
                pd.to_numeric(df["age"], errors="coerce").between(0, 150).fillna(False)
            )
            if "age" in df.columns
            else pd.Series(True, index=df.index),
        ),

        ValidationRule(
            rule_id="range_price_non_negative",
            category="range_check",
            description="price must be >= 0. Negative prices indicate data entry errors.",
            columns_checked=["price"],
            check_fn=lambda df: (
                pd.to_numeric(df["price"], errors="coerce").ge(0).fillna(False)
            )
            if "price" in df.columns
            else pd.Series(True, index=df.index),
        ),

        ValidationRule(
            rule_id="range_discount_pct",
            category="range_check",
            description="discount_pct must be between 0 and 50. Discounts above 50 % are not approved.",
            columns_checked=["discount_pct"],
            check_fn=lambda df: (
                pd.to_numeric(df["discount_pct"], errors="coerce")
                .between(0, 50)
                .fillna(False)
            )
            if "discount_pct" in df.columns
            else pd.Series(True, index=df.index),
        ),

        # ── REFERENTIAL INTEGRITY ─────────────────────────────────────────

        ValidationRule(
            rule_id="ref_country_code",
            category="referential",
            description=(
                "country_code must be a valid ISO 3166-1 alpha-2 code. "
                "Unknown codes cannot be mapped to regions."
            ),
            columns_checked=["country_code"],
            check_fn=lambda df: (
                df["country_code"].isin(VALID_COUNTRY_CODES)
            )
            if "country_code" in df.columns
            else pd.Series(True, index=df.index),
        ),

        # ── BUSINESS RULES ────────────────────────────────────────────────

        ValidationRule(
            rule_id="biz_campaign_date_order",
            category="business_rule",
            description=(
                "campaign_end_date must be >= campaign_start_date. "
                "A campaign cannot end before it starts."
            ),
            columns_checked=["campaign_start_date", "campaign_end_date"],
            check_fn=lambda df: (
                pd.to_datetime(df["campaign_end_date"], errors="coerce")
                >= pd.to_datetime(df["campaign_start_date"], errors="coerce")
            ).fillna(False)
            if "campaign_start_date" in df.columns and "campaign_end_date" in df.columns
            else pd.Series(True, index=df.index),
        ),

    ]

    return rules


# ---------------------------------------------------------------------------
# Core validation engine
# ---------------------------------------------------------------------------

def run_validation(
    df: pd.DataFrame,
    rules: list[ValidationRule],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Apply every rule in *rules* to *df*.

    For each rule a boolean flag column ``pass_<rule_id>`` is appended.
    A final ``passes_all_checks`` column marks rows that pass every rule.

    Parameters
    ----------
    df    : Raw input DataFrame.
    rules : List of ValidationRule objects from build_rule_registry().

    Returns
    -------
    (annotated_df, clean_df, failures_df, rule_report)

    annotated_df : Full DataFrame with per-rule flag columns appended.
    clean_df     : Rows where passes_all_checks is True (ready for analysis).
    failures_df  : Rows where passes_all_checks is False (isolated for audit).
    rule_report  : List of dicts; one entry per rule with pass/fail counts.
    """
    annotated = df.copy()
    rule_report: list[dict] = []
    flag_columns: list[str] = []

    for rule in rules:
        flag_col = f"pass_{rule.rule_id}"
        flag_columns.append(flag_col)

        try:
            result_mask = rule.check_fn(annotated)
            # Ensure boolean dtype, fill any NaN with False (conservative)
            annotated[flag_col] = result_mask.astype(bool).fillna(False)
        except Exception as exc:  # noqa: BLE001
            # If the check itself errors, mark all rows as failing that rule
            print(f"  [WARNING] Rule '{rule.rule_id}' raised an exception: {exc}")
            annotated[flag_col] = False

        pass_count = int(annotated[flag_col].sum())
        fail_count = int((~annotated[flag_col]).sum())

        rule_report.append(
            {
                "rule_id": rule.rule_id,
                "category": rule.category,
                "description": rule.description,
                "columns_checked": rule.columns_checked,
                "total_rows": len(annotated),
                "passed": pass_count,
                "failed": fail_count,
                "pass_rate_pct": round(pass_count / len(annotated) * 100, 2)
                if len(annotated) > 0
                else 0.0,
            }
        )

        status = "✓" if fail_count == 0 else "✗"
        print(
            f"  [{status}] {rule.rule_id:<35}  pass={pass_count}  fail={fail_count}"
        )

    # Master pass/fail column
    annotated["passes_all_checks"] = annotated[flag_columns].all(axis=1)

    # Build a per-row failure reason list for the failures file
    def _failure_reasons(row: pd.Series) -> str:
        failed_rules = [
            col.removeprefix("pass_")
            for col in flag_columns
            if not row[col]
        ]
        return "; ".join(failed_rules) if failed_rules else ""

    annotated["failure_reasons"] = annotated.apply(_failure_reasons, axis=1)

    clean_df = (
        annotated[annotated["passes_all_checks"]]
        .drop(columns=flag_columns + ["passes_all_checks", "failure_reasons"])
        .reset_index(drop=True)
    )

    failures_df = (
        annotated[~annotated["passes_all_checks"]]
        .reset_index(drop=True)
    )

    return annotated, clean_df, failures_df, rule_report


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(
    df_raw: pd.DataFrame,
    clean_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    rule_report: list[dict],
) -> None:
    """Print a structured validation summary to stdout."""

    sep = "=" * 65
    total = len(df_raw)
    passed = len(clean_df)
    failed = len(failures_df)

    print(f"\n{sep}")
    print("DATA CONSISTENCY & VALIDATION — REPORT")
    print(sep)

    print(f"\nTotal records   : {total}")
    print(f"Passed all rules: {passed}  ({round(passed / total * 100, 1)} %)")
    print(f"Failed ≥1 rule  : {failed}  ({round(failed / total * 100, 1)} %)")

    # Group by category
    categories: dict[str, list[dict]] = {}
    for entry in rule_report:
        categories.setdefault(entry["category"], []).append(entry)

    category_order = [
        "null_constraint",
        "format_pattern",
        "range_check",
        "referential",
        "business_rule",
    ]

    for cat in category_order:
        if cat not in categories:
            continue
        print(f"\n  ── {cat.replace('_', ' ').upper()} ──")
        for entry in categories[cat]:
            status = "✓" if entry["failed"] == 0 else "✗"
            print(
                f"  [{status}] {entry['rule_id']:<35} "
                f"pass={entry['passed']:>4}  "
                f"fail={entry['failed']:>4}  "
                f"({entry['pass_rate_pct']} %)"
            )
            print(f"       {entry['description']}")

    if not failures_df.empty:
        print(f"\n  ── FAILED RECORDS (first 10) ──")
        display_cols = ["failure_reasons"] + [
            c for c in df_raw.columns if c in failures_df.columns
        ][:6]
        print(failures_df[display_cols].head(10).to_string(index=True))

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Persist outputs
# ---------------------------------------------------------------------------

def save_outputs(
    clean_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    rule_report: list[dict],
    output_csv: str = "output/validated_clean.csv",
) -> None:
    """
    Write validated clean data, failure records, and JSON report to output/.

    Files created
    -------------
    output/validated_clean.csv         — records that pass all checks
    output/validation_failures.csv     — isolated failing records with reasons
    output/validation_report.json      — per-rule pass/fail summary
    """
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_csv)
    clean_df.to_csv(csv_path, index=False)
    print(f"[save_outputs] Clean data saved      → {csv_path}  ({len(clean_df)} rows)")

    failures_path = output_dir / "validation_failures.csv"
    failures_df.to_csv(failures_path, index=False)
    print(f"[save_outputs] Failures saved         → {failures_path}  ({len(failures_df)} rows)")

    report_path = output_dir / "validation_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(rule_report, fh, indent=4, default=str)
    print(f"[save_outputs] Validation report saved → {report_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data Consistency & Validation Rules — NexPulse"
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to a CSV file. Omit to use the built-in demo dataset.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="output/validated_clean.csv",
        help="Destination path for the validated clean CSV.",
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

    # 1. Load
    df_raw, source_label = load_input(args)
    print(f"\nSource  : {source_label}")
    print(f"Shape   : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    print(f"Columns : {list(df_raw.columns)}\n")

    # 2. Build rule registry
    rules = build_rule_registry()
    print(f"Validation rules loaded: {len(rules)}\n")

    # 3. Run all checks
    print("Running validation checks …")
    annotated_df, clean_df, failures_df, rule_report = run_validation(df_raw, rules)

    # 4. Console report
    print_report(df_raw, clean_df, failures_df, rule_report)

    # 5. Persist
    save_outputs(clean_df, failures_df, rule_report, args.output_path)

    print(f"\nValidation complete.")
    print(f"  Records passed : {len(clean_df)} → ready for analysis")
    print(f"  Records failed : {len(failures_df)} → output/validation_failures.csv")
    print(f"  Rules evaluated: {len(rules)}")


if __name__ == "__main__":
    main()
