import json
import argparse
from pathlib import Path

import pandas as pd


RAW_DATE_FORMAT = "%Y-%m-%d"
TEXT_CLEANING_RULES = {
    "customer_name": {"lowercase": True, "strip": True},
    "city": {"lowercase": True, "strip": True, "remove_special": True},
    "product_category": {
        "lowercase": True,
        "strip": True,
        "remove_special": True,
        "mapping": {
            "electronics": "electronics",
            "electro nics": "electronics",
            "home garden": "home garden",
            "home and garden": "home garden",
        },
    },
    "segment": {
        "lowercase": True,
        "strip": True,
        "remove_special": True,
        "mapping": {
            "b2b": "b2b",
            "b 2 b": "b2b",
            "business to business": "b2b",
            "businesstobusiness": "b2b",
            "retail": "retail",
        },
    },
    "status": {
        "lowercase": True,
        "strip": True,
    },
    "source": {
        "lowercase": True,
        "strip": True,
    },
}


def _coerce_currency(series):
    """Convert currency text into numeric values."""
    cleaned = series.astype("string").str.replace(r"[$,]", "", regex=True)
    return pd.to_numeric(cleaned, errors="raise")


def _coerce_boolean(series):
    """Convert 0/1 markers into boolean values."""
    numeric = pd.to_numeric(series, errors="raise")

    if not numeric.dropna().isin([0, 1]).all():
        raise ValueError("Boolean columns must contain only 0/1 values")

    return numeric.astype("boolean")


def clean_text_column(series, lowercase=True, strip=True, remove_special=False, mapping=None):
    """Apply reusable text normalisation steps to a single column."""
    result = series.astype("string")

    if strip:
        result = result.str.strip()

    if lowercase:
        result = result.str.lower()

    if remove_special:
        result = result.str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)

    if mapping:
        mapped = result.map(mapping)
        result = mapped.fillna(result)

    return result


def build_demo_data():
    """Create a small dataset that contains type, text, and duplicate issues."""
    return pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 3],
            "customer_name": [" John ", "john", "Ana-Maria", "ana maria", "Marta  "],
            "city": [" São Paulo ", "Sao Paulo", "Montréal", "MONTREAL!", " Lisbon"],
            "product_category": [
                " Electronics ",
                "electronics",
                "ELECTRONICS",
                "Electro nics",
                "Home & Garden",
            ],
            "segment": ["B2B", "b 2 b", "business-to-business", "Retail ", " retail"],
            "transaction_date": [
                "2025-01-15",
                "2025-01-15",
                "2025-01-20",
                "2025-01-20",
                "2025-02-01",
            ],
            "amount": ["$150.50", "$150.50", "$200.00", "$200.00", "$75.25"],
            "is_active": [0, 0, 1, 1, 0],
            "status": ["completed", "completed", "pending", None, "completed"],
            "source": ["crm", "crm", "crm", "import", "crm"],
        }
    )


def enforce_types(df):
    """Apply explicit type conversions and log the dtype changes."""
    before = df.dtypes.astype(str).to_dict()
    conversion_log = []

    type_rules = {
        "customer_id": lambda series: pd.to_numeric(series, errors="raise").astype("Int64"),
        "transaction_amount": _coerce_currency,
        "amount": _coerce_currency,
        "quantity": lambda series: pd.to_numeric(series, errors="raise").astype("Int64"),
        "transaction_date": lambda series: pd.to_datetime(series, format=RAW_DATE_FORMAT, errors="raise"),
        "last_updated": lambda series: pd.to_datetime(series, format=RAW_DATE_FORMAT, errors="raise"),
    }

    for column in df.columns:
        original_dtype = str(df[column].dtype)

        if column in type_rules:
            df[column] = type_rules[column](df[column])
        elif column.startswith(("is_", "has_")):
            df[column] = _coerce_boolean(df[column])
        else:
            df[column] = df[column].astype("string")

        updated_dtype = str(df[column].dtype)

        if original_dtype != updated_dtype:
            conversion_log.append(
                {
                    "column": column,
                    "before": original_dtype,
                    "after": updated_dtype,
                }
            )

    after = df.dtypes.astype(str).to_dict()
    return df, before, after, conversion_log


def _null_count(series):
    return int(series.isna().sum())


def deduplicate_records(df, key_columns):
    """Remove exact and near duplicates while preserving the most complete record."""
    original = df.copy()
    removed_frames = []
    duplicate_log = []

    exact_duplicate_mask = original.duplicated(keep="first")
    exact_duplicates = original.loc[exact_duplicate_mask].copy()

    if not exact_duplicates.empty:
        exact_duplicates.loc[:, "duplicate_type"] = "exact"
        exact_duplicates.loc[:, "duplicate_key"] = "|".join(key_columns)
        exact_duplicates.loc[:, "dedupe_reason"] = "Exact duplicate row"
        removed_frames.append(exact_duplicates)

    working = original.drop_duplicates(keep="first").copy()

    if key_columns:
        working["_null_count"] = working.apply(_null_count, axis=1)
        working["_row_order"] = range(len(working))

        grouped = working.groupby(key_columns, dropna=False, sort=False)
        keep_indexes = []

        for key_values, group in grouped:
            if len(group) == 1:
                keep_indexes.append(group.index[0])
                continue

            winner = group.sort_values(
                by=["_null_count", "_row_order"],
                ascending=[True, True],
            ).index[0]
            keep_indexes.append(winner)

            for index, row in group.drop(index=winner).iterrows():
                duplicate_log.append(
                    {
                        "duplicate_type": "near",
                        "duplicate_key": {column: row[column] for column in key_columns},
                        "kept_index": int(winner),
                        "removed_index": int(index),
                        "dedupe_reason": "Same business key; kept record with fewest nulls",
                    }
                )

        near_duplicate_mask = ~working.index.isin(keep_indexes)
        near_duplicates = working.loc[near_duplicate_mask].copy()

        if not near_duplicates.empty:
            near_duplicates.loc[:, "duplicate_type"] = "near"
            near_duplicates.loc[:, "duplicate_key"] = near_duplicates[key_columns].astype(str).agg("|".join, axis=1)
            near_duplicates.loc[:, "dedupe_reason"] = "Same business key; kept record with fewest nulls"
            removed_frames.append(near_duplicates)

        deduplicated = working.loc[keep_indexes].drop(columns=["_null_count", "_row_order"])
    else:
        deduplicated = working.drop(columns=["_null_count", "_row_order"])

    removed_records = pd.concat(removed_frames, ignore_index=True) if removed_frames else pd.DataFrame(columns=list(original.columns) + ["duplicate_type", "duplicate_key", "dedupe_reason"])

    summary = {
        "rows_before": int(len(original)),
        "exact_duplicates_found": int(len(exact_duplicates)),
        "near_duplicates_found": int(len(duplicate_log)),
        "rows_after": int(len(deduplicated)),
        "rows_removed": int(len(original) - len(deduplicated)),
        "removal_pct": round(((len(original) - len(deduplicated)) / len(original)) * 100, 2) if len(original) else 0.0,
        "key_columns": key_columns,
    }

    audit_log = duplicate_log

    return deduplicated.reset_index(drop=True), removed_records.reset_index(drop=True), summary, audit_log


def normalize_text_columns(df):
    """Clean any configured text columns that exist in the dataset."""
    cleaned = df.copy()
    summary = []

    for column, rule in TEXT_CLEANING_RULES.items():
        if column not in cleaned.columns:
            continue

        before_non_null = cleaned[column].dropna()
        before_unique = int(before_non_null.nunique())
        before_sample = [str(value) for value in before_non_null.head(3).tolist()]

        cleaned[column] = clean_text_column(
            cleaned[column],
            lowercase=rule.get("lowercase", True),
            strip=rule.get("strip", True),
            remove_special=rule.get("remove_special", False),
            mapping=rule.get("mapping"),
        )

        after_non_null = cleaned[column].dropna()
        after_unique = int(after_non_null.nunique())
        after_sample = [str(value) for value in after_non_null.head(3).tolist()]

        summary.append(
            {
                "column": column,
                "before_unique": before_unique,
                "after_unique": after_unique,
                "before_sample": before_sample,
                "after_sample": after_sample,
            }
        )

    return cleaned, summary


def ingest_data(filepath):
    """
    Load data from a CSV file.

    Input:
        filepath (str): Path to the CSV file.

    Returns:
        Pandas DataFrame containing the raw data.
    """
    # Read CSV file
    df = pd.read_csv(filepath)
    return df


def process_data(df):
    """
    Clean and process the dataset.

    Input:
        df (DataFrame): Raw dataset.

    Returns:
        Cleaned DataFrame.
    """
    # Normalise text columns before downstream analysis
    df, text_summary = normalize_text_columns(df)

    # Enforce explicit types before any downstream analysis
    df, before_dtypes, after_dtypes, conversion_log = enforce_types(df)

    # Remove exact and near duplicates using the business key
    deduplicated_df, removed_records, dedupe_summary, audit_log = deduplicate_records(
        df,
        key_columns=["customer_id", "transaction_date"],
    )

    print("\n===== TYPE ENFORCEMENT =====")
    print("Before dtypes:")
    print(json.dumps(before_dtypes, indent=4))
    print("After dtypes:")
    print(json.dumps(after_dtypes, indent=4))

    print("\n===== TEXT NORMALISATION =====")
    print(json.dumps(text_summary, indent=4, default=str))

    if conversion_log:
        print("\nConversions applied:")
        print(json.dumps(conversion_log, indent=4))
    else:
        print("\nNo explicit type conversions were needed.")

    print("\n===== DEDUPLICATION =====")
    print(json.dumps(dedupe_summary, indent=4, default=str))

    return deduplicated_df, removed_records, dedupe_summary, audit_log


def output_results(df, output_path, removed_records=None, summary=None, audit_log=None):
    """
    Save processed data.

    Input:
        df (DataFrame): Processed dataset.
        output_path (str): Output CSV path.
    """
    # Save processed data
    df.to_csv(output_path, index=False)

    if removed_records is not None:
        removed_path = Path("output/removed_duplicates_audit.csv")
        removed_records.to_csv(removed_path, index=False)

    if summary is not None:
        summary_path = Path("output/deduplication_summary.json")
        with summary_path.open("w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, indent=4, default=str)

    if audit_log is not None:
        audit_path = Path("output/deduplication_audit_log.json")
        with audit_path.open("w", encoding="utf-8") as file_handle:
            json.dump(audit_log, file_handle, indent=4, default=str)


def parse_args():
    """Parse CLI arguments for reusable workflow execution."""
    parser = argparse.ArgumentParser(
        description="Run type enforcement and deduplication on a dataset."
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        help="Path to a CSV file. If omitted, a demo dataset is used.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default="output/processed.csv",
        help="Path for the cleaned CSV output.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force the built-in demo dataset even if --input is provided.",
    )

    return parser.parse_args()


def load_dataset(args):
    """Load the chosen dataset for the workflow."""
    if args.demo or not args.input_path:
        print("Using built-in demo dataset.")
        return build_demo_data(), "demo dataset"

    input_path = Path(args.input_path)
    print(f"Loading data from {input_path}")
    return ingest_data(str(input_path)), str(input_path)

if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output_path)

    data, source_label = load_dataset(args)
    processed, removed_records, summary, audit_log = process_data(data)
    output_results(processed, str(output_path), removed_records, summary, audit_log)

    print("Data successfully processed")
    print(f"Rows processed: {len(processed)}")
    print(f"Source used: {source_label}")
    print(f"Output saved to {output_path}")