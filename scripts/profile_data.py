import json
import os

import numpy as np
import pandas as pd


def profile_nulls_and_duplicates(df):
    """
    Compute null counts, null percentages and duplicate metrics.
    """
    profile = {
        "null_counts": {},
        "null_percentages": {},
        "exact_duplicate_count": int(df.duplicated().sum()),
    }

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round((null_count / len(df)) * 100, 2)

        profile["null_counts"][col] = null_count
        profile["null_percentages"][col] = null_pct

    profile["duplicate_percentage"] = round(
        (int(df.duplicated().sum()) / len(df)) * 100, 2
    )

    return profile


def profile_numerical_columns(df):
    """
    Generate statistics for numerical columns.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    stats = {}

    for col in numerical_cols:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(round(df[col].mean(), 2)),
            "median": float(round(df[col].median(), 2)),
            "std": float(round(df[col].std(), 2)),
            "null_count": int(df[col].isnull().sum()),
        }

    return stats


def profile_categorical_columns(df, top_n=5):
    """
    Generate categorical statistics.
    """
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns

    profile = {}

    for col in categorical_cols:
        profile[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": {
                str(k): int(v)
                for k, v in df[col].value_counts().head(top_n).to_dict().items()
            },
            "null_count": int(df[col].isnull().sum()),
        }

    return profile


def identify_quality_issues(df, null_threshold=30):
    """
    Identify common quality issues.
    """
    issues = []

    null_pcts = (df.isnull().sum() / len(df)) * 100

    for col, pct in null_pcts.items():
        if pct > null_threshold:
            issues.append(
                {
                    "type": "High Nulls",
                    "column": col,
                    "severity": "HIGH",
                    "value": f"{pct:.1f}%",
                    "recommendation": "Consider imputation or dropping the column",
                }
            )

    dup_count = int(df.duplicated().sum())

    if dup_count > 0:
        issues.append(
            {
                "type": "Duplicate Records",
                "column": "Entire Row",
                "severity": "MEDIUM",
                "value": dup_count,
                "recommendation": "Remove duplicate rows before analysis",
            }
        )

    if "amount" in df.columns:
        if (df["amount"] < 0).any():
            issues.append(
                {
                    "type": "Negative Amount",
                    "column": "amount",
                    "severity": "HIGH",
                    "value": "Negative values found",
                    "recommendation": "Investigate invalid monetary values",
                }
            )

    return issues


def generate_profile_report(df):
    """
    Generate complete profiling report.
    """

    report = {
        "records": int(len(df)),
        "columns": int(len(df.columns)),
        "nulls_duplicates": profile_nulls_and_duplicates(df),
        "numerical_statistics": profile_numerical_columns(df),
        "categorical_statistics": profile_categorical_columns(df),
        "quality_issues": identify_quality_issues(df),
    }

    os.makedirs("output", exist_ok=True)

    with open("output/profile_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Quality Issues Found: {len(report['quality_issues'])}")
    print("Report saved to output/profile_report.json")
    print("=" * 60)


if __name__ == "__main__":

    df = pd.read_csv("data/raw/quality_test.csv")

    generate_profile_report(df)