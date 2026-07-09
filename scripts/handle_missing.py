import json
import os

import pandas as pd


def analyze_missing_before(df):
    """
    Analyze missing values before treatment.
    """
    print("\n===== BEFORE IMPUTATION =====")

    report = {}

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / len(df)) * 100, 2)

        report[col] = {
            "null_count": null_count,
            "null_percentage": null_pct
        }

        print(f"{col}: {null_count} null(s) ({null_pct}%)")

    return report


def apply_imputation(df):
    """
    Apply different missing value strategies.
    """

    decisions = []

    # Drop rows with missing customer_id
    before = len(df)
    df = df.dropna(subset=["customer_id"])
    dropped = before - len(df)

    decisions.append({
        "column": "customer_id",
        "strategy": "Drop Rows",
        "reason": "Primary key cannot be missing",
        "affected_rows": int(dropped)
    })

    # Numerical columns → Median
    numerical_cols = ["amount", "quantity"]

    for col in numerical_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

        decisions.append({
            "column": col,
            "strategy": "Median",
            "value_used": float(median),
            "reason": "Median is robust to outliers"
        })

    # Categorical columns → Mode
    categorical_cols = ["name", "email", "category", "region"]

    for col in categorical_cols:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)

        decisions.append({
            "column": col,
            "strategy": "Mode",
            "value_used": str(mode),
            "reason": "Most frequent value preserves distribution"
        })

    # Forward fill dates
    df["last_updated"] = df["last_updated"].ffill()

    decisions.append({
        "column": "last_updated",
        "strategy": "Forward Fill",
        "reason": "Maintains chronological consistency"
    })

    return df, decisions


def validate_after(df):
    """
    Validate remaining missing values.
    """
    print("\n===== AFTER IMPUTATION =====")

    for col in df.columns:
        print(f"{col}: {int(df[col].isnull().sum())} null(s)")


def save_outputs(df, decisions):
    """
    Save cleaned dataset and imputation log.
    """

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    df.to_csv(
        "data/processed/cleaned_data.csv",
        index=False
    )

    with open(
        "output/imputation_decisions.json",
        "w"
    ) as f:
        json.dump(decisions, f, indent=4)

    print("\n✓ Cleaned dataset saved.")
    print("✓ Imputation decisions saved.")


if __name__ == "__main__":

    df = pd.read_csv("data/raw/missing_data.csv")

    analyze_missing_before(df)

    cleaned_df, decisions = apply_imputation(df)

    validate_after(cleaned_df)

    save_outputs(cleaned_df, decisions)

    print("\n===== MISSING VALUE HANDLING COMPLETED =====")