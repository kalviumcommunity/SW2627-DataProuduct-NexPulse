"""Reusable text cleaning utilities for the NexPulse workflow."""

from __future__ import annotations

import unicodedata

import pandas as pd


TEXT_CLEANING_RULES = {
    "customer_name": {"lowercase": True, "strip": True, "normalize_unicode": True},
    "city": {"lowercase": True, "strip": True, "remove_special": True, "normalize_unicode": True},
    "product_category": {
        "lowercase": True,
        "strip": True,
        "remove_special": True,
        "normalize_unicode": True,
        "mapping": {
            "electronics": "electronics",
            "electro nics": "electronics",
            "home garden": "home garden",
            "home and garden": "home garden",
            "home garden": "home garden",
        },
    },
    "segment": {
        "lowercase": True,
        "strip": True,
        "remove_special": True,
        "normalize_unicode": True,
        "mapping": {
            "b2b": "b2b",
            "b 2 b": "b2b",
            "business to business": "b2b",
            "businesstobusiness": "b2b",
            "retail": "retail",
        },
    },
    "status": {"lowercase": True, "strip": True, "normalize_unicode": True},
    "source": {"lowercase": True, "strip": True, "normalize_unicode": True},
}


def normalize_unicode_text(series: pd.Series) -> pd.Series:
    """Convert accented or compatibility characters into a consistent form."""
    return series.astype("string").map(
        lambda value: pd.NA
        if pd.isna(value)
        else unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def clean_text_column(series, lowercase=True, strip=True, remove_special=False, mapping=None, normalize_unicode=False):
    """Apply reusable text normalisation steps to a single column."""
    result = series.astype("string")

    if normalize_unicode:
        result = normalize_unicode_text(result)

    if strip:
        result = result.str.strip()

    if lowercase:
        result = result.str.lower()

    if remove_special:
        result = result.str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
        result = result.str.replace(r"\s+", " ", regex=True).str.strip()

    if mapping:
        mapped = result.map(mapping)
        result = mapped.fillna(result)

    return result


def normalize_text_columns(df: pd.DataFrame):
    """Clean any configured text columns that exist in the dataset."""
    cleaned = df.copy()
    report = []

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
            normalize_unicode=rule.get("normalize_unicode", False),
        )

        after_non_null = cleaned[column].dropna()
        after_unique = int(after_non_null.nunique())
        after_sample = [str(value) for value in after_non_null.head(3).tolist()]

        report.append(
            {
                "column": column,
                "before_unique": before_unique,
                "after_unique": after_unique,
                "before_sample": before_sample,
                "after_sample": after_sample,
                "rule": rule,
            }
        )

    return cleaned, report