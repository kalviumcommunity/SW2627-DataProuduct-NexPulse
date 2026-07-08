import pandas as pd


def ingest_csv(filepath, delimiter=",", encoding="utf-8", dtype_dict=None):
    """
    Load CSV file with explicit parameters.

    Args:
        filepath: Path to CSV file
        delimiter: Field delimiter
        encoding: File encoding
        dtype_dict: Optional data type mapping

    Returns:
        Pandas DataFrame
    """
    try:
        df = pd.read_csv(
            filepath,
            delimiter=delimiter,
            encoding=encoding,
            dtype=dtype_dict
        )

        print(f"\nCSV loaded: {filepath}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")

        return df

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        raise

    except UnicodeDecodeError:
        print(f"Encoding error while reading {filepath}")
        raise


def ingest_json(filepath, is_nested=False):
    """
    Load JSON data.

    Args:
        filepath: JSON file path
        is_nested: Flatten nested JSON if True

    Returns:
        Pandas DataFrame
    """
    try:
        df = pd.read_json(filepath)

        if is_nested:
            df = pd.json_normalize(df)
            print("Nested JSON flattened")

        print(f"\nJSON loaded: {filepath}")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

        return df

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        raise


def ingest_csv_with_fallback(filepath, delimiters=[","], fallback_encodings=None):
    """
    Try multiple encodings until one works.
    """

    if fallback_encodings is None:
        fallback_encodings = [
            "utf-8",
            "latin-1",
            "iso-8859-1",
            "cp1252",
        ]

    for delimiter in delimiters:
        for encoding in fallback_encodings:
            try:
                df = pd.read_csv(
                    filepath,
                    delimiter=delimiter,
                    encoding=encoding,
                )

                print(
                    f"Loaded using delimiter='{delimiter}', encoding='{encoding}'"
                )

                return df

            except Exception:
                continue

    raise ValueError("Unable to read file")


def document_ingestion(df, source_file):
    """
    Print ingestion report.
    """

    print("\n" + "=" * 60)
    print(f"INGESTION REPORT : {source_file}")
    print("=" * 60)

    print(f"Rows : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nData Types")
    print(df.dtypes)

    print("\nNull Values")
    print(df.isnull().sum())

    print("\nFirst 3 Rows")
    print(df.head(3))

    print("=" * 60)


if __name__ == "__main__":

    print("Starting multi-format ingestion...")

    csv_df = ingest_csv(
        "data/raw/customers.csv",
        delimiter=",",
        encoding="utf-8",
    )

    document_ingestion(csv_df, "customers.csv")

    json_df = ingest_json(
        "data/raw/transactions.json",
        is_nested=True,
    )

    document_ingestion(json_df, "transactions.json")

    csv_df.to_csv(
        "data/processed/customers_ingested.csv",
        index=False,
    )

    json_df.to_csv(
        "data/processed/transactions_ingested.csv",
        index=False,
    )

    print("\nAll data ingested successfully.")