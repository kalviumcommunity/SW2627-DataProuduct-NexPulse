import pandas as pd


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
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Fill missing numeric values with 0
    df = df.fillna(0)

    return df


def output_results(df, output_path):
    """
    Save processed data.

    Input:
        df (DataFrame): Processed dataset.
        output_path (str): Output CSV path.
    """
    # Save processed data
    df.to_csv(output_path, index=False)

print("Data successfully processed")
print(f"Rows processed: {len(df)}")
print(f"Output saved to {output_path}")


if __name__ == "__main__":
    data = ingest_data("data/raw/sample.csv")
    processed = process_data(data)
    output_results(processed, "output/processed.csv")