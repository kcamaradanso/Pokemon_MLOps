import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Preparation stage:
    Clean data without removing rows (to preserve legendary Pokémon)
    """

    print("Starting data preparation...")

    df.columns = df.columns.str.lower()

    # Impute numeric columns with median instead of dropping rows
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    print(f"Rows after cleaning: {len(df)}")
    print(f"Legendary after cleaning: {df['is_legendary'].sum()}")

    return df
