import pandas as pd
import os
from pokemon_mlops.domain.transform import clean_data


def run():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    raw_path = os.path.join(base_dir, "data", "raw.csv")
    cleaned_path = os.path.join(base_dir, "data", "cleaned.csv")

    df = pd.read_csv(raw_path)

    cleaned = clean_data(df)

    cleaned.to_csv(cleaned_path, index=False)

    print("Data preparation completed.")


if __name__ == "__main__":
    run()
