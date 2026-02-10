import pandas as pd
import os
from pokemon_mlops.domain.feature_engineering import engineer_features


def run():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    cleaned_path = os.path.join(base_dir, "data", "cleaned.csv")
    engineered_path = os.path.join(base_dir, "data", "engineered.csv")

    df = pd.read_csv(cleaned_path)

    engineered = engineer_features(df)

    engineered.to_csv(engineered_path, index=False)

    print("Feature engineering completed.")


if __name__ == "__main__":
    run()
