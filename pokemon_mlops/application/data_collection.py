import os
from pokemon_mlops.infrastructure.extract import extract_data


def run():
    # Move up TWO levels to reach the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    output_path = os.path.join(base_dir, "data", "raw.csv")

    df = extract_data()

    df.to_csv(output_path, index=False)

    print("Data collection completed.")


if __name__ == "__main__":
    run()
