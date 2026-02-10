import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completely safe feature engineering for Pokemon dataset
    """

    print("Starting feature engineering...")

    # Safe numeric columns
    numeric_columns = [
        "hp", "attack", "defense",
        "sp_attack", "sp_defense", "speed",
        "base_total", "height_m", "weight_kg",
        "generation", "against_bug", "against_dark",
        "against_dragon", "against_electric",
        "against_fairy", "against_fight",
        "against_fire", "against_flying",
        "against_ghost", "against_grass",
        "against_ground", "against_ice",
        "against_normal", "against_poison",
        "against_psychic", "against_rock",
        "against_steel", "against_water",
        "is_legendary"
    ]

    available = [c for c in numeric_columns if c in df.columns]

    print("Using only these columns:")
    print(available)

    clean_df = df[available].copy()

    # Separate target
    X = clean_df.drop("is_legendary", axis=1)
    y = clean_df["is_legendary"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_df = pd.DataFrame(X_scaled, columns=X.columns)
    final_df["is_legendary"] = y.values

    print(f"Final dataset shape: {final_df.shape}")

    return final_df
