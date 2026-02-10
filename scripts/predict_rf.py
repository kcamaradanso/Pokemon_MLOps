import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


def main():
    df = pd.read_csv("data/engineered.csv")

    X = df.drop("is_legendary", axis=1)
    y = df["is_legendary"]

    # Split to keep training stable (we will predict on full X later)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid_rf = {
        "n_estimators": [100],
        "max_depth": [None],
        "min_samples_split": [2],
        "class_weight": ["balanced"]
    }

    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        scoring="f1",
        cv=2,
        n_jobs=-1
    )

    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_

    # Predict on the full dataset
    preds = best_rf.predict(X)
    probs = best_rf.predict_proba(X)[:, 1]

    out = df.copy()
    out["predicted_is_legendary"] = preds
    out["predicted_proba"] = probs

    out_path = "data/predictions_rf.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
