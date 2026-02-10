import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def run():

    print("Starting modeling stage...")

    df = pd.read_csv("data/engineered.csv")

    X = df.drop("is_legendary", axis=1)
    y = df["is_legendary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # With very few legendary samples, we must reduce CV folds
    cv_folds = 2

    # -------- LOGISTIC REGRESSION GRIDSEARCH --------

    print("Optimizing Logistic Regression...")

    param_grid_lr = {
        "C": [0.01, 0.1, 1, 10],
        "class_weight": ["balanced", None],
        "solver": ["lbfgs"],
        "max_iter": [1000]
    }

    grid_lr = GridSearchCV(
        LogisticRegression(),
        param_grid_lr,
        scoring="f1",
        cv=cv_folds
    )

    grid_lr.fit(X_train, y_train)

    best_lr = grid_lr.best_estimator_

    # -------- RANDOM FOREST GRIDSEARCH --------

    print("Optimizing Random Forest...")

    param_grid_rf = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"]
    }

    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        scoring="f1",
        cv=cv_folds
    )

    grid_rf.fit(X_train, y_train)

    best_rf = grid_rf.best_estimator_

    # -------- EVALUATION --------

    print("\n===== FINAL MODEL EVALUATION =====\n")

    for name, model in [
        ("Logistic Regression", best_lr),
        ("Random Forest", best_rf)
    ]:

        print(f"\n--- Results for {name} ---")

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        print(classification_report(y_test, preds, zero_division=0))

        print("F1 Score:", round(f1_score(y_test, preds, zero_division=0), 4))
        print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))
        print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, preds), 4))
        print("PR-AUC:", round(average_precision_score(y_test, proba), 4))

    print("\nBest Logistic Regression parameters:")
    print(grid_lr.best_params_)

    print("\nBest Random Forest parameters:")
    print(grid_rf.best_params_)


if __name__ == "__main__":
    run()
