import pandas as pd
import mlflow
import mlflow.sklearn

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


def evaluate_and_log(model, X_test, y_test, model_name):

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    bal_acc = balanced_accuracy_score(y_test, preds)
    pr_auc = average_precision_score(y_test, proba)

    print(f"\n--- Results for {model_name} ---")
    print(classification_report(y_test, preds))

    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Balanced Accuracy:", round(bal_acc, 4))
    print("PR-AUC:", round(pr_auc, 4))

    # Log metrics to MLflow
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_metric("balanced_accuracy", bal_acc)
    mlflow.log_metric("pr_auc", pr_auc)

    # Log model
    mlflow.sklearn.log_model(model, model_name)

    return f1, auc


def run():

    print("Starting modeling stage with MLflow...")

    mlflow.set_experiment("pokemon_mlops_experiment")

    df = pd.read_csv("data/engineered.csv")

    X = df.drop("is_legendary", axis=1)
    y = df["is_legendary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------- LOGISTIC REGRESSION --------

    with mlflow.start_run(run_name="Logistic_Regression"):

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
            cv=5
        )

        grid_lr.fit(X_train, y_train)

        best_lr = grid_lr.best_estimator_

        mlflow.log_params(grid_lr.best_params_)

        f1_lr, auc_lr = evaluate_and_log(best_lr, X_test, y_test, "logistic_regression")

        print("\nBest Logistic Regression parameters:")
        print(grid_lr.best_params_)


    # -------- RANDOM FOREST --------

    with mlflow.start_run(run_name="Random_Forest"):

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
            cv=5
        )

        grid_rf.fit(X_train, y_train)

        best_rf = grid_rf.best_estimator_

        mlflow.log_params(grid_rf.best_params_)

        f1_rf, auc_rf = evaluate_and_log(best_rf, X_test, y_test, "random_forest")

        print("\nBest Random Forest parameters:")
        print(grid_rf.best_params_)


    # -------- SELECT BEST MODEL --------

    print("\n===== MODEL COMPARISON =====")

    results = {
        "Model": ["Logistic Regression", "Random Forest"],
        "F1 Score": [f1_lr, f1_rf],
        "ROC-AUC": [auc_lr, auc_rf]
    }

    df_results = pd.DataFrame(results)
    print(df_results)

    best_model = df_results.sort_values(by=["F1 Score", "ROC-AUC"], ascending=False).iloc[0]

    print("\nBest model selected based on F1 and AUC:")
    print(best_model)


if __name__ == "__main__":
    run()
