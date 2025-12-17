import argparse
import json
import os

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_ready", type=str, required=True)
    parser.add_argument("--test_ready", type=str, required=True)
    parser.add_argument("--target_col", type=str, default="house_affiliation")

    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--metrics_output", type=str, required=True)

    args = parser.parse_args()

    # Load data
    X_train_path = os.path.join(args.train_ready, "X_train.csv")
    y_train_path = os.path.join(args.train_ready, "y_train.csv")
    X_test_path = os.path.join(args.test_ready, "X_test.csv")
    y_test_path = os.path.join(args.test_ready, "y_test.csv")

    for p in [X_train_path, y_train_path, X_test_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)[args.target_col].astype(str)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)[args.target_col].astype(str)

    # Train
    clf = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Log params + metrics (this is the part you need for the Azure ML metrics screenshot)
    mlflow.log_param("max_depth", -1 if args.max_depth is None else int(args.max_depth))
    mlflow.log_param("min_samples_split", int(args.min_samples_split))
    mlflow.log_param("min_samples_leaf", int(args.min_samples_leaf))
    mlflow.log_param("random_state", int(args.random_state))

    mlflow.log_metric("accuracy", acc)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{label}_{k}", float(v))

    # Save metrics to component output folder (will be uploaded as pipeline output)
    os.makedirs(args.metrics_output, exist_ok=True)
    metrics_path = os.path.join(args.metrics_output, "classification_report.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)

    # Save MLflow model folder to model_output (contains MLmodel file)
    os.makedirs(args.model_output, exist_ok=True)
    mlflow.sklearn.save_model(sk_model=clf, path=args.model_output)

    # IMPORTANT: do NOT call mlflow.log_artifact() on AzureML here (can crash due to version mismatch)
    print("Training done. Accuracy:", acc)


if __name__ == "__main__":
    main()
