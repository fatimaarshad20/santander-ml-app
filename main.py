"""
Santander Customer Transaction Prediction
Run from VS Code with: python main.py

Place the Kaggle training file at data/train.csv.
The script trains XGBoost, Random Forest, and Logistic Regression,
prints evaluation metrics, saves plots/CSV outputs, and exports the XGBoost model.
"""
from pathlib import Path
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Save plots without requiring an interactive window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }
    return metrics, predictions, probabilities


def save_confusion_matrix(name, y_test, predictions):
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"]
    )
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()


def main():
    print("=" * 64)
    print("SANTANDER CUSTOMER TRANSACTION PREDICTION")
    print("=" * 64)

    if not DATA_PATH.exists():
        print("\nERROR: Dataset not found.")
        print(f"Expected file: {DATA_PATH}")
        print("\nDownload train.csv from the Kaggle Santander Customer Transaction")
        print("Prediction dataset and place it inside the project's data folder.")
        print("The expected CSV must contain columns 'ID_code' and 'target'.")
        return

    print(f"\nLoading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "target" not in df.columns:
        raise ValueError("The CSV must contain a 'target' column.")
    if "ID_code" not in df.columns:
        print("Warning: 'ID_code' column was not found; proceeding without dropping it.")

    print(f"Dataset loaded successfully: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nTarget distribution:")
    print(df["target"].value_counts().sort_index().to_string())
    print(f"\nMissing values before preprocessing: {int(df.isnull().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")

    # Separate features and target; ID_code is an identifier, not a predictive feature.
    drop_columns = ["target"]
    if "ID_code" in df.columns:
        drop_columns.append("ID_code")
    X = df.drop(columns=drop_columns).copy()
    y = df["target"].astype(int)

    # Fill missing numeric values using each column's median.
    numeric_columns = X.select_dtypes(include="number").columns
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    # If any non-numeric columns exist, fill them with their most frequent value.
    for column in X.columns:
        if X[column].isnull().any():
            mode_values = X[column].mode()
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            X[column] = X[column].fillna(fill_value)

    print(f"Missing values after preprocessing: {int(X.isnull().sum().sum())}")
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train_model, X_val, y_train_model, y_val = train_test_split(
        X_train, y_train, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"\nModel-training rows: {len(X_train_model):,}")
    print(f"Validation rows:     {len(X_val):,}")
    print(f"Final test rows:     {len(X_test):,}")

    models = {}
    training_times = {}

    print("\n[1/3] Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=30
    )
    start = time.time()
    xgb_model.fit(X_train_model, y_train_model, eval_set=[(X_val, y_val)], verbose=False)
    training_times["XGBoost"] = time.time() - start
    models["XGBoost"] = xgb_model
    print(f"XGBoost training complete in {training_times['XGBoost']:.2f} seconds.")
    if hasattr(xgb_model, "best_iteration"):
        print(f"Best iteration: {xgb_model.best_iteration}")

    print("\n[2/3] Training Random Forest (class-balanced)...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    start = time.time()
    rf_model.fit(X_train_model, y_train_model)
    training_times["Random Forest"] = time.time() - start
    models["Random Forest"] = rf_model
    print(f"Random Forest training complete in {training_times['Random Forest']:.2f} seconds.")

    print("\n[3/3] Training Logistic Regression...")
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ))
    ])
    start = time.time()
    logistic_model.fit(X_train_model, y_train_model)
    training_times["Logistic Regression"] = time.time() - start
    models["Logistic Regression"] = logistic_model
    print(f"Logistic Regression training complete in {training_times['Logistic Regression']:.2f} seconds.")

    results = []
    predictions_by_model = {}
    probabilities_by_model = {}
    print("\n" + "=" * 64)
    print("TEST SET PERFORMANCE")
    print("=" * 64)
    for name, model in models.items():
        metrics, predictions, probabilities = evaluate_model(name, model, X_test, y_test)
        metrics["Training Time (s)"] = training_times[name]
        results.append(metrics)
        predictions_by_model[name] = predictions
        probabilities_by_model[name] = probabilities
        print(f"\n{name}")
        for key in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
            print(f"{key:10s}: {metrics[key]:.4f}")
        print(f"{'Train time':10s}: {training_times[name]:.2f} seconds")
        save_confusion_matrix(name, y_test, predictions)
        report = classification_report(
            y_test, predictions, target_names=["Class 0", "Class 1"], zero_division=0
        )
        (OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_classification_report.txt").write_text(
            report, encoding="utf-8"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print("\nFINAL MODEL COMPARISON")
    print(results_df.round(4).to_string(index=False))

    # ROC curve comparison
    plt.figure(figsize=(8, 6))
    for name, probabilities in probabilities_by_model.items():
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_value = roc_auc_score(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve_comparison.png", dpi=150)
    plt.close()

    # Train/test accuracy gap as a simple overfitting check.
    print("\nTRAINING VS TEST ACCURACY")
    for name, model in models.items():
        train_accuracy = accuracy_score(y_train_model, model.predict(X_train_model))
        test_accuracy = accuracy_score(y_test, predictions_by_model[name])
        print(
            f"{name}: train={train_accuracy:.4f}, test={test_accuracy:.4f}, "
            f"gap={train_accuracy - test_accuracy:.4f}"
        )

    # Export the XGBoost model in the same package style as the notebook.
    model_package = {
        "model": xgb_model,
        "features": list(X.columns),
        "model_name": "XGBoost",
        "dataset": "Santander Customer Transaction Prediction"
    }
    model_path = OUTPUT_DIR / "santander_xgboost_model.pkl"
    joblib.dump(model_package, model_path)
    print(f"\nSaved XGBoost model: {model_path}")

    # Load it back and demonstrate predictions on ten test rows.
    loaded_package = joblib.load(model_path)
    loaded_model = loaded_package["model"]
    loaded_predictions = loaded_model.predict(X_test.head(10))
    print("Model reload test passed.")
    print("Predictions for first 10 test rows:", loaded_predictions.tolist())

    print(f"\nAll output files are saved in: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
