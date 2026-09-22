"""Retrain the seizure detector with class-balanced models and a validation-learned threshold.

Run locally from the project root:
    python retrain_balanced.py

The script uses the same 11,500-window EEG dataset as the notebook. It keeps a
stratified train/validation/test split, uses class_weight='balanced', selects a
model and alert threshold using validation F1, and evaluates once on the held-out test set.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA_DIR = ROOT / "data"
LOCAL = DATA_DIR / "epileptic_seizure_data.csv"
DATA_URL = "https://raw.githubusercontent.com/Jreevo/Epileptic-Seizure-Binary-Classification/master/epilepsy.csv"


def load_data():
    path = LOCAL if LOCAL.exists() else DATA_URL
    df = pd.read_csv(path)
    features = [f"X{i}" for i in range(1, 179)]
    missing = [c for c in features + ["y"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:10]}")
    X = df[features].astype(float).values
    y = (pd.to_numeric(df["y"], errors="coerce") == 1).astype(int).values
    return X, y


def best_threshold(y_true, proba):
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
        })
    return max(rows, key=lambda r: (r["f1"], r["recall"])) , pd.DataFrame(rows)


def evaluate(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }, pred


def main():
    OUT.mkdir(exist_ok=True)
    X, y = load_data()
    print(f"Dataset: {X.shape[0]} windows × {X.shape[1]} EEG samples")
    print(f"Seizure prevalence: {y.mean():.1%}")

    # 70/15/15 stratified split. Test is untouched until final evaluation.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    models = {
        "LogisticRegression_balanced": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=42
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=42,
            n_jobs=-1, min_samples_leaf=1
        ),
        "SVM_RBF_balanced": SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=42
        ),
    }

    validation_rows = []
    fitted = {}
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_s, y_train)
        val_proba = clf.predict_proba(X_val_s)[:, 1]
        tuned, threshold_table = best_threshold(y_val, val_proba)
        metrics, val_pred = evaluate(y_val, val_proba, tuned["threshold"])
        validation_rows.append({"model": name, **metrics, "threshold": tuned["threshold"]})
        fitted[name] = (clf, val_proba, threshold_table)
        print(
            f"Validation: F1={metrics['f1']:.4f} recall={metrics['recall']:.4f} "
            f"precision={metrics['precision']:.4f} threshold={tuned['threshold']:.3f}"
        )

    validation_df = pd.DataFrame(validation_rows).sort_values("f1", ascending=False)
    best_name = validation_df.iloc[0]["model"]
    selected_model, _, threshold_table = fitted[best_name]
    threshold = float(validation_df.iloc[0]["threshold"])

    test_proba = selected_model.predict_proba(X_test_s)[:, 1]
    test_metrics, test_pred = evaluate(y_test, test_proba, threshold)
    fpr, tpr, _ = roc_curve(y_test, test_proba)

    meta = {
        "best_name": best_name,
        "threshold_method": "validation F1 maximization with class-balanced training",
        "alert_threshold": threshold,
        "results_df": pd.DataFrame([test_metrics | {"model": best_name}]),
        "validation_results_df": validation_df,
        "confusion_matrix": confusion_matrix(y_test, test_pred),
        "y_test": np.asarray(y_test),
        "best_proba": np.asarray(test_proba),
        "fpr": np.asarray(fpr),
        "tpr": np.asarray(tpr),
        "roc_auc": float(test_metrics["roc_auc"]),
        "classification_report": classification_report(
            y_test, test_pred, target_names=["Non-Seizure", "Seizure"], output_dict=True
        ),
        "class_distribution": {
            "non_seizure": int((y == 0).sum()),
            "seizure": int((y == 1).sum()),
        },
        "split_sizes": {
            "train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))
        },
    }
    if hasattr(selected_model, "feature_importances_"):
        meta["feature_importances"] = selected_model.feature_importances_

    bundle = {
        "model": selected_model,
        "scaler": scaler,
        "threshold": threshold,
        "model_name": best_name,
    }
    joblib.dump(selected_model, OUT / "best_model.pkl")
    joblib.dump(scaler, OUT / "scaler.pkl")
    joblib.dump(meta, OUT / "metadata.joblib")
    joblib.dump(bundle, OUT / "best_model_bundle.joblib")

    print("\nFINAL TEST RESULTS")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Saved deployment artifacts to {OUT}")


if __name__ == "__main__":
    main()
