"""
prediction.py -- Module 6: Location Prediction

Trains a supervised model (Random Forest / XGBoost / KNN) to predict which
"place cluster" (from clustering.py) a person is likely to be in, given
contextual features (hour, weekday, last known coords, speed, etc.).

The output is a ranked probability distribution over historically visited
place-clusters -- i.e. "probable geographical areas," not a single point.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, top_k_accuracy_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEATURE_COLS = ["hour", "weekday", "is_weekend", "speed_kmh", "distance_km", "month"]
TARGET_COL = "dbscan_cluster"


def prepare_training_data(df: pd.DataFrame):
    data = df[df[TARGET_COL] != -1].copy()  # exclude DBSCAN noise as a target
    X = data[FEATURE_COLS].fillna(0)
    y = data[TARGET_COL].astype(int)
    return X, y


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    results = {}

    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate(rf, X_test, y_test, X_train.shape[1])

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    results["knn"] = evaluate(knn, X_test, y_test, X_train.shape[1])

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="mlogloss", use_label_encoder=False
        )
        # XGBoost needs contiguous 0..n-1 labels
        label_map = {lab: i for i, lab in enumerate(sorted(y.unique()))}
        inv_map = {i: lab for lab, i in label_map.items()}
        y_train_x = y_train.map(label_map)
        y_test_x = y_test.map(label_map)
        xgb.fit(X_train, y_train_x)
        results["xgboost"] = evaluate(xgb, X_test, y_test_x, X_train.shape[1], label_map=inv_map)

    return results, (X_train, X_test, y_train, y_test)


def evaluate(model, X_test, y_test, n_features, label_map=None):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    proba = model.predict_proba(X_test)
    labels_present = sorted(y_test.unique())
    for k in (1, 3, 5):
        k_eff = min(k, proba.shape[1])
        try:
            metrics[f"top_{k}_accuracy"] = top_k_accuracy_score(
                y_test, proba, k=k_eff, labels=model.classes_
            )
        except Exception:
            metrics[f"top_{k}_accuracy"] = None

    metrics["model"] = model
    metrics["label_map"] = label_map
    return metrics


def predict_probable_locations(model, context_features: dict, cluster_centers: pd.DataFrame,
                                top_n: int = 5, label_map=None):
    """
    Given a trained model and a context (hour, weekday, etc. for the last
    known observation), return a ranked list of probable place-clusters
    with probability, and their approximate lat/lon centers.
    """
    X = pd.DataFrame([context_features])[FEATURE_COLS]
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    rows = []
    for cls, p in zip(classes, proba):
        real_cluster = label_map[cls] if label_map else cls
        center = cluster_centers[cluster_centers["dbscan_cluster"] == real_cluster]
        lat = center["center_lat"].values[0] if len(center) else None
        lon = center["center_lon"].values[0] if len(center) else None
        rows.append({"cluster": real_cluster, "probability": p, "center_lat": lat, "center_lon": lon})

    ranked = pd.DataFrame(rows).sort_values("probability", ascending=False).head(top_n)
    ranked = ranked.dropna(subset=["center_lat", "center_lon"])
    return ranked.reset_index(drop=True)


def run_prediction(features_path, models_dir, out_dir):
    df = pd.read_csv(features_path, parse_dates=["timestamp"])
    X, y = prepare_training_data(df)

    if y.nunique() < 2:
        print("Not enough distinct location clusters to train a classifier. "
              "Try a longer synthetic history or lower DBSCAN min_samples.")
        return None

    results, splits = train_models(X, y)

    print("\n=== Model comparison ===")
    for name, m in results.items():
        print(f"{name}: acc={m['accuracy']:.3f} f1={m['f1']:.3f} "
              f"top1={m.get('top_1_accuracy')} top3={m.get('top_3_accuracy')}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]
    print(f"\nSelected best model: {best_name} (highest weighted F1)")

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best["model"], "label_map": best.get("label_map"),
                 "features": FEATURE_COLS, "model_name": best_name},
                Path(models_dir) / "location_model.pkl")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    summary_rows = [{"model": n, "accuracy": m["accuracy"], "precision": m["precision"],
                      "recall": m["recall"], "f1": m["f1"],
                      "top_1_accuracy": m.get("top_1_accuracy"),
                      "top_3_accuracy": m.get("top_3_accuracy"),
                      "top_5_accuracy": m.get("top_5_accuracy")}
                     for n, m in results.items()]
    pd.DataFrame(summary_rows).to_csv(Path(out_dir) / "model_comparison.csv", index=False)

    return results, best_name


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    run_prediction(
        features_path=base / "data" / "processed" / "trajectory_anomalies.csv",
        models_dir=base / "models",
        out_dir=base / "reports",
    )
