"""
anomaly_detection.py -- Module 5: Anomaly Detection

Flags movements that differ significantly from the subject's normal
behavior using Isolation Forest, Local Outlier Factor, and One-Class SVM.

IMPORTANT (ethical requirement): an anomaly flag does NOT indicate
suspicious or criminal behavior. It only means the movement statistically
differs from the person's historical pattern (e.g. unusual hour, unusual
speed, unusual location). This must be stated wherever anomalies are shown.
"""

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["latitude", "longitude", "hour", "speed_kmh", "distance_km"]


def _prep(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, scaler


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.03):
    Xs, scaler = _prep(df)
    model = IsolationForest(contamination=contamination, random_state=42)
    pred = model.fit_predict(Xs)          # -1 = anomaly, 1 = normal
    score = model.decision_function(Xs)   # higher = more normal
    return model, scaler, pred, score


def run_lof(df: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.03):
    Xs, _ = _prep(df)
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    pred = model.fit_predict(Xs)
    score = model.negative_outlier_factor_
    return model, pred, score


def run_one_class_svm(df: pd.DataFrame, nu: float = 0.03):
    Xs, _ = _prep(df)
    model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    pred = model.fit_predict(Xs)
    score = model.decision_function(Xs)
    return model, pred, score


def run_anomaly_detection(features_path, out_path, models_dir):
    df = pd.read_csv(features_path, parse_dates=["timestamp"])

    if_model, scaler, if_pred, if_score = run_isolation_forest(df)
    df["iforest_anomaly"] = (if_pred == -1).astype(int)
    df["iforest_score"] = if_score

    lof_model, lof_pred, lof_score = run_lof(df)
    df["lof_anomaly"] = (lof_pred == -1).astype(int)
    df["lof_score"] = lof_score

    svm_model, svm_pred, svm_score = run_one_class_svm(df)
    df["ocsvm_anomaly"] = (svm_pred == -1).astype(int)
    df["ocsvm_score"] = svm_score

    # Consensus flag: anomalous if at least 2 of 3 methods agree
    df["anomaly_consensus"] = (
        df[["iforest_anomaly", "lof_anomaly", "ocsvm_anomaly"]].sum(axis=1) >= 2
    ).astype(int)

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": if_model, "scaler": scaler}, Path(models_dir) / "anomaly_model.pkl")

    df.to_csv(out_path, index=False)

    n_anom = df["anomaly_consensus"].sum()
    print(f"Flagged {n_anom} / {len(df)} points as anomalous (consensus of >=2 methods).")
    print("Reminder: anomaly != suspicious/criminal behavior. Document false positives.")
    return df


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    run_anomaly_detection(
        features_path=base / "data" / "processed" / "trajectory_clustered.csv",
        out_path=base / "data" / "processed" / "trajectory_anomalies.csv",
        models_dir=base / "models",
    )
