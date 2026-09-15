"""
clustering.py -- Module 4: Movement Pattern Analysis

Identifies frequently visited locations and normal movement patterns
using K-Means and DBSCAN on GPS coordinates.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


def fit_kmeans(df: pd.DataFrame, n_clusters: int = 6, random_state: int = 42):
    coords = df[["latitude", "longitude"]].values
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(coords)
    return model, labels


def fit_dbscan(df: pd.DataFrame, eps_km: float = 0.3, min_samples: int = 15):
    """
    DBSCAN on lat/lon converted to radians with haversine metric, so eps is
    interpretable in real-world kilometers.
    """
    coords = np.radians(df[["latitude", "longitude"]].values)
    eps_rad = eps_km / 6371.0  # convert km to radians
    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = model.fit_predict(coords)
    return model, labels


def frequently_visited_locations(df: pd.DataFrame, cluster_col: str = "dbscan_cluster",
                                  top_n: int = 10) -> pd.DataFrame:
    """Rank clusters (excluding noise = -1) by number of pings -> frequently visited places."""
    valid = df[df[cluster_col] != -1]
    summary = (valid.groupby(cluster_col)
               .agg(visit_count=("latitude", "count"),
                    center_lat=("latitude", "mean"),
                    center_lon=("longitude", "mean"))
               .reset_index()
               .sort_values("visit_count", ascending=False)
               .head(top_n))
    return summary


def run_clustering(features_path, out_path, models_dir):
    df = pd.read_csv(features_path, parse_dates=["timestamp"])

    kmeans_model, kmeans_labels = fit_kmeans(df, n_clusters=6)
    df["kmeans_cluster"] = kmeans_labels

    dbscan_model, dbscan_labels = fit_dbscan(df, eps_km=0.3, min_samples=15)
    df["dbscan_cluster"] = dbscan_labels

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans_model, Path(models_dir) / "clustering_kmeans.pkl")
    joblib.dump(dbscan_model, Path(models_dir) / "clustering_dbscan.pkl")

    df.to_csv(out_path, index=False)

    freq_locations = frequently_visited_locations(df)
    freq_locations.to_csv(Path(out_path).parent / "frequently_visited_locations.csv", index=False)

    print(f"KMeans clusters: {df['kmeans_cluster'].nunique()}")
    print(f"DBSCAN clusters (excl. noise): {df[df['dbscan_cluster'] != -1]['dbscan_cluster'].nunique()}, "
          f"noise points: {(df['dbscan_cluster'] == -1).sum()}")
    print(freq_locations)
    return df


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    run_clustering(
        features_path=base / "data" / "processed" / "trajectory_features.csv",
        out_path=base / "data" / "processed" / "trajectory_clustered.csv",
        models_dir=base / "models",
    )
