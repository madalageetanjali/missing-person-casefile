"""
route_prediction.py -- Module 7: Route Prediction

Builds a first-order Markov Chain / transition-probability model over the
sequence of visited place-clusters (from clustering.py) to predict the
most probable NEXT locations from the last known location, and can chain
this forward to produce a probable multi-hop route:
  Location A -> Location B -> Location C -> Location D
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict


def build_transition_matrix(df: pd.DataFrame, cluster_col: str = "dbscan_cluster"):
    """
    df must be time-sorted per user. Builds P(next_cluster | current_cluster)
    using observed consecutive-visit transitions (consecutive DISTINCT
    cluster visits, so we don't just count "stayed in place").
    """
    seq = df[df[cluster_col] != -1].sort_values("timestamp")[cluster_col].tolist()

    # collapse consecutive repeats -> sequence of *visits* (place changes)
    visit_seq = []
    for c in seq:
        if not visit_seq or visit_seq[-1] != c:
            visit_seq.append(c)

    counts = defaultdict(lambda: defaultdict(int))
    for a, b in zip(visit_seq[:-1], visit_seq[1:]):
        counts[a][b] += 1

    transition = {}
    for src, dests in counts.items():
        total = sum(dests.values())
        transition[src] = {dst: cnt / total for dst, cnt in dests.items()}

    return transition, visit_seq


def predict_route(transition: dict, start_cluster: int, n_hops: int = 4):
    """
    Greedily walk the Markov chain from start_cluster, always taking the
    highest-probability next cluster, for n_hops steps. Returns the route
    and the joint probability estimate.
    """
    route = [start_cluster]
    probs = []
    current = start_cluster

    for _ in range(n_hops):
        next_probs = transition.get(current)
        if not next_probs:
            break
        next_cluster = max(next_probs, key=next_probs.get)
        probs.append(next_probs[next_cluster])
        route.append(next_cluster)
        current = next_cluster

    joint_prob = float(np.prod(probs)) if probs else 0.0
    return route, probs, joint_prob


def route_with_coordinates(route, cluster_centers: pd.DataFrame):
    rows = []
    for step, cluster in enumerate(route):
        center = cluster_centers[cluster_centers["dbscan_cluster"] == cluster]
        lat = center["center_lat"].values[0] if len(center) else None
        lon = center["center_lon"].values[0] if len(center) else None
        rows.append({"step": step, "cluster": cluster, "lat": lat, "lon": lon})
    return pd.DataFrame(rows)


def run_route_prediction(features_path, models_dir, out_dir, start_cluster=None):
    df = pd.read_csv(features_path, parse_dates=["timestamp"])
    transition, visit_seq = build_transition_matrix(df)

    if start_cluster is None:
        start_cluster = visit_seq[-1] if visit_seq else None

    if start_cluster is None or start_cluster not in transition:
        print("No sufficient transition history to predict a route from this location.")
        return None

    route, probs, joint_prob = predict_route(transition, start_cluster, n_hops=4)

    freq_path = Path(features_path).parent / "frequently_visited_locations.csv"
    if freq_path.exists():
        centers = pd.read_csv(freq_path)
        route_df = route_with_coordinates(route, centers)
    else:
        route_df = pd.DataFrame({"step": range(len(route)), "cluster": route})

    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(transition, Path(models_dir) / "route_transition_matrix.pkl")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    route_df.to_csv(Path(out_dir) / "predicted_route.csv", index=False)

    print("Predicted route (most probable path):")
    print(" -> ".join(f"Cluster {c}" for c in route))
    print(f"Joint probability estimate: {joint_prob:.4f}")
    return route_df, joint_prob


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    run_route_prediction(
        features_path=base / "data" / "processed" / "trajectory_anomalies.csv",
        models_dir=base / "models",
        out_dir=base / "data" / "processed",
    )
