"""
search_priority.py -- Module 8: Search Priority Score

Combines multiple signals into a single ranked "search priority score" per
candidate location/cluster, so investigators know where to look first.

Default weighting (student may modify, but must justify the change in the
report -- see reports/investigation_report_template.md):

    score = 0.35 * visit_frequency_norm
          + 0.30 * model_probability
          + 0.15 * recency_norm
          + 0.10 * route_probability
          + 0.10 * (1 - distance_from_last_known_norm)

Rationale for the default weights:
- visit_frequency (0.35): historically, people are most often found at
  places they visit often (home, work, regular haunts).
- model_probability (0.30): the supervised location-prediction model
  directly encodes contextual similarity to past behavior at this time.
- recency (0.15): recently-visited places are weighted slightly higher
  than places not visited in a long time.
- route_probability (0.10): places reachable via the predicted route from
  the last known location get a boost.
- proximity (0.10): all else equal, closer candidate locations are
  slightly preferred (easier/faster for search teams to check first),
  but this is deliberately the smallest weight since missing persons can
  travel far.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from feature_engineering import haversine_km


DEFAULT_WEIGHTS = {
    "visit_frequency": 0.35,
    "model_probability": 0.30,
    "recency": 0.15,
    "route_probability": 0.10,
    "proximity": 0.10,
}


def _minmax_norm(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def compute_search_priority(freq_locations: pd.DataFrame,
                             model_predictions: pd.DataFrame,
                             route_probs: dict,
                             last_known_lat: float,
                             last_known_lon: float,
                             recency_by_cluster: pd.Series = None,
                             weights: dict = None) -> pd.DataFrame:
    """
    freq_locations: columns [dbscan_cluster, visit_count, center_lat, center_lon]
    model_predictions: columns [cluster, probability, center_lat, center_lon]
    route_probs: {cluster_id: probability_of_being_on_predicted_route}
    recency_by_cluster: Series indexed by cluster -> days since last visit (lower=more recent)
    """
    weights = weights or DEFAULT_WEIGHTS

    df = freq_locations.rename(columns={"dbscan_cluster": "cluster"}).copy()
    df["visit_frequency_norm"] = _minmax_norm(df["visit_count"])

    model_map = model_predictions.set_index("cluster")["probability"].to_dict() \
        if model_predictions is not None and len(model_predictions) else {}
    df["model_probability"] = df["cluster"].map(model_map).fillna(0.0)

    df["route_probability"] = df["cluster"].map(route_probs or {}).fillna(0.0)

    if recency_by_cluster is not None:
        rec = recency_by_cluster.reindex(df["cluster"]).fillna(recency_by_cluster.max())
        # invert + normalize so "more recent" = higher score
        df["recency_norm"] = 1 - _minmax_norm(pd.Series(rec.values, index=df.index))
    else:
        df["recency_norm"] = 0.5  # neutral if unknown

    df["distance_km_from_last_known"] = haversine_km(
        last_known_lat, last_known_lon, df["center_lat"], df["center_lon"]
    )
    df["proximity_score"] = 1 - _minmax_norm(df["distance_km_from_last_known"])

    df["search_priority_score"] = (
        weights["visit_frequency"] * df["visit_frequency_norm"]
        + weights["model_probability"] * df["model_probability"]
        + weights["recency"] * df["recency_norm"]
        + weights["route_probability"] * df["route_probability"]
        + weights["proximity"] * df["proximity_score"]
    )

    df = df.sort_values("search_priority_score", ascending=False).reset_index(drop=True)
    df["search_rank"] = df.index + 1
    return df


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    freq = pd.read_csv(base / "data" / "processed" / "frequently_visited_locations.csv")
    case = pd.read_csv(base / "data" / "synthetic" / "synthetic_case_file.csv")

    last_lat = case["last_known_latitude"].iloc[0]
    last_lon = case["last_known_longitude"].iloc[0]

    result = compute_search_priority(
        freq_locations=freq,
        model_predictions=pd.DataFrame(columns=["cluster", "probability"]),
        route_probs={},
        last_known_lat=last_lat,
        last_known_lon=last_lon,
    )
    out = base / "reports" / "search_priority_ranking.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(result[["search_rank", "cluster", "visit_count", "search_priority_score"]])
