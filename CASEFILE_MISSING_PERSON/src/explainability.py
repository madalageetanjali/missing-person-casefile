"""
explainability.py -- Module 9: Explainable AI

Explains WHY a given area received a high prediction/search-priority score,
using SHAP values (if available) plus simple rule-based, human-readable
reason codes as a robust fallback:

  - High historical visit frequency
  - Similar movement pattern detected (model probability)
  - Current time matches previous visits at this place
  - Frequently used route
  - Distance consistent with historical behavior
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def rule_based_explanation(row: pd.Series, thresholds=None) -> list:
    thresholds = thresholds or {
        "visit_frequency_norm": 0.6,
        "model_probability": 0.4,
        "recency_norm": 0.6,
        "route_probability": 0.3,
        "proximity_score": 0.6,
    }
    reasons = []
    if row.get("visit_frequency_norm", 0) >= thresholds["visit_frequency_norm"]:
        reasons.append("High historical visit frequency at this location.")
    if row.get("model_probability", 0) >= thresholds["model_probability"]:
        reasons.append("Similar movement pattern detected by the location-prediction model.")
    if row.get("recency_norm", 0) >= thresholds["recency_norm"]:
        reasons.append("Current time matches previous visit patterns at this place.")
    if row.get("route_probability", 0) >= thresholds["route_probability"]:
        reasons.append("Lies on a frequently used predicted route from the last known location.")
    if row.get("proximity_score", 0) >= thresholds["proximity_score"]:
        reasons.append("Distance from last known location is consistent with historical movement range.")
    if not reasons:
        reasons.append("Included due to combined lower-weight signals; no single dominant factor.")
    return reasons


def add_explanations(ranked_df: pd.DataFrame) -> pd.DataFrame:
    df = ranked_df.copy()
    df["explanation"] = df.apply(lambda r: " ".join(rule_based_explanation(r)), axis=1)
    return df


def shap_explain_location_model(model, X_background: pd.DataFrame, X_query: pd.DataFrame):
    """
    Optional SHAP-based explanation for the supervised location model
    (works for tree-based models: RandomForest / XGBoost).
    Returns a DataFrame of feature -> mean(|SHAP value|) for the query row(s).
    """
    if not HAS_SHAP:
        return None
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_query)

    # shap_values may be a list (per-class) for multiclass RF; average abs across classes
    if isinstance(shap_values, list):
        arr = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        arr = np.abs(shap_values)

    importance = pd.DataFrame({
        "feature": X_query.columns,
        "mean_abs_shap": arr.mean(axis=0) if arr.ndim > 1 else arr
    }).sort_values("mean_abs_shap", ascending=False)
    return importance


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    ranking_path = base / "reports" / "search_priority_ranking.csv"
    if ranking_path.exists():
        df = pd.read_csv(ranking_path)
        explained = add_explanations(df)
        explained.to_csv(base / "reports" / "search_priority_ranking_explained.csv", index=False)
        print(explained[["search_rank", "cluster", "search_priority_score", "explanation"]])
    else:
        print(f"Run search_priority.py first to generate {ranking_path}")
