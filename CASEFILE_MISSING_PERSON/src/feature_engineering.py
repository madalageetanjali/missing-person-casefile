"""
feature_engineering.py -- Module 3: Feature Engineering

Derives movement-related features: distance between consecutive points,
speed, total/average/max speed & distance, visit counts, and
movement frequency by hour/day.
"""

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points (vectorized)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_movement_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"]).copy()

    df["prev_lat"] = df.groupby("user_id")["latitude"].shift(1)
    df["prev_lon"] = df.groupby("user_id")["longitude"].shift(1)
    df["prev_ts"] = df.groupby("user_id")["timestamp"].shift(1)

    df["distance_km"] = haversine_km(df["prev_lat"], df["prev_lon"],
                                      df["latitude"], df["longitude"])
    df["time_delta_hr"] = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 3600.0
    df["speed_kmh"] = np.where(df["time_delta_hr"] > 0,
                                df["distance_km"] / df["time_delta_hr"], 0.0)

    df[["distance_km", "time_delta_hr", "speed_kmh"]] = \
        df[["distance_km", "time_delta_hr", "speed_kmh"]].fillna(0)

    df = df.drop(columns=["prev_lat", "prev_lon", "prev_ts"])
    return df


def user_summary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-user aggregate features: total distance, average/max speed,
    number of distinct rounded locations visited, and movement
    frequency by hour/day-of-week.
    """
    agg = df.groupby("user_id").agg(
        total_distance_km=("distance_km", "sum"),
        avg_speed_kmh=("speed_kmh", "mean"),
        max_speed_kmh=("speed_kmh", "max"),
        num_points=("distance_km", "count"),
    ).reset_index()

    # visit frequency at rounded grid cells (proxy for "distinct locations")
    df["_grid_lat"] = df["latitude"].round(3)
    df["_grid_lon"] = df["longitude"].round(3)
    visits = (df.groupby(["user_id", "_grid_lat", "_grid_lon"])
                .size().reset_index(name="visit_count"))
    n_locations = visits.groupby("user_id")["visit_count"].count() \
                          .reset_index(name="num_distinct_locations")

    agg = agg.merge(n_locations, on="user_id", how="left")
    return agg


def hourly_daily_frequency(df: pd.DataFrame) -> pd.DataFrame:
    freq = df.groupby(["user_id", "weekday", "hour"]).size().reset_index(name="ping_count")
    return freq


if __name__ == "__main__":
    from pathlib import Path
    base = Path(__file__).resolve().parents[1]
    df = pd.read_csv(base / "data" / "processed" / "trajectory_clean.csv", parse_dates=["timestamp"])
    df = add_movement_features(df)
    df.to_csv(base / "data" / "processed" / "trajectory_features.csv", index=False)

    summary = user_summary_features(df)
    summary.to_csv(base / "data" / "processed" / "user_summary_features.csv", index=False)

    freq = hourly_daily_frequency(df)
    freq.to_csv(base / "data" / "processed" / "hourly_daily_frequency.csv", index=False)

    print("Feature engineering complete.")
    print(summary)
