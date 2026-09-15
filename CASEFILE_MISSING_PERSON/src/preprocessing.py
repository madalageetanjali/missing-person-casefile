"""
preprocessing.py -- Module 2: Data Preprocessing

Handles missing values, duplicates, invalid GPS coordinates, timestamp
conversion, and creation of basic time-based features.
"""

import pandas as pd
import numpy as np


def load_raw_trajectory(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def clean_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Missing-value handling
    df = df.dropna(subset=["user_id", "latitude", "longitude", "timestamp"])

    # 2. Duplicate removal
    df = df.drop_duplicates(subset=["user_id", "latitude", "longitude", "timestamp"])

    # 3. Invalid GPS-coordinate removal
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]

    # 4. Timestamp conversion (ensure datetime dtype)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 5. Sort chronologically per user
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekday"] = df["timestamp"].dt.weekday          # 0=Mon ... 6=Sun
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df


def remove_outlier_speed(df: pd.DataFrame, max_speed_kmh: float = 200.0) -> pd.DataFrame:
    """
    Outlier analysis: drop points implying implausible speed between
    consecutive pings for the same user (GPS glitches / teleport errors).
    Requires 'speed_kmh' to already be computed (see feature_engineering.py);
    if not present, this is a no-op.
    """
    if "speed_kmh" not in df.columns:
        return df
    return df[df["speed_kmh"].fillna(0) <= max_speed_kmh].reset_index(drop=True)


def run_preprocessing(raw_path, processed_path):
    df = load_raw_trajectory(raw_path)
    df = clean_trajectory(df)
    df = add_time_features(df)
    df.to_csv(processed_path, index=False)
    print(f"Preprocessed {len(df)} rows -> {processed_path}")
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path
    base = Path(__file__).resolve().parents[1]
    raw = base / "data" / "synthetic" / "synthetic_gps_trajectory.csv"
    out = base / "data" / "processed" / "trajectory_clean.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    run_preprocessing(raw, out)
