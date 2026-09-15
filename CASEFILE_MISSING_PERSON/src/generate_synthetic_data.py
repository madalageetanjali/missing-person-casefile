"""
generate_synthetic_data.py
---------------------------------
Generates a synthetic GPS-trajectory dataset (schema-compatible with the
Microsoft GeoLife GPS Trajectory Dataset) plus a fictional missing-person
case file, for the CASEFILE academic ML project.

This script is a STAND-IN for real data. If you download the real GeoLife
dataset (see README.md), place the .plt files under data/raw/geolife/ and
use src/load_geolife.py instead -- the rest of the pipeline (preprocessing,
clustering, anomaly detection, prediction, routing) works unchanged because
it only depends on the common schema:

    user_id, latitude, longitude, timestamp

All identities, coordinates and case details below are FICTIONAL and were
generated for educational purposes only. This is not a tool for locating
real missing persons.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

# Fictional "anchor" locations for one synthetic user (home, work, gym, market, park)
ANCHORS = {
    "home":    (28.6139, 77.2090),   # fictional coordinates, Delhi-ish
    "work":    (28.6304, 77.2177),
    "gym":     (28.6100, 77.2300),
    "market":  (28.6250, 77.1950),
    "park":    (28.6000, 77.2050),
    "friend":  (28.6400, 77.2250),
}

# Typical weekly routine: which anchor the user tends to be at, by hour bucket
# (weekday routine differs from weekend routine)
WEEKDAY_ROUTINE = {
    range(0, 7):   "home",
    range(7, 9):   "commute_to_work",
    range(9, 18):  "work",
    range(18, 20): "commute_to_home",
    range(20, 22): "gym",
    range(22, 24): "home",
}
WEEKEND_ROUTINE = {
    range(0, 9):   "home",
    range(9, 12):  "market",
    range(12, 16): "park",
    range(16, 20): "friend",
    range(20, 24): "home",
}


def _hour_bucket_location(hour, is_weekend, shift=0):
    """
    shift: a per-day random offset (in hours, can be fractional-ish via
    rounding) applied to routine boundaries, so the routine isn't identical
    clockwork every single day -- mimics a real person's variable schedule.
    """
    routine = WEEKEND_ROUTINE if is_weekend else WEEKDAY_ROUTINE
    shifted_hour = hour - shift
    for hr_range, label in routine.items():
        if shifted_hour in hr_range:
            return label
    return "home"


def _jitter(lat, lon, scale=0.0025):
    """Add small random GPS noise around an anchor point."""
    return lat + RNG.normal(0, scale), lon + RNG.normal(0, scale)


def _interpolate(p1, p2, frac):
    return (p1[0] + (p2[0] - p1[0]) * frac, p1[1] + (p2[1] - p1[1]) * frac)


def generate_user_trajectory(user_id="U001", start_date="2025-01-01", days=60,
                              points_per_hour=4):
    """
    Simulate `days` days of GPS pings for a single fictional user following
    a semi-regular daily routine, with occasional realistic anomalies
    (a few late-night/off-pattern trips) injected near the end.
    """
    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")

    for day in range(days):
        date = start + timedelta(days=day)
        is_weekend = date.weekday() >= 5

        # --- day-to-day variability, so the schedule isn't identical clockwork ---
        daily_shift = int(RNG.integers(-1, 2))          # routine boundaries wobble +/-1hr
        skip_gym = (not is_weekend) and (RNG.random() < 0.35)     # ~35% of weekdays: no gym
        detour_day = RNG.random() < 0.12                # ~12% of days: an extra out-of-routine stop
        detour_anchor = None
        if detour_day:
            detour_anchor = list(ANCHORS.values())[int(RNG.integers(0, len(ANCHORS)))]
            detour_hour = int(RNG.integers(11, 21))

        for hour in range(24):
            label = _hour_bucket_location(hour, is_weekend, shift=daily_shift)

            if skip_gym and label == "gym":
                label = "home"

            if detour_day and hour == detour_hour:
                label = "detour"

            if label == "commute_to_work":
                frac = np.clip((hour - 7) / 2, 0, 1)
                base = _interpolate(ANCHORS["home"], ANCHORS["work"], frac)
            elif label == "commute_to_home":
                frac = np.clip((hour - 18) / 2, 0, 1)
                base = _interpolate(ANCHORS["work"], ANCHORS["home"], frac)
            elif label == "detour":
                base = detour_anchor
            else:
                base = ANCHORS.get(label, ANCHORS["home"])

            # randomize how many pings this hour actually gets (real GPS logs
            # aren't perfectly uniform -- sometimes fewer/more pings land)
            n_points = max(1, points_per_hour + int(RNG.integers(-1, 2)))

            for p in range(n_points):
                minute = int(RNG.integers(0, 60))
                ts = date.replace(hour=hour, minute=min(minute, 59), second=int(RNG.integers(0, 59)))
                # wider, more variable jitter -- real people don't stand on
                # one exact GPS point the whole time they're "at" a place
                lat, lon = _jitter(*base, scale=RNG.uniform(0.0012, 0.0028))
                records.append([user_id, lat, lon, ts])

    df = pd.DataFrame(records, columns=["user_id", "latitude", "longitude", "timestamp"])

    # ---- inject a handful of anomalous / off-pattern movements in the last week ----
    anomaly_date = start + timedelta(days=days - 3)
    anomalous_spot = (28.70, 77.35)  # far from all normal anchors -> anomaly
    for h in [1, 2, 3]:
        ts = anomaly_date.replace(hour=h, minute=int(RNG.integers(0, 59)))
        lat, lon = _jitter(*anomalous_spot, scale=0.003)
        df.loc[len(df)] = [user_id, lat, lon, ts]

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_case_file(user_id="U001", last_seen_date=None):
    """Create a fictional missing-person case record consistent with the trajectory."""
    if last_seen_date is None:
        last_seen_date = datetime.now().strftime("%Y-%m-%d")
    case = {
        "case_id": "CASE-2025-0001",
        "subject_id": user_id,
        "subject_name": "Fictional Subject A (synthetic case, not a real person)",
        "age_group": "25-34",
        "last_known_latitude": ANCHORS["work"][0],
        "last_known_longitude": ANCHORS["work"][1],
        "last_known_timestamp": f"{last_seen_date} 18:45:00",
        "reported_missing_timestamp": f"{last_seen_date} 22:00:00",
        "notes": "Synthetic/fictional case created for academic ML project purposes only.",
    }
    return pd.DataFrame([case])


def main():
    out_dir = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = generate_user_trajectory()
    traj.to_csv(out_dir / "synthetic_gps_trajectory.csv", index=False)

    case = generate_case_file()
    case.to_csv(out_dir / "synthetic_case_file.csv", index=False)

    print(f"Wrote {len(traj)} GPS points -> {out_dir/'synthetic_gps_trajectory.csv'}")
    print(f"Wrote case file -> {out_dir/'synthetic_case_file.csv'}")


if __name__ == "__main__":
    main()
