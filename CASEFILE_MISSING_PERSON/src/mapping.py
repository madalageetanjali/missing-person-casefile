"""
mapping.py -- Module 10: Interactive Map

Builds an interactive Folium map showing:
  - last known location
  - frequently visited areas
  - predicted probable areas
  - probable route
  - anomalous locations
  - search-priority areas (color/size coded by score)
"""

import pandas as pd
import folium
from pathlib import Path


def build_investigation_map(last_known: tuple,
                             frequent_locations: pd.DataFrame = None,
                             predicted_locations: pd.DataFrame = None,
                             route_df: pd.DataFrame = None,
                             anomalies_df: pd.DataFrame = None,
                             search_priority_df: pd.DataFrame = None,
                             zoom_start: int = 13) -> folium.Map:

    m = folium.Map(location=last_known, zoom_start=zoom_start, tiles="OpenStreetMap")

    # Last known location
    folium.Marker(
        location=last_known,
        popup="Last Known Location",
        icon=folium.Icon(color="red", icon="exclamation-sign"),
    ).add_to(m)

    # Frequently visited areas
    if frequent_locations is not None:
        for _, row in frequent_locations.iterrows():
            folium.CircleMarker(
                location=(row["center_lat"], row["center_lon"]),
                radius=6 + min(row.get("visit_count", 1) / 5, 10),
                color="blue", fill=True, fill_opacity=0.5,
                popup=f"Frequently visited (visits={row.get('visit_count', '?')})",
            ).add_to(m)

    # Predicted probable areas
    if predicted_locations is not None:
        for _, row in predicted_locations.iterrows():
            folium.CircleMarker(
                location=(row["center_lat"], row["center_lon"]),
                radius=8, color="green", fill=True, fill_opacity=0.6,
                popup=f"Predicted area (p={row.get('probability', 0):.2f})",
            ).add_to(m)

    # Probable route
    if route_df is not None and len(route_df.dropna(subset=["lat", "lon"])) > 1:
        coords = list(zip(route_df["lat"], route_df["lon"]))
        folium.PolyLine(coords, color="purple", weight=4, opacity=0.8,
                         popup="Predicted route").add_to(m)
        for i, (lat, lon) in enumerate(coords):
            folium.Marker((lat, lon), popup=f"Route step {i}",
                           icon=folium.Icon(color="purple", icon="arrow-right")).add_to(m)

    # Anomalous locations
    if anomalies_df is not None:
        anom = anomalies_df[anomalies_df.get("anomaly_consensus", 0) == 1]
        for _, row in anom.iterrows():
            folium.CircleMarker(
                location=(row["latitude"], row["longitude"]),
                radius=5, color="orange", fill=True, fill_opacity=0.9,
                popup="Anomalous movement (not necessarily suspicious)",
            ).add_to(m)

    # Search priority areas (heat-style ranked markers)
    if search_priority_df is not None:
        for _, row in search_priority_df.iterrows():
            folium.CircleMarker(
                location=(row["center_lat"], row["center_lon"]),
                radius=10, color="black", weight=2, fill=True,
                fill_color="yellow", fill_opacity=0.4,
                popup=f"Search priority #{row.get('search_rank')} "
                      f"(score={row.get('search_priority_score', 0):.2f})",
            ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    case = pd.read_csv(base / "data" / "synthetic" / "synthetic_case_file.csv")
    last_known = (case["last_known_latitude"].iloc[0], case["last_known_longitude"].iloc[0])

    freq_path = base / "data" / "processed" / "frequently_visited_locations.csv"
    freq = pd.read_csv(freq_path) if freq_path.exists() else None

    route_path = base / "data" / "processed" / "predicted_route.csv"
    route = pd.read_csv(route_path) if route_path.exists() else None

    anom_path = base / "data" / "processed" / "trajectory_anomalies.csv"
    anomalies = pd.read_csv(anom_path) if anom_path.exists() else None

    priority_path = base / "reports" / "search_priority_ranking.csv"
    priority = pd.read_csv(priority_path) if priority_path.exists() else None

    m = build_investigation_map(
        last_known=last_known,
        frequent_locations=freq,
        route_df=route,
        anomalies_df=anomalies,
        search_priority_df=priority,
    )
    out = base / "reports" / "investigation_map.html"
    m.save(str(out))
    print(f"Map saved -> {out}")
