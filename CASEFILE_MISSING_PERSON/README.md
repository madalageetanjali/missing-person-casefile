<<<<<<< HEAD
# CASEFILE: AI-Powered Missing Person Investigation & Location Prediction System

Individual Advanced Machine Learning project. Analyzes movement-pattern data
to identify frequently visited locations, detect anomalous movement, predict
probable locations/routes, and rank search-priority areas for a **fictional**
missing-person case.

> ⚠️ **This is an academic simulation only.** All identities and case details
> in this repository are synthetic/fictional. Predictions are probabilistic
> and must never be used, or presented, as proof of a real person's location.
> Do not use this system, or adapt it, for real missing-person cases.

---

## 1. What's already included (no download needed)

The pipeline runs **out of the box** on synthetically generated data, so you
can demo the entire system with zero external downloads:

```bash
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/app.py
```

`run_pipeline.py` will: generate a synthetic GPS trajectory + fictional case
file → clean it → engineer features → cluster movement patterns → detect
anomalies → train/compare location-prediction models → predict a probable
route → compute search-priority scores → generate explanations → build the
interactive map. Then the Streamlit app ties it all together in a dashboard.

## 2. What you should download for the "real" version of this project

Your project guide (`CASEFILE_Missing_Person_...Guide.docx`) asks for real,
public datasets. This sandbox environment can't reach those external sites,
so I generated realistic synthetic data as a stand-in with the **same
schema** — swap it out with real data and the rest of the pipeline (Modules
2–11) works unchanged. To use real data, download:

| # | Dataset | Where to get it | What to do with it |
|---|---|---|---|
| 1 | **Microsoft GeoLife GPS Trajectory Dataset** (required — main trajectory data) | https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/ (direct download link is on that page) | Unzip into `data/raw/geolife/`. Each `.plt` file is one trajectory; write a small loader (see note below) to combine them into the same columns used everywhere else: `user_id, latitude, longitude, timestamp`. |
| 2 | **OpenStreetMap extract** (optional — for roads/POIs context) | https://www.openstreetmap.org/ (use the "Export" tool, or https://download.geofabrik.de/ for a full-region extract) | Save to `data/raw/osm/`. Use with `osmnx` or `geopandas` (not included by default — see requirements.txt comments) to add POI/road context features. |
| 3 | **data.gov.in datasets** (optional — extra contextual data) | https://www.data.gov.in/ | Save to `data/raw/context/`, document in your Data Source table (see `reports/investigation_report_template.md`). |

**GeoLife loader note:** each `.plt` file's first 6 lines are header junk;
data rows are `lat,lon,0,altitude,days_since_epoch,date,time`. A minimal
loader:
```python
import pandas as pd
def load_plt(path, user_id):
    df = pd.read_csv(path, skiprows=6, header=None,
                      names=["latitude","longitude","_zero","altitude",
                             "_days","date","time"])
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"])
    df["user_id"] = user_id
    return df[["user_id","latitude","longitude","timestamp"]]
```
Concatenate all users' trajectories into one CSV at
`data/raw/geolife_combined.csv`, then point `src/preprocessing.py`'s
`raw_path` at it instead of the synthetic file.

## 3. Required pip installs

Everything needed is in `requirements.txt`:

```
pandas, numpy, scipy, scikit-learn, xgboost, shap, folium, streamlit,
matplotlib, joblib
```

Install with:
```bash
pip install -r requirements.txt
```

`geopandas`/`osmnx` are listed as **optional** (commented out) — only needed
if you bring in the OpenStreetMap dataset for polygon-based geography.
They can be finicky to install (GDAL dependency); the project works fully
without them using plain lat/lon + `folium`.

## 4. Folder structure

```
CASEFILE_MISSING_PERSON/
│
├── data/
│   ├── raw/            <- put real downloaded datasets here (empty by default)
│   ├── processed/      <- pipeline writes cleaned/feature/cluster data here
│   └── synthetic/      <- synthetic GPS trajectory + fictional case file
│
├── notebooks/           <- one notebook per project phase, already executed
│   ├── 01_data_collection.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_anomaly_detection.ipynb
│   ├── 06_location_prediction.ipynb
│   └── 07_route_prediction.ipynb   (also covers Modules 8-9: search priority + XAI)
│
├── models/               <- trained model artifacts (.pkl), written by the pipeline
├── src/                  <- all pipeline source code (see below)
├── app/app.py            <- Streamlit dashboard (Module 11)
├── reports/              <- generated rankings, model comparison, map, report template
├── run_pipeline.py       <- runs the whole pipeline end-to-end
├── requirements.txt
└── README.md             <- this file
```

### `src/` module map (matches the project guide's Modules 1–10)

| File | Module |
|---|---|
| `generate_synthetic_data.py` | Module 1 (stand-in data source) |
| `preprocessing.py` | Module 2: Data Preprocessing |
| `feature_engineering.py` | Module 3: Feature Engineering |
| `clustering.py` | Module 4: Movement Pattern Analysis (K-Means + DBSCAN) |
| `anomaly_detection.py` | Module 5: Anomaly Detection (Isolation Forest, LOF, One-Class SVM) |
| `prediction.py` | Module 6: Location Prediction (Random Forest, XGBoost, KNN) |
| `route_prediction.py` | Module 7: Route Prediction (Markov Chain) |
| `search_priority.py` | Module 8: Search Priority Score |
| `explainability.py` | Module 9: Explainable AI (rule-based + SHAP hook) |
| `mapping.py` | Module 10: Interactive Map (Folium) |
| `app/app.py` | Module 11: Interactive Dashboard (Streamlit) |

## 5. Running each stage individually

```bash
cd CASEFILE_MISSING_PERSON
python src/generate_synthetic_data.py     # or plug in real GeoLife data
python src/preprocessing.py
python src/feature_engineering.py
python src/clustering.py
python src/anomaly_detection.py
python src/prediction.py
python src/route_prediction.py
python src/search_priority.py
python src/explainability.py
python src/mapping.py                     # writes reports/investigation_map.html
streamlit run app/app.py                  # full interactive dashboard
```

Or just run all of them: `python run_pipeline.py`

## 6. What's left for you to write (the guide requires these to be your own work)

This scaffold gives you a fully working, tested pipeline and code — but your
project guide (Section 13: Individual Project Requirements) requires the
report, presentation, and viva explanations to be **your own independent
work**. Still to do:

- Swap in the real GeoLife dataset (Section 2 above) if your instructor
  requires real public data rather than synthetic data.
- Fill in `reports/investigation_report_template.md` with your own writing,
  screenshots, and analysis (do not submit generated boilerplate as-is).
- Build `reports/presentation.pptx` from your findings.
- Review every module's code so you can explain it in the viva — algorithm
  choices, hyperparameters, evaluation metrics, and limitations are all your
  responsibility to understand and defend.

## 7. Ethical requirements (from the project guide, Section 16)

- Use fictional case identities only — never real people's data.
- Do not use personally identifiable information or private datasets.
- Always state predictions are probabilistic, never definitive proof.
- Never label individuals as criminal/suspicious based solely on model
  output (this applies especially to the anomaly-detection module).
- Document model limitations, bias, and possible false positives in your
  report.
=======
# casefile-missing-person
>>>>>>> 00911b196ca0d7219d2b153b3689939e8730a1e0
