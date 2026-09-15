"""
run_pipeline.py -- runs the full CASEFILE pipeline end-to-end, in order:

1. generate_synthetic_data.py  (skip this step and drop real GeoLife data
   into data/raw/ if you obtained it -- see README.md)
2. preprocessing.py
3. feature_engineering.py
4. clustering.py
5. anomaly_detection.py
6. prediction.py
7. route_prediction.py
8. search_priority.py
9. explainability.py
10. mapping.py

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "src"

STEPS = [
    "generate_synthetic_data.py",
    "preprocessing.py",
    "feature_engineering.py",
    "clustering.py",
    "anomaly_detection.py",
    "prediction.py",
    "route_prediction.py",
    "search_priority.py",
    "explainability.py",
    "mapping.py",
]


def main():
    for step in STEPS:
        print(f"\n{'='*60}\nRUNNING: {step}\n{'='*60}")
        result = subprocess.run([sys.executable, str(SRC / step)], cwd=str(SRC))
        if result.returncode != 0:
            print(f"Step {step} failed (exit code {result.returncode}). Stopping.")
            sys.exit(result.returncode)
    print("\nPipeline complete. Now run: streamlit run app/app.py")


if __name__ == "__main__":
    main()
