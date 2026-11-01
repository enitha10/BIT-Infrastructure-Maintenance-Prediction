"""Train and persist the Random Forest model from the selected labelled CSV."""
import json
from pathlib import Path

import joblib
import pandas as pd

from model_utils import prepare_data, train_model, data_fingerprint

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_maintenance_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "trained_model.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    X, y, _ = prepare_data(raw)
    if y.nunique() < 2:
        raise ValueError("Training requires verified examples of both target classes (0 and 1).")
    model, metrics, feature_importance = train_model(X, y)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "data_fingerprint": data_fingerprint(X, y),
        "model_version": "random-forest-v1"
    }
    joblib.dump(bundle, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump({
            "model_version": "random-forest-v1",
            "training_file": str(DATA_PATH.relative_to(ROOT)),
            "records_used": int(len(y)),
            "data_fingerprint": data_fingerprint(X, y),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
            **{key: float(value) for key, value in metrics.items()}
        }, file, indent=2)

    print(f"Saved trained model: {MODEL_PATH}")
    print(f"Saved evaluation metrics: {METRICS_PATH}")
    print("Hold-out metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print("Top feature importances:")
    print(feature_importance.to_string(index=False))

if __name__ == "__main__":
    main()
