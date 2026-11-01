import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from model_utils import prepare_data, train_model, predict_assets, REQUIRED_COLUMNS, data_fingerprint
from database import init_db, save_prediction, fetch_predictions, prediction_count

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "trained_model.joblib"
METRICS_PATH = PROJECT_DIR / "outputs" / "model_metrics.json"
MODEL_VERSION = "random-forest-v1"

st.set_page_config(page_title="BIT Infrastructure Maintenance Prediction",
                   page_icon="🛠️", layout="wide")
init_db()

st.title("🛠️ BIT Infrastructure Maintenance Prediction Using Historical Data")
st.caption("Historical-data-based maintenance risk decision support")

st.info(
    "Demo note: The included dataset is synthetic sample data, not verified BIT campus records. "
    "Model metrics and risk estimates are for prototype demonstration only."
)

@st.cache_data
def load_default():
    path = DATA_DIR / "sample_maintenance_data.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

uploaded = st.sidebar.file_uploader("Upload maintenance CSV (optional)", type=["csv"])
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    data_source = "Uploaded CSV"
else:
    raw = load_default()
    data_source = "Included synthetic demo dataset"

st.sidebar.write(f"Data source: **{data_source}**")
st.sidebar.write("Expected columns:")
st.sidebar.code(", ".join(REQUIRED_COLUMNS))

if raw.empty:
    st.error("No dataset found. Place a CSV at data/sample_maintenance_data.csv or upload one in the sidebar.")
    st.stop()

st.subheader("1. Dataset preview")
st.write(f"Rows: {len(raw)} | Columns: {len(raw.columns)}")
st.dataframe(raw.head(20), use_container_width=True)

missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.markdown("See `data/real_data_template.csv` for the expected format.")
    st.stop()

try:
    X, y, clean = prepare_data(raw)
except Exception as exc:
    st.error(f"Could not prepare dataset: {exc}")
    st.stop()

if y.nunique() < 2:
    st.warning("The target column needs both 0 (no failure) and 1 (failure) examples to train a classifier.")
    st.stop()

current_fingerprint = data_fingerprint(X, y)
model_status = "Loaded saved trained model"
bundle = None
if MODEL_PATH.exists():
    try:
        candidate = joblib.load(MODEL_PATH)
        if isinstance(candidate, dict) and candidate.get("data_fingerprint") == current_fingerprint:
            bundle = candidate
    except Exception:
        bundle = None

if bundle is None:
    with st.spinner("Training and evaluating the Random Forest model..."):
        model, metrics, feature_importance = train_model(X, y)
        bundle = {
            "model": model,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "data_fingerprint": current_fingerprint,
            "model_version": MODEL_VERSION
        }
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        model_status = "Trained and saved a new model"
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with METRICS_PATH.open("w", encoding="utf-8") as file:
            json.dump(
                {"model_version": MODEL_VERSION, "data_source": data_source,
                 "records_used": int(len(y)), "data_fingerprint": current_fingerprint,
                 **{k: float(v) for k, v in metrics.items()}},
                file, indent=2
            )
else:
    model = bundle["model"]
    metrics = bundle["metrics"]
    feature_importance = bundle["feature_importance"]

st.caption(f"{model_status}. Model file: `{MODEL_PATH.relative_to(PROJECT_DIR)}`")

st.subheader("2. Data quality")
c1, c2, c3 = st.columns(3)
c1.metric("Missing cells", int(raw[REQUIRED_COLUMNS].isna().sum().sum()))
c2.metric("Assets", int(raw["asset_id"].nunique()))
c3.metric("Failure records", int(pd.to_numeric(raw["failure_next_14_days"], errors="coerce").fillna(0).sum()))
st.dataframe(clean.head(10), use_container_width=True)

st.subheader("3. Model evaluation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
m2.metric("Precision", f"{metrics['precision']:.1%}")
m3.metric("Recall", f"{metrics['recall']:.1%}")
m4.metric("F1-score", f"{metrics['f1']:.1%}")
st.caption("Metrics use a held-out test split. Synthetic/small data does not establish real-world campus performance.")
st.bar_chart(feature_importance.set_index("feature")["importance"])

st.subheader("4. Dataset asset risk ranking")
predictions = predict_assets(model, X, clean)

def risk_label(probability):
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.35:
        return "MEDIUM"
    return "LOW"

predictions["risk_level"] = predictions["failure_probability"].apply(risk_label)
predictions["recommended_action"] = predictions["risk_level"].map({
    "HIGH": "Inspect soon; verify logs and schedule maintenance",
    "MEDIUM": "Review trend and plan preventive inspection",
    "LOW": "Continue routine monitoring"
})
predictions = predictions.sort_values("failure_probability", ascending=False)
st.dataframe(predictions, use_container_width=True)
st.download_button(
    "Download dataset risk report CSV",
    predictions.to_csv(index=False).encode("utf-8"),
    "bit_asset_risk_report.csv", "text/csv"
)
st.caption("This ranking is calculated from the selected dataset. It is not automatically inserted into prediction history.")

st.subheader("5. Generate and save a new prediction")
st.write("Each successful submission below is saved automatically in the SQLite database.")
with st.form("new_observation"):
    a1, a2, a3 = st.columns(3)
    asset_id = a1.text_input("Asset ID", "BIT-LAB-NEW")
    asset_type = a2.selectbox("Asset type", ["Server", "Network Device", "Storage", "Lab PC", "UPS", "Other"])
    age = a3.number_input("Age (years)", min_value=0.0, max_value=40.0, value=3.0)
    b1, b2, b3 = st.columns(3)
    usage = b1.number_input("Usage hours/day", min_value=0.0, max_value=24.0, value=8.0)
    temp = b2.number_input("Operating temperature (°C)", min_value=0.0, max_value=120.0, value=40.0)
    errors = b3.number_input("Error count (recent period)", min_value=0, max_value=100000, value=2)
    c1, c2 = st.columns(2)
    maintenance_count = c1.number_input("Maintenance count (last 12 months)", min_value=0, max_value=1000, value=1)
    days_since = c2.number_input("Days since last maintenance", min_value=0, max_value=10000, value=90)
    submitted = st.form_submit_button("Estimate risk and save prediction", type="primary")

if submitted:
    if not asset_id.strip():
        st.error("Please enter an asset ID.")
    else:
        observation = pd.DataFrame([{
            "asset_id": asset_id.strip(),
            "asset_type": asset_type,
            "age_years": age,
            "usage_hours_per_day": usage,
            "temperature_c": temp,
            "error_count": errors,
            "maintenance_count_12m": maintenance_count,
            "days_since_maintenance": days_since,
            # Placeholder label is required by the shared validation helper.
            # It is not included in the model's feature columns.
            "failure_next_14_days": 0
        }])
        try:
            one_X, _, _ = prepare_data(observation)
            probability = float(model.predict_proba(one_X)[0, 1])
            level = risk_label(probability)
            action = {
                "HIGH": "Inspect soon; verify logs and schedule maintenance",
                "MEDIUM": "Review trend and plan preventive inspection",
                "LOW": "Continue routine monitoring"
            }[level]
            record = {
                "asset_id": asset_id.strip(), "asset_type": asset_type,
                "age_years": age, "usage_hours_per_day": usage,
                "temperature_c": temp, "error_count": errors,
                "maintenance_count_12m": maintenance_count,
                "days_since_maintenance": days_since,
                "failure_probability": probability, "risk_level": level,
                "recommended_action": action, "model_version": MODEL_VERSION
            }
            new_id = save_prediction(record)
            st.success(f"Prediction saved successfully. Database record ID: {new_id}")
            st.metric("Estimated failure risk (next 14 days)", f"{probability:.1%}")
            st.metric("Risk level", level)
            st.write("Suggested next step:", action)
            st.warning("Prototype estimate only. Verify asset records with campus maintenance staff; this is not a safety-critical diagnosis.")
        except Exception as exc:
            st.error(f"Could not generate/save prediction: {exc}")

st.subheader("6. Saved prediction history (SQLite)")
history = pd.DataFrame(fetch_predictions(limit=500))
total_saved = prediction_count()
h1, h2, h3 = st.columns(3)
h1.metric("Total saved predictions", total_saved)
h2.metric("High-risk records", int((history["risk_level"] == "HIGH").sum()) if not history.empty else 0)
h3.metric("Database", "Connected")
if history.empty:
    st.info("No user-generated predictions have been saved yet. Submit a new observation above to create the first database record.")
else:
    st.dataframe(history, use_container_width=True)
    st.download_button(
        "Download saved prediction history CSV",
        history.to_csv(index=False).encode("utf-8"),
        "bit_prediction_history.csv", "text/csv"
    )

st.subheader("7. Real-data collection checklist")
st.markdown("""
- Obtain permission from the relevant campus IT/facilities team.
- Record asset ID/type, observation date, usage, temperature (if available), error count,
  maintenance history, and confirmed failure label.
- Do not upload personal data, passwords, network secrets, or confidential logs.
- Clearly identify synthetic sample data; do not report its metrics as real campus performance.
""")
