# BIT Infrastructure Maintenance Prediction Using Historical Data

A runnable Phase 2 software proof-of-concept for the project described in the first-review PPT. It includes CSV ingestion, data quality checks, a Random Forest failure-risk classifier, evaluation metrics, feature importance, asset risk ranking, and a Streamlit dashboard.

## Important honesty note
`data/sample_maintenance_data.csv` is **synthetic demo data**, not real BIT campus data. Do not present its scores as real-world results. For the hackathon, collect authorized real observations and photos, keep evidence of collection, and upload them to the organiser's Drive folder. The app supports CSV upload.

## Requirements
- Python 3.10 or newer
- Internet connection for the first installation only
- Windows PowerShell, Command Prompt, macOS Terminal, or Linux shell

## Run on Windows (PowerShell)
Open PowerShell in the extracted project folder, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:
```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py` is not recognized, install Python 3.10+ from https://www.python.org/downloads/ and enable **Add Python to PATH** during installation.

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The terminal will print a local URL, usually `http://localhost:8501`. Open it in your browser. Stop the app with `Ctrl+C`.


## Train the model and use the prediction database

The application trains a Random Forest pipeline from the selected labelled CSV, evaluates it on a held-out split, and saves the fitted pipeline to `models/trained_model.joblib`. The app also writes evaluation metadata to `outputs/model_metrics.json`.

To train separately before launching the dashboard:

```powershell
python train_model.py
```

The Streamlit app also trains/saves the model from the currently selected CSV when it starts.

### SQLite prediction database
- The app automatically creates `database/predictions.db` and the `prediction_history` table.
- When a user submits the **Estimate risk and save prediction** form, the app stores the input features, predicted failure probability, risk tier, recommended action, model version, and timestamp.
- The **Saved prediction history (SQLite)** section displays saved records and lets you download them as CSV.
- Dataset-wide risk rankings are displayed separately; they are not silently inserted as user-generated prediction history.
- To inspect the database, open `database/predictions.db` with DB Browser for SQLite, choose **Browse Data**, and select `prediction_history`. You can also run:
  ```sql
  SELECT * FROM prediction_history ORDER BY prediction_id DESC;
  ```

The database file is created automatically on first app launch. It will initially have an empty table; generate one or more predictions in the dashboard to populate it.

### Important data-quality note
The included 240-row CSV is synthetic demonstration data. The model is genuinely fitted by the training code, but its scores are not evidence of real BIT campus performance. Only train on real historical data when the target labels are verified (failure within the following 14 days = 1; otherwise = 0). Unknown labels are excluded rather than silently treated as no-failure labels.

## CSV format
Required column names (exact spelling):
- `asset_id`: non-sensitive asset identifier, e.g. BIT-LAB-001
- `asset_type`: Server, Network Device, Storage, Lab PC, UPS, etc.
- `age_years`: equipment age in years
- `usage_hours_per_day`: approximate operating hours/day
- `temperature_c`: temperature in Celsius, if measured
- `error_count`: count of relevant errors in a defined recent period
- `maintenance_count_12m`: maintenance events in the last 12 months
- `days_since_maintenance`: days since last maintenance
- `failure_next_14_days`: confirmed target label; 1 if a failure occurred within the following 14 days, otherwise 0

Use `data/real_data_template.csv` as the blank collection template. Only assign a failure label after the observation window has passed and the event is verified. If you do not have historical outcomes, you cannot honestly train/evaluate a supervised failure predictor yet.

## Data collection protocol
1. Obtain permission from the BIT lab/IT/facilities staff.
2. Agree on which assets and fields can be recorded.
3. Record date/time in a separate controlled sheet if needed; do not add confidential records to the demo app.
4. Take photos of the collection activity only where permitted. Avoid faces, screens with private data, serial numbers, passwords, and network diagrams.
5. Upload the real dataset and collection photos to the organiser's Google Drive folder. Use folder name `Team number-Leader Name`.
6. Keep a copy of consent/permission notes and a small data dictionary.
7. Report dataset size, collection dates, class balance, missing values, and model metrics honestly.

## Prototype architecture
Authorised campus records / CSV -> validation and cleaning -> imputation + encoding -> Random Forest classifier -> risk score and risk tier -> dashboard and downloadable CSV report.

## Limitations
- The included data is synthetic.
- Risk thresholds (35% and 70%) are illustrative and must be calibrated with stakeholders.
- This is a decision-support prototype, not a safety-critical alarm system.
- It does not connect to live campus systems or automatically dispatch work orders.
- A valid supervised model needs enough examples of both classes and trustworthy labels.
