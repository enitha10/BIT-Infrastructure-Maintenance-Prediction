import pandas as pd
import numpy as np
import hashlib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

REQUIRED_COLUMNS = [
    "asset_id", "asset_type", "age_years", "usage_hours_per_day",
    "temperature_c", "error_count", "maintenance_count_12m",
    "days_since_maintenance", "failure_next_14_days"
]
NUMERIC_FEATURES = [
    "age_years", "usage_hours_per_day", "temperature_c", "error_count",
    "maintenance_count_12m", "days_since_maintenance"
]
CATEGORICAL_FEATURES = ["asset_type"]

def prepare_data(raw):
    df = raw.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    for c in NUMERIC_FEATURES + ["failure_next_14_days"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["asset_id"] = df["asset_id"].fillna("UNKNOWN").astype(str)
    df["asset_type"] = df["asset_type"].fillna("Unknown").astype(str)
    # Never convert an unknown outcome into a negative label. Keep only verified 0/1 labels.
    df["failure_next_14_days"] = pd.to_numeric(df["failure_next_14_days"], errors="coerce")
    df = df.dropna(subset=["failure_next_14_days"]).copy()
    df = df[df["failure_next_14_days"].isin([0, 1])].copy()
    df["failure_next_14_days"] = df["failure_next_14_days"].astype(int)
    if df.empty:
        raise ValueError("No valid labelled rows found. failure_next_14_days must contain verified 0 or 1 values.")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df["failure_next_14_days"].astype(int)
    # Imputation is performed inside the model pipeline to avoid train/test leakage.
    return X, y, df

def make_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                             ("cat", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocess", pre),
                     ("model", RandomForestClassifier(n_estimators=200, random_state=42,
                                                      class_weight="balanced", min_samples_leaf=2))])

def train_model(X, y):
    if len(y) < 10:
        raise ValueError("At least 10 labelled rows are recommended for a demonstration split.")
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    if y_train.nunique() < 2:
        # Very small/imbalanced datasets may not contain both classes in training.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = make_pipeline()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0)
    }
    try:
        names = model.named_steps["preprocess"].get_feature_names_out()
        values = model.named_steps["model"].feature_importances_
        fi = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False).head(12)
    except Exception:
        fi = pd.DataFrame({"feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES, "importance": [0]*len(NUMERIC_FEATURES + CATEGORICAL_FEATURES)})
    # Refit on all available records for the app's current predictions after measuring hold-out metrics.
    model.fit(X, y)
    return model, metrics, fi

def predict_assets(model, X, clean):
    probabilities = model.predict_proba(X)[:, 1]
    result = clean[["asset_id", "asset_type"]].copy()
    result["failure_probability"] = probabilities
    result["failure_probability"] = result["failure_probability"].clip(0, 1)
    return result


def data_fingerprint(X, y):
    """Stable fingerprint for the feature matrix and target used to train the model."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return digest.hexdigest()
