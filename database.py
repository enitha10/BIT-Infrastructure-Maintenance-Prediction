"""SQLite persistence for infrastructure maintenance predictions."""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = PROJECT_DIR / "database"
DB_PATH = DB_DIR / "predictions.db"

def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_id TEXT NOT NULL,
                asset_type TEXT NOT NULL,
                age_years REAL NOT NULL,
                usage_hours_per_day REAL NOT NULL,
                temperature_c REAL NOT NULL,
                error_count INTEGER NOT NULL,
                maintenance_count_12m INTEGER NOT NULL,
                days_since_maintenance INTEGER NOT NULL,
                failure_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommended_action TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prediction_date TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_date
            ON prediction_history(prediction_date)
        """)

def save_prediction(record):
    init_db()
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO prediction_history (
                asset_id, asset_type, age_years, usage_hours_per_day,
                temperature_c, error_count, maintenance_count_12m,
                days_since_maintenance, failure_probability, risk_level,
                recommended_action, model_version, prediction_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(record["asset_id"]), str(record["asset_type"]),
            float(record["age_years"]), float(record["usage_hours_per_day"]),
            float(record["temperature_c"]), int(record["error_count"]),
            int(record["maintenance_count_12m"]), int(record["days_since_maintenance"]),
            float(record["failure_probability"]), str(record["risk_level"]),
            str(record["recommended_action"]), str(record.get("model_version", "rf-v1")),
            datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        ))
        return cur.lastrowid

def fetch_predictions(limit=500):
    init_db()
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM prediction_history
            ORDER BY prediction_id DESC
            LIMIT ?
        """, (int(limit),)).fetchall()
    return [dict(row) for row in rows]

def prediction_count():
    init_db()
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
