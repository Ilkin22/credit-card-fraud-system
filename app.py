"""Flask application serving the intelligent credit card fraud detection system.

The API accepts transaction details, predicts fraud probability, maps it to
a risk level and recommended action, logs the decision into a SQLite
database, and exposes endpoints for both the web dashboard and recent
transaction history. The user interface lives in the `templates` and
`static` folders.
"""

from __future__ import annotations

import json
from datetime import datetime
import sqlite3
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

from src.utils import (
    FEATURES,
    get_model_and_scaler_paths,
    map_probability_to_risk,
    log_transaction,
)


def init_db(db_path: Path) -> None:
    """Initialize the transactions database if it does not already exist."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            probability REAL,
            risk_level TEXT,
            action TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def create_app() -> Flask:
    # Explicitly specify template and static folders relative to the project root.  This
    # avoids issues when the app is executed from different working directories.
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Determine model and scaler paths
    model_path, scaler_path = get_model_and_scaler_paths()
    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Ensure data directory and DB exist
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    db_path = data_dir / "transactions.db"
    init_db(db_path)

    @app.route("/")
    def index():
        """Render the main dashboard page."""
        return render_template("index.html", features=FEATURES)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Predict fraud probability and return risk assessment."""
        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify(error="Invalid JSON payload."), 400

        # Validate inputs
        if not isinstance(data, dict):
            return jsonify(error="Invalid input. Expected JSON object."), 400

        missing_keys = set(FEATURES) - data.keys()
        if missing_keys:
            return (
                jsonify(error=f"Missing fields: {', '.join(missing_keys)}"),
                400,
            )

        # Attempt to parse values as floats
        try:
            values = [float(data[feature]) for feature in FEATURES]
        except (TypeError, ValueError):
            return (
                jsonify(error="Invalid input. All fields must be numeric."),
                400,
            )

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0][1])
        predicted_class = int(model.predict(X_scaled)[0])
        risk_level, action = map_probability_to_risk(prob)

        # Insert into DB via utility function
        log_transaction(
            timestamp=datetime.utcnow().isoformat(),
            amount=values[-1],
            probability=prob,
            risk_level=risk_level,
            action=action,
        )

        return jsonify(
            predicted_class=predicted_class,
            fraud_probability=prob,
            risk_level=risk_level,
            action=action,
        )

    @app.route("/recent", methods=["GET"])
    def recent():
        """Return the last 10 decisions from the database."""
        conn = get_db_connection(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timestamp, amount, probability, risk_level, action
            FROM transactions
            ORDER BY id DESC
            LIMIT 10
            """
        )
        rows = cur.fetchall()
        conn.close()
        result = [
            {
                "timestamp": row[0],
                "amount": row[1],
                "probability": row[2],
                "risk_level": row[3],
                "action": row[4],
            }
            for row in rows
        ]
        return jsonify(result)

    return app


if __name__ == "__main__":
    # Create and run the Flask app
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)