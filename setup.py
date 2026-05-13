"""Setup script to bootstrap the credit card fraud detection system locally.

Run this script using Python (3.9+). It will create the required
directory structure, write all necessary files with the correct
content, and prepare the environment for training and running the
application. This is useful when direct pushing to GitHub is not
possible.
"""

import os
from pathlib import Path


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    # Define file contents
    files = {
        "requirements.txt": """flask\npandas\nnumpy\nscikit-learn\nimbalanced-learn\njoblib\nmatplotlib\nseaborn\n""",
        "src/utils.py": """"""Utilities shared across the fraud detection system.

This module centralizes definitions for feature names and helper
functions used throughout the training script and the Flask app. Keeping
these values in one place reduces the risk of mismatches between
training and prediction.
""""""

from pathlib import Path
from typing import Tuple


# Define the feature columns used both for training and prediction.
FEATURES = [\"V1\", \"V2\", \"V3\", \"V4\", \"V5\", \"Amount\"]

# Define the name of the target column.
TARGET = \"Class\"


def ensure_models_dir() -> Path:
    """Ensure the `models/` directory exists at the project root.

    Returns
    -------
    Path
        The absolute Path object pointing to the models directory.
    """
    models_path = Path(__file__).resolve().parents[2] / \"models\"
    models_path.mkdir(parents=True, exist_ok=True)
    return models_path


def get_model_and_scaler_paths() -> Tuple[Path, Path]:
    """Return absolute paths to the saved model and scaler.

    Returns
    -------
    Tuple[Path, Path]
        A tuple containing paths for `model.pkl` and `scaler.pkl`.
    """
    models_dir = ensure_models_dir()
    model_path = models_dir / \"model.pkl\"
    scaler_path = models_dir / \"scaler.pkl\"
    return model_path, scaler_path


def map_probability_to_risk(probability: float) -> Tuple[str, str]:
    """Map a fraud probability to a risk level and recommended action.

    This helper centralizes the business rules for converting a raw
    probability into a categorical risk assessment and suggested
    mitigation step. Adjusting thresholds here will affect both the API
    and the user interface consistently.

    Parameters
    ----------
    probability : float
        The probability of the transaction being fraudulent (between 0 and 1).

    Returns
    -------
    Tuple[str, str]
        A tuple of the risk level and the recommended action.
    """
    if probability > 0.80:
        return \"Critical\", \"Block Transaction\"
    elif probability > 0.50:
        return \"Suspicious\", \"Request SMS Verification\"
    else:
        return \"Low\", \"Allow Transaction\"
""",
        "src/train_model.py": """"""Training script for the intelligent credit card fraud detection system.

This module handles data loading (either from a real credit card dataset
or synthetic generation), preprocessing, balancing via SMOTE, model
training using a Random Forest classifier, evaluation, and saving of
artifacts. Run this script to (re)train the model before serving
predictions through the Flask app.
""""""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

from src.utils import FEATURES, TARGET, ensure_models_dir, get_model_and_scaler_paths


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Load the credit card fraud dataset or generate a synthetic one.

    Parameters
    ----------
    data_dir : Path
        The directory containing the `creditcard.csv` file.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the feature columns and the target.
    """
    csv_path = data_dir / \"creditcard.csv\"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
        if missing_cols:
            raise ValueError(f\"Missing required columns in dataset: {missing_cols}\")
        return df[FEATURES + [TARGET]].copy()
    else:
        print(\"creditcard.csv not found. Generating synthetic dataset for testing...\")
        X, y = make_classification(
            n_samples=5000,
            n_features=len(FEATURES),
            n_informative=4,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=2,
            weights=[0.995, 0.005],
            random_state=42,
        )
        df = pd.DataFrame(X, columns=FEATURES)
        df[TARGET] = y
        return df


def train() -> Tuple[RandomForestClassifier, StandardScaler]:
    """Execute the full training pipeline and save the model and scaler.

    Returns
    -------
    Tuple[RandomForestClassifier, StandardScaler]
        The trained Random Forest classifier and fitted scaler.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / \"data\"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_dir)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train_resampled)

    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=[\"Safe\", \"Fraud\"])
    print(\"Classification Report:\n\", report)

    models_dir = ensure_models_dir()
    model_path, scaler_path = get_model_and_scaler_paths()
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f\"Model saved to {model_path}\")
    print(f\"Scaler saved to {scaler_path}\")

    return clf, scaler


if __name__ == \"__main__\":
    train()
""",
        "app.py": """"""Flask application serving the intelligent credit card fraud detection system.

The API accepts transaction details, predicts fraud probability, maps it to
a risk level and recommended action, logs the decision into a SQLite
database, and exposes endpoints for both the web dashboard and recent
transaction history. The user interface lives in the `templates` and
`static` folders.
""""""

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
)


def init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        \"\"\"
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            amount REAL,
            probability REAL,
            risk_level TEXT,
            action TEXT
        )
        \"\"\"
    )
    conn.commit()
    conn.close()


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def create_app() -> Flask:
    app = Flask(__name__)

    model_path, scaler_path = get_model_and_scaler_paths()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    project_root = Path(__file__).resolve().parent
    data_dir = project_root / \"data\"
    data_dir.mkdir(exist_ok=True)
    db_path = data_dir / \"transactions.db\"
    init_db(db_path)

    @app.route(\"/\")
    def index():
        return render_template(\"index.html\", features=FEATURES)

    @app.route(\"/predict\", methods=[\"POST\"])
    def predict():
        try:
            data = request.get_json(force=True)
        except Exception:
            return jsonify(error=\"Invalid JSON payload.\"), 400

        if not isinstance(data, dict):
            return jsonify(error=\"Invalid input. Expected JSON object.\"), 400

        missing_keys = set(FEATURES) - data.keys()
        if missing_keys:
            return (
                jsonify(error=f\"Missing fields: {', '.join(missing_keys)}\"),
                400,
            )

        try:
            values = [float(data[feature]) for feature in FEATURES]
        except (TypeError, ValueError):
            return (
                jsonify(error=\"Invalid input. All fields must be numeric.\"),
                400,
            )

        X = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0][1])
        predicted_class = int(model.predict(X_scaled)[0])
        risk_level, action = map_probability_to_risk(prob)

        conn = get_db_connection(db_path)
        cur = conn.cursor()
        cur.execute(
            \"\"\"
            INSERT INTO transactions (timestamp, amount, probability, risk_level, action)
            VALUES (?, ?, ?, ?, ?)
            \"\"\",
            (
                datetime.utcnow().isoformat(),
                values[-1],
                prob,
                risk_level,
                action,
            ),
        )
        conn.commit()
        conn.close()

        return jsonify(
            predicted_class=predicted_class,
            fraud_probability=prob,
            risk_level=risk_level,
            action=action,
        )

    @app.route(\"/recent\", methods=[\"GET\"])
    def recent():
        conn = get_db_connection(db_path)
        cur = conn.cursor()
        cur.execute(
            \"\"\"
            SELECT timestamp, amount, probability, risk_level, action
            FROM transactions
            ORDER BY id DESC
            LIMIT 10
            \"\"\"
        )
        rows = cur.fetchall()
        conn.close()
        result = [
            {
                \"timestamp\": row[0],
                \"amount\": row[1],
                \"probability\": row[2],
                \"risk_level\": row[3],
                \"action\": row[4],
            }
            for row in rows
        ]
        return jsonify(result)

    return app


if __name__ == \"__main__\":
    app = create_app()
    app.run(host=\"0.0.0.0\", port=5000, debug=False)
""",
        "templates/index.html": """<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>BankSecure – Fraud Management System</title>
    <!-- Bootstrap 5 CSS -->
    <link
      href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css\"
      rel=\"stylesheet\"
      integrity=\"sha384-ENjdO4Dr2bkBIFxQpeo1TAdHGld/6Y5bYl5HaEWld7y4HI42Htfdb24J5I4ycLtE\"
      crossorigin=\"anonymous\"
    />
    <!-- Custom CSS -->
    <link rel=\"stylesheet\" href=\"{{ url_for('static', filename='css/style.css') }}\" />
  </head>
  <body class=\"bg-dark text-light\">
    <!-- Navbar -->
    <nav class=\"navbar navbar-dark bg-primary mb-4\">
      <div class=\"container-fluid\">
        <span class=\"navbar-brand mb-0 h1\">BankSecure – Fraud Management System</span>
      </div>
    </nav>
    <div class=\"container\">
      <div class=\"row\">
        <!-- Input Form Column -->
        <div class=\"col-lg-4 mb-4\">
          <div class=\"card bg-secondary shadow-sm\">
            <div class=\"card-header\">Transaction Details</div>
            <div class=\"card-body\">
              <form id=\"transaction-form\">
                {% for feature in features %}
                <div class=\"mb-3\">
                  <label for=\"{{ feature }}\" class=\"form-label\">{{ feature }}</label>
                  <input
                    type=\"number\"
                    step=\"any\"
                    class=\"form-control\"
                    id=\"{{ feature }}\"
                    name=\"{{ feature }}\"
                    required
                  />
                </div>
                {% endfor %}
                <button type=\"submit\" class=\"btn btn-primary w-100\">Analyze Transaction</button>
              </form>
            </div>
          </div>
        </div>
        <!-- Result and Recent Column -->
        <div class=\"col-lg-8 mb-4\">
          <!-- Result Card -->
          <div id=\"result-card\" class=\"card text-dark mb-4\">
            <div class=\"card-body\">
              <h5 class=\"card-title\" id=\"result-title\">Fraud Detection Result</h5>
              <p class=\"card-text\" id=\"result-probability\">Fraud Probability: N/A</p>
              <p class=\"card-text\" id=\"result-level\">Risk Level: N/A</p>
              <p class=\"card-text\" id=\"result-action\">Recommended Action: N/A</p>
            </div>
          </div>
          <!-- Recent Decisions Table -->
          <div class=\"card bg-secondary shadow-sm\">
            <div class=\"card-header\">Recent Decisions</div>
            <div class=\"card-body\">
              <table class=\"table table-dark table-striped table-hover mb-0\" id=\"recent-table\">
                <thead>
                  <tr>
                    <th scope=\"col\">Time</th>
                    <th scope=\"col\">Amount</th>
                    <th scope=\"col\">Risk</th>
                    <th scope=\"col\">Action</th>
                    <th scope=\"col\">Prob</th>
                  </tr>
                </thead>
                <tbody></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
    <!-- Bootstrap 5 JS and dependencies -->
    <script
      src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js\"
      integrity=\"sha384-qmC6eNMRmgKEvF7Zd3rTyW07oVUAoz7ktk1bcGJ2L+HvN70WNRx0Qv2tc3DYdjwK\"
      crossorigin=\"anonymous\"
    ></script>
    <!-- Custom JS -->
    <script src=\"{{ url_for('static', filename='js/main.js') }}\"></script>
  </body>
</html>
""",
        "static/css/style.css": """/* Custom styling to complement the Bootstrap dark theme */
body {
  font-family: \"Segoe UI\", Tahoma, Geneva, Verdana, sans-serif;
}

.card {
  border-radius: 0.5rem;
}

.card-header {
  font-weight: bold;
}

#result-card {
  transition: background-color 0.3s ease-in-out;
}

#recent-table tbody tr td {
  vertical-align: middle;
}
""",
        "static/js/main.js": """// Attach event listeners and handle API interactions for the dashboard

document.addEventListener(\"DOMContentLoaded\", () => {
  const form = document.getElementById(\"transaction-form\");
  const resultCard = document.getElementById(\"result-card\");
  const resultProbability = document.getElementById(\"result-probability\");
  const resultLevel = document.getElementById(\"result-level\");
  const resultAction = document.getElementById(\"result-action\");
  const recentTableBody = document.querySelector(\"#recent-table tbody\");

  async function loadRecent() {
    try {
      const response = await fetch(\"/recent\");
      if (!response.ok) throw new Error(\"Failed to fetch recent decisions\");
      const data = await response.json();
      recentTableBody.innerHTML = \"\";
      data.forEach((row) => {
        const tr = document.createElement(\"tr\");
        tr.innerHTML = `
          <td>${new Date(row.timestamp).toLocaleString()}</td>
          <td>${row.amount.toFixed(2)}</td>
          <td>${row.risk_level}</td>
          <td>${row.action}</td>
          <td>${(row.probability * 100).toFixed(2)}%</td>
        `;
        recentTableBody.appendChild(tr);
      });
    } catch (err) {
      console.error(err);
    }
  }

  loadRecent();

  form.addEventListener(\"submit\", async (e) => {
    e.preventDefault();
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      payload[key] = value;
    });

    try {
      const response = await fetch(\"/predict\", {
        method: \"POST\",
        headers: {
          \"Content-Type\": \"application/json\",
        },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        alert(data.error || \"Prediction failed\");
        return;
      }

      const probPercent = (data.fraud_probability * 100).toFixed(2);
      resultProbability.textContent = `Fraud Probability: ${probPercent}%`;
      resultLevel.textContent = `Risk Level: ${data.risk_level}`;
      resultAction.textContent = `Recommended Action: ${data.action}`;
      resultCard.classList.remove(\"bg-success\", \"bg-warning\", \"bg-danger\");
      if (data.risk_level === \"Low\") {
        resultCard.classList.add(\"bg-success\");
      } else if (data.risk_level === \"Suspicious\") {
        resultCard.classList.add(\"bg-warning\");
      } else if (data.risk_level === \"Critical\") {
        resultCard.classList.add(\"bg-danger\");
      }
      loadRecent();
    } catch (err) {
      console.error(err);
      alert(\"An error occurred while processing the prediction.\");
    }
  });
});
""",
        "README.md": """# Intelligent Credit Card Fraud Detection System (MIS Portfolio)

## Overview

This project implements a decision support system for detecting fraudulent
credit card transactions. It leverages a **Random Forest classifier**
trained on imbalanced transaction data using **SMOTE** to balance the
classes. The system exposes a web dashboard through a Flask application
that allows analysts to input transaction details and receive a real-time
fraud probability, a categorical risk level, and a recommended action
(Block, Verify, or Allow). All predictions are logged into a small
**SQLite** database, and the last ten decisions are displayed on the
dashboard.

## Tech Stack

* **Python** (Flask)
* **scikit-learn** – Random Forest model and preprocessing
* **imbalanced-learn** – SMOTE for oversampling the minority class
* **SQLite** – lightweight database for transaction logs
* **Bootstrap 5** + **vanilla JS** – responsive dark-themed UI

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Ilkin22/credit-card-fraud-system.git
   cd credit-card-fraud-system
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:

   ```bash
   python src/train_model.py
   ```

   > If you have access to the Kaggle credit card dataset, place it at
   > `data/creditcard.csv` before running the training script. Otherwise
   > a synthetic dataset will be generated automatically.

4. **Run the application**:

   ```bash
   python app.py
   ```

5. **Usage**:

   * Navigate to `http://localhost:5000` in your web browser.
   * Enter values for the features `V1`, `V2`, `V3`, `V4`, `V5`, and `Amount`.
   * Click **Analyze Transaction** to receive a fraud probability, risk
     level, and recommended action.
   * Review the recent decisions in the table below the result card.

## MIS / Business Perspective

This system is designed as a component of a broader Management
Information System (MIS) for risk management. The risk level thresholds
and recommended actions can be tuned by fraud analysts to align with
internal policies and evolving threat models. Logged decisions stored in
SQLite provide a lightweight audit trail for monitoring model performance
and decision rationale, which is critical for regulatory compliance and
continuous improvement.
""",
    }
    # Create directories and write files
    for relative_path, content in files.items():
        write_file(project_root / relative_path, content)

    print("Project structure created successfully.")


if __name__ == "__main__":
    main()