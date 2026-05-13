# Intelligent Credit Card Fraud Detection System (MIS Portfolio)

## Overview

This repository contains a fully fledged **credit card fraud detection system** built as a decision‑support tool for Management Information Systems (MIS) and risk teams.  It demonstrates how to take a machine‑learning model from the data science notebook stage all the way to a production‑ready web application with a dark‑themed dashboard and lightweight logging.

At its core the system uses a **Random Forest** classifier trained on highly imbalanced transaction data.  To counter the skew towards legitimate transactions, the training pipeline applies **SMOTE** (Synthetic Minority Over‑sampling Technique).  After training, the model outputs a probability that a given transaction is fraudulent.  Business rules then map this probability to a **risk level** and an **action recommendation** (Block, Verify via SMS, or Allow).  Every prediction is logged into a simple **SQLite** database so that analysts can review recent decisions and audit model behaviour over time.

## Architecture

The project is organized into clear layers:

- **Data & Features** – The model operates on 28 PCA‑derived components (`V1`–`V28`) plus the transaction `Amount`.  The target column is named `Class` where `1` indicates fraud.
- **Training Pipeline** – Implemented in `src/train_model.py`.  It loads `data/creditcard.csv` if present (otherwise generates synthetic imbalanced data), splits the data with stratification, applies SMOTE to the training set, scales features with `StandardScaler`, trains a `RandomForestClassifier`, evaluates on a hold‑out test set, and saves both the model and scaler to `models/model.pkl` and `models/scaler.pkl` respectively.
- **Utilities** – Shared constants and helper functions live in `src/utils.py`.  This module defines the `FEATURES` list, the `map_probability_to_risk` function that encapsulates the risk thresholds, helpers to locate the `models` and `data` directories, and functions to log predictions to the SQLite database.
- **Flask API & Dashboard** – The web application in `app.py` exposes three routes:
  - `/` renders the dashboard where analysts can enter feature values and view results.
  - `/predict` accepts a JSON payload with keys matching `FEATURES`, scales the input, predicts the fraud probability, maps it to a risk level and action, logs the decision, and returns a JSON response.
  - `/recent` returns the last 10 logged transactions as JSON for the dashboard’s “Recent Decisions” table.
- **User Interface** – The UI is built with **Bootstrap 5** (dark theme) and minimal JavaScript.  The form inputs match the feature list, and the result card changes colour based on the risk level (green for Low, yellow/orange for Suspicious, red for Critical).  Recent decisions are displayed in a responsive table.
- **Logging Layer** – A lightweight SQLite database (`data/transactions.db`) stores each prediction with timestamp, amount, fraud probability, risk level, and recommended action.  The database is created on demand if it does not exist.

## Project Structure

```
credit-card-fraud-system/
├── README.md               # You are here
├── app.py                 # Flask application (API + dashboard)
├── requirements.txt       # Python dependencies
├── models/
│   ├── .gitkeep           # Placeholder to track the directory
│   ├── model.pkl          # Saved Random Forest (created after training)
│   └── scaler.pkl         # Saved StandardScaler (created after training)
├── data/
│   ├── .gitkeep           # Placeholder to track the directory
│   ├── creditcard.csv     # Optional real dataset (not provided in repo)
│   └── transactions.db    # SQLite log database (created at runtime)
├── src/
│   ├── __init__.py        # Package marker
│   ├── train_model.py     # End‑to‑end model training pipeline
│   └── utils.py           # Shared constants and helper functions
├── templates/
│   └── index.html         # Jinja2 template for the dashboard
└── static/
    ├── css/
    │   └── style.css      # Small CSS overrides
    └── js/
        └── main.js        # Front‑end logic (fetch API, DOM updates)
```

## Model & Evaluation

Running the training script will print a classification report on the hold‑out test set.  An example run on a synthetic dataset yielded the following metrics for the fraud class (`Class = 1`):

| Metric        | Fraud Class (1) |
|--------------:|---------------:|
| Precision     | 0.93           |
| Recall        | 0.88           |
| F1‑score      | 0.90           |

These numbers will vary depending on the data.  When using the real Kaggle dataset, the model may achieve different scores.  The training script reports the full `classification_report` for transparency.

## Risk Rules

The system maps the raw fraud probability to a categorical risk level and recommended action according to the following business rules:

| Fraud Probability `p` | Risk Level  | Action                    |
|----------------------:|------------:|---------------------------|
| `p > 0.80`            | Critical    | Block Transaction         |
| `0.50 < p ≤ 0.80`     | Suspicious  | Request SMS Verification  |
| `p ≤ 0.50`            | Low         | Allow Transaction         |

These thresholds are defined in `src/utils.py` and can be adjusted centrally to tune the system’s aggressiveness.

## How to Run Locally

1. **Clone the repository and navigate into it**:

   ```bash
   git clone https://github.com/Ilkin22/credit-card-fraud-system.git
   cd credit-card-fraud-system
   ```

2. **Create and activate a virtual environment (optional but recommended)**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (generates `models/model.pkl` and `models/scaler.pkl`):

   ```bash
   python src/train_model.py
   ```

   If you have the real credit card transactions dataset, place it at
   `data/creditcard.csv` before running the training script.  Otherwise a synthetic
   dataset will be generated automatically.

5. **Start the Flask application**:

   ```bash
   python app.py
   ```

6. **Use the dashboard**:

   - Open `http://localhost:5000` in your browser.
   - Fill in values for `V1`–`V28` and `Amount`.
   - Click **Analyze Transaction** to receive the fraud probability, risk level, and recommended action.
   - Review the last ten logged decisions in the table below the result card.

## API Examples

### `/predict` – POST

Send a JSON payload with keys matching the `FEATURES` list.  Example:

```json
{
  "V1": -1.23,
  "V2": 0.45,
  "V3": 1.02,
  "V4": -0.67,
  "V5": 0.11,
  "V6": 0.00,
  // ...
  "V28": -0.34,
  "Amount": 123.45
}
```

**Response:**

```json
{
  "predicted_class": 1,
  "fraud_probability": 0.86,
  "risk_level": "Critical",
  "action": "Block Transaction"
}
```

### `/recent` – GET

Returns an array of the last ten logged transactions.  Each entry contains the timestamp, amount, fraud probability, risk level, and action.

## MIS / Business Perspective

In a real‑world banking environment, this system would act as a **decision support component** within a larger approval workflow.  Risk teams can adjust the thresholds in `src/utils.py` to tune sensitivity, and the Random Forest model can be retrained periodically with fresh data.  By logging every prediction to a database, the system provides an audit trail for compliance and performance monitoring.  Over time the stored logs can be analysed to recalibrate thresholds, detect drift, and produce management reports.

---

> **Disclaimer:** This project is intended for educational purposes in a MIS portfolio.  It should not be used as-is in production without thorough validation, security hardening, and integration into a broader risk management framework.