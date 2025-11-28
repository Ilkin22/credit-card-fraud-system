"""Utilities shared across the fraud detection system.

This module centralizes definitions for feature names and helper
functions used throughout the training script and the Flask app. Keeping
these values in one place reduces the risk of mismatches between
training and prediction.
"""

from pathlib import Path
from typing import Tuple


# Define the feature columns used both for training and prediction.
# Feature columns used for training and prediction. These follow the
# Kaggle credit card fraud dataset naming convention, including
# principal component analysis (PCA) components V1–V28 and the transaction
# `Amount`.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Define the name of the target column. The fraud class is denoted by
# `1` and the safe class by `0`.
TARGET = "Class"


def ensure_models_dir() -> Path:
    """Ensure the `models/` directory exists at the project root.

    Returns
    -------
    Path
        The absolute Path object pointing to the models directory.
    """
    models_path = Path(__file__).resolve().parents[2] / "models"
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
    model_path = models_dir / "model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    return model_path, scaler_path


def get_db_path() -> Path:
    """Return the absolute path to the SQLite transactions database.

    Ensures the `data/` directory exists.

    Returns
    -------
    Path
        Path to `data/transactions.db`.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / "transactions.db"


def connect_db() -> sqlite3.Connection:
    """Establish a connection to the SQLite database.

    Returns
    -------
    sqlite3.Connection
        A connection object to the database.
    """
    import sqlite3  # local import to avoid unconditional dependency
    db_path = get_db_path()
    return sqlite3.connect(db_path)


def log_transaction(timestamp: str, amount: float, probability: float, risk_level: str, action: str) -> None:
    """Insert a transaction record into the SQLite database.

    Parameters
    ----------
    timestamp : str
        ISO formatted timestamp of the prediction.
    amount : float
        Transaction amount.
    probability : float
        Fraud probability (0–1).
    risk_level : str
        Categorical risk level (Critical, Suspicious, Low, etc.).
    action : str
        Recommended action corresponding to the risk.
    """
    conn = connect_db()
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
    cur.execute(
        """
        INSERT INTO transactions (timestamp, amount, probability, risk_level, action)
        VALUES (?, ?, ?, ?, ?)
        """,
        (timestamp, amount, probability, risk_level, action),
    )
    conn.commit()
    conn.close()


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
        # Critical risk: transaction is very likely fraudulent
        return "Critical", "Block"
    elif probability > 0.50:
        # Suspicious transaction: recommend verifying via SMS or other method
        return "Suspicious", "Verify (SMS)"
    else:
        # Safe transaction
        return "Safe", "Allow"