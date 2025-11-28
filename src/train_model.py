"""Training script for the intelligent credit card fraud detection system.

This module handles data loading (either from a real credit card dataset
or synthetic generation), preprocessing, balancing via SMOTE, model
training using a Random Forest classifier, evaluation, and saving of
artifacts. Run this script to (re)train the model before serving
predictions through the Flask app.
"""

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
    csv_path = data_dir / "creditcard.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in dataset: {missing_cols}")
        return df[FEATURES + [TARGET]].copy()
    else:
        print("creditcard.csv not found. Generating synthetic dataset for testing...")
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
    # Determine the project root and data directory relative to this file
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate the dataset
    df = load_dataset(data_dir)
    X = df[FEATURES].values
    y = df[TARGET].values

    # Train/test split with stratification on target
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Address class imbalance using SMOTE on the training data only
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Train the Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train_resampled)

    # Evaluate on test set
    y_pred = clf.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=["Safe", "Fraud"])
    print("Classification Report:\n", report)

    # Save model and scaler
    models_dir = ensure_models_dir()
    model_path, scaler_path = get_model_and_scaler_paths()
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    return clf, scaler


if __name__ == "__main__":
    train()
