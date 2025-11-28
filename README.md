# Intelligent Credit Card Fraud Detection System (MIS Portfolio)

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