"""
utils/train.py
──────────────────────────────────────────────────────────────────────────────
Model training for AegisAI.

After convert.py:
  cpu, power, network, mem  → [0, 100]  (percent)
  temp                      → [30, 100] (degrees C)
  everything else           → [0, 1]    (raw SMD normalised)

Failure label thresholds are set accordingly.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier


def train_anomaly_model(X: pd.DataFrame) -> IsolationForest:
    model = IsolationForest(
        contamination=0.08,
        n_estimators=200,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    return model


def create_failure_labels(df: pd.DataFrame) -> list:
    """
    Rule-based labelling using the rescaled column values produced by convert.py.
    cpu / power / network → 0-100 (%)
    temp                  → 30-100 (°C)
    """
    labels = []
    for _, row in df.iterrows():
        cpu     = float(row.get("cpu",     0))
        temp    = float(row.get("temp",    0))
        power   = float(row.get("power",   0))
        network = float(row.get("network", 0))

        if temp > 80 and cpu > 85:
            labels.append("cooling_issue")
        elif power > 90:
            labels.append("power_failure")
        elif cpu > 90:
            labels.append("server_overload")
        elif network > 90:
            labels.append("network_saturation")
        else:
            labels.append("normal")
    return labels


def train_failure_model(X: pd.DataFrame, y: list) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def save_model(model, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def get_anomaly_scores(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """Normalised anomaly scores in [0, 1]. Higher = more anomalous."""
    raw        = model.decision_function(X)
    normalised = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    return normalised