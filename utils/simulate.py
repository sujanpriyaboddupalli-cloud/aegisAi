import pandas as pd
from typing import Tuple, Dict

FIX_DELTAS: Dict[str, Dict[str, float]] = {
    "cooling_issue":      {"temp": -12, "cpu": -15, "power": -5,  "network":  0},
    "power_failure":      {"temp":  -3, "cpu": -10, "power": -22, "network":  0},
    "server_overload":    {"temp":  -8, "cpu": -28, "power":  -8, "network": -5},
    "network_saturation": {"temp":  -2, "cpu": -12, "power":  -3, "network":-30},
    "normal":             {"temp":   0, "cpu":   0, "power":   0, "network":  0},
}

def simulate_fix(row: pd.Series, prediction: str) -> Tuple[pd.Series, Dict[str, float]]:
    new_row = row.copy()
    deltas  = FIX_DELTAS.get(prediction, FIX_DELTAS["normal"])
    for metric, delta in deltas.items():
        if metric in new_row.index:
            new_row[metric] = max(float(new_row[metric]) + delta, 0)
    return new_row, deltas

def compute_health_score(row: pd.Series, prediction: str, anomaly: int) -> int:
    score = 100
    if prediction != "normal":
        score -= 30
    if anomaly == 1:
        score -= 20
    if float(row.get("cpu",   0)) > 85: score -= 10
    if float(row.get("temp",  0)) > 75: score -= 10
    if float(row.get("power", 0)) > 85: score -= 10
    return max(0, score)