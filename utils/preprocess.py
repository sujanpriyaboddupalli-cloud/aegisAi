import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

RAW_FILE = "data/smd_sample.csv"
OUTPUT_FILE = "data/processed.csv"
SCALER_FILE = "models/scaler.pkl"
FEATURES_FILE = "models/feature_columns.pkl"

FEATURE_COLUMNS = ["cpu", "temp", "power", "network"]


def clean_column_names(df):
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    )
    return df


def load_data(path=RAW_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    df = clean_column_names(df)
    return df


def clean_data(df):
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    missing_required = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    return df


def add_time_features(df):
    if "timestamp" in df.columns and df["timestamp"].notna().sum() > 0:
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["dayofweek"] = df["timestamp"].dt.dayofweek
    return df


def add_features(df):
    df = add_time_features(df)

    for col in FEATURE_COLUMNS:
        df[f"{col}_roll_mean"] = df[col].rolling(window=5, min_periods=1).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window=5, min_periods=1).std().fillna(0)
        df[f"{col}_diff"] = df[col].diff().fillna(0)

    return df


def get_feature_columns(df):
    feature_cols = FEATURE_COLUMNS + \
        [f"{c}_roll_mean" for c in FEATURE_COLUMNS] + \
        [f"{c}_roll_std" for c in FEATURE_COLUMNS] + \
        [f"{c}_diff" for c in FEATURE_COLUMNS]

    for col in ["hour", "day", "month", "dayofweek"]:
        if col in df.columns:
            feature_cols.append(col)

    return feature_cols


def scale_features(df):
    feature_cols = get_feature_columns(df)

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(feature_cols, FEATURES_FILE)

    return df_scaled, scaler, feature_cols


def preprocess_data(input_file=RAW_FILE, output_file=OUTPUT_FILE):
    df = load_data(input_file)
    print("Loaded shape:", df.shape)

    df = clean_data(df)
    df = add_features(df)
    df_scaled, scaler, feature_cols = scale_features(df)

    os.makedirs("data", exist_ok=True)
    df_scaled.to_csv(output_file, index=False)

    print("Processed shape:", df_scaled.shape)
    print("Saved processed file:", output_file)
    print("Saved scaler:", SCALER_FILE)
    print("Saved features:", FEATURES_FILE)

    return df_scaled, scaler, feature_cols


if __name__ == "__main__":
    preprocess_data()