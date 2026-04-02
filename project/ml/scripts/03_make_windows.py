import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CLEAN_PATH = Path("gps_logs/processed/gps_log_20260402_184048_cleaned.csv")
LABELS_PATH = Path("gps_logs/processed/row_labels.csv")

ARTIFACTS_DIR = Path("ml/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LEN = 30
STRIDE = 15

BASE_FEATURE_COLS = [
    "lat_deg",
    "lon_deg",
    "alt_m",
    "rel_alt_m",
    "vel_m_s",
    "hdg_deg",
    "fix_type",
    "satellites_visible",
    "eph_m",
    "epv_m",
    "battery_voltage",
    "battery_remaining_pct",
    "armed",
    "failsafe",
    "connection_ok",
    "is_stale_repeat",
]


def gps_to_local(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    first_lat = df["lat_deg"].iloc[0]
    first_lon = df["lon_deg"].iloc[0]

    earth_radius_m = 6371000.0

    df["x_m"] = np.radians(df["lat_deg"] - first_lat) * earth_radius_m
    df["y_m"] = (
        np.radians(df["lon_deg"] - first_lon)
        * earth_radius_m
        * np.cos(np.radians(first_lat))
    )

    return df


def create_windows(df: pd.DataFrame, feature_cols, window_len: int, stride: int):
    windows = []
    labels = []
    window_start_times = []
    window_end_times = []

    for start in range(0, len(df) - window_len + 1, stride):
        end = start + window_len
        window_df = df.iloc[start:end]

        x_window = window_df[feature_cols].values.astype(np.float32)

        row_labels = window_df["label"].values.astype(int)
        counts = np.bincount(row_labels)
        anomaly_ratio = counts[1] / len(row_labels) if len(counts) > 1 else 0.0
        y_window = int(anomaly_ratio > 0.5)

        windows.append(x_window)
        labels.append(y_window)
        window_start_times.append(float(window_df["time_s"].iloc[0]))
        window_end_times.append(float(window_df["time_s"].iloc[-1]))

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(window_start_times, dtype=np.float32),
        np.array(window_end_times, dtype=np.float32),
    )


def safe_label_dist(y):
    values, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(values, counts)}


def main():
    print("Loading data...")

    df_clean = pd.read_csv(CLEAN_PATH)
    labels_df = pd.read_csv(LABELS_PATH)

    df = pd.merge(
        df_clean,
        labels_df[["time_s", "label"]],
        on="time_s",
        how="left",
    )

    df["label"] = df["label"].fillna(0).astype(int)

    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Labeled rows: {len(df)}, anomaly ratio: {df['label'].mean():.1%}")

    if int(df["label"].sum()) == 0:
        print("ERROR: No anomaly labels found. Check row_labels.csv or segments.json.")
        return

    print("Converting GPS...")
    df = gps_to_local(df)

    feature_cols = BASE_FEATURE_COLS.copy()
    feature_cols[0] = "x_m"
    feature_cols[1] = "y_m"

    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")

    df["is_stale_repeat"] = df["is_stale_repeat"].astype(int)

    usable_cols = feature_cols + ["time_s", "label"]
    df_labeled = df[usable_cols].copy()

    df_labeled = df_labeled.replace([np.inf, -np.inf], np.nan)
    df_labeled = df_labeled.dropna().reset_index(drop=True)

    print(f"After trim: {len(df_labeled)} rows")

    print("Creating windows...")
    X, y, win_start, win_end = create_windows(
        df_labeled,
        feature_cols=feature_cols,
        window_len=WINDOW_LEN,
        stride=STRIDE,
    )

    print(f"Windows created: {X.shape}")
    print(f"Label dist: {np.bincount(y)}")

    if len(X) == 0:
        print("ERROR: No windows created. Reduce window size or check data.")
        return

    n_total = len(X)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_end = n_train
    val_end = n_train + n_val

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    win_start_train, win_end_train = win_start[:train_end], win_end[:train_end]
    win_start_val, win_end_val = win_start[train_end:val_end], win_end[train_end:val_end]
    win_start_test, win_end_test = win_start[val_end:], win_end[val_end:]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("ERROR: One of the splits is empty. Adjust split ratio or collect more data.")
        return

    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)

    X_train_sc = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_sc = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_sc = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    np.savez_compressed(
        ARTIFACTS_DIR / "dataset.npz",
        X_train=X_train_sc,
        y_train=y_train,
        X_val=X_val_sc,
        y_val=y_val,
        X_test=X_test_sc,
        y_test=y_test,
        win_start_train=win_start_train,
        win_end_train=win_end_train,
        win_start_val=win_start_val,
        win_end_val=win_end_val,
        win_start_test=win_start_test,
        win_end_test=win_end_test,
    )

    with open(ARTIFACTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    info = {
        "window_len": int(WINDOW_LEN),
        "stride": int(STRIDE),
        "num_features": int(len(feature_cols)),
        "feature_names": feature_cols,
        "n_total_windows": int(n_total),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "train_time_range_s": [
            float(win_start_train[0]),
            float(win_end_train[-1]),
        ],
        "val_time_range_s": [
            float(win_start_val[0]),
            float(win_end_val[-1]),
        ],
        "test_time_range_s": [
            float(win_start_test[0]),
            float(win_end_test[-1]),
        ],
        "label_dist": {
            "train": safe_label_dist(y_train),
            "val": safe_label_dist(y_val),
            "test": safe_label_dist(y_test),
            "all": safe_label_dist(y),
        },
    }

    with open(ARTIFACTS_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n✅ Dataset ready in ml/artifacts/")
    print(f"Train: {X_train_sc.shape}, labels={safe_label_dist(y_train)}")
    print(f"Val:   {X_val_sc.shape}, labels={safe_label_dist(y_val)}")
    print(f"Test:  {X_test_sc.shape}, labels={safe_label_dist(y_test)}")
    print("Saved:")
    print("- ml/artifacts/dataset.npz")
    print("- ml/artifacts/scaler.pkl")
    print("- ml/artifacts/feature_names.json")
    print("- ml/artifacts/dataset_info.json")


if __name__ == "__main__":
    main()
