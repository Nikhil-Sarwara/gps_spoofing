from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_FEATURE_COLS = [
    "x_m",
    "y_m",
    "alt_m",
    "rel_alt_m",
    "vel_m_s",
    "hdg_deg",
    "fix_type",
    "satellites_visible",
    "eph_m",
    "epv_m",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "rollspeed_radps",
    "pitchspeed_radps",
    "yawspeed_radps",
    "vibration_x",
    "vibration_y",
    "vibration_z",
    "clipping_0",
    "clipping_1",
    "clipping_2",
    "vel_ratio",
    "pos_horiz_ratio",
    "pos_vert_ratio",
    "vel_innov",
    "pos_horiz_innov",
    "pos_vert_innov",
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


def main(
    input_path: Union[str, Path],
    labels_path: Optional[Union[str, Path]],
    artifacts_dir: Union[str, Path],
    windows_config: Optional[dict] = None,
    split_config: Optional[dict] = None,
) -> dict:
    """
    Create ML-ready sliding windows from labeled data (Single file or Directory).
    """
    windows_config = windows_config or {}
    split_config = split_config or {}

    window_len = windows_config.get("length", 30)
    stride = windows_config.get("stride", 15)
    train_ratio = split_config.get("train_ratio", 0.7)
    val_ratio = split_config.get("val_ratio", 0.15)

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(input_path)
    all_dfs = []

    if input_path.is_dir():
        print(f"Loading all cleaned files from {input_path}...")
        # Recursively find all cleaned files
        cleaned_files = sorted(input_path.glob("**/*_cleaned.csv"))
        for cf in cleaned_files:
            df_clean = pd.read_csv(cf)
            # Find the matching unique labels file
            lp = cf.parent / cf.name.replace("_cleaned.csv", "_labels.csv")
            if not lp.exists():
                # Fallback to old name just in case
                lp = cf.parent / "row_labels_auto.csv"
            
            if not lp.exists():
                print(f"  [SKIP] No labels found for {cf.name}")
                continue
            
            labels_df = pd.read_csv(lp)
            if "label" not in labels_df.columns:
                print(f"  [SKIP] 'label' column missing in {lp.name}. Columns: {list(labels_df.columns)}")
                continue
                
            # Perform merge
            df = pd.merge(df_clean, labels_df[["time_s", "label"]], on="time_s", how="left", suffixes=('', '_y'))
            
            # If 'label' ended up as 'label_y' because it existed in df_clean
            if "label_y" in df.columns:
                df["label"] = df["label_y"]
                df = df.drop(columns=["label_y"])
                
            if "label" not in df.columns:
                print(f"  [ERROR] Merge failed to produce 'label' for {cf.name}")
                continue

            df["label"] = df["label"].fillna(0).astype(int)
            all_dfs.append(df)
            print(f"  - Loaded {cf.name}: {len(df)} rows, {df['label'].sum()} labels")
    else:
        print(f"Loading single file {input_path}...")
        df_clean = pd.read_csv(input_path)
        labels_df = pd.read_csv(labels_path)
        df = pd.merge(df_clean, labels_df[["time_s", "label"]], on="time_s", how="left")
        df["label"] = df["label"].fillna(0).astype(int)
        all_dfs.append(df)

    if not all_dfs:
        print("ERROR: No data loaded.")
        return {}

    feature_cols = BASE_FEATURE_COLS
    X_list, y_list, start_list, end_list = [], [], [], []
    
    for df in all_dfs:
        if len(df) < window_len:
            print(f"  [SKIP] Too few rows ({len(df)}) for window length {window_len}")
            continue
            
        print(f"Processing {len(df)} rows for windows...")
        df = gps_to_local(df)
        
        # Fill NaN
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        X_f, y_f, s_f, e_f = create_windows(
            df,
            feature_cols=feature_cols,
            window_len=window_len,
            stride=stride,
        )
        if len(X_f) > 0:
            X_list.append(X_f)
            y_list.append(y_f)
            start_list.append(s_f)
            end_list.append(e_f)

    if not X_list:
        print("ERROR: No windows created.")
        return {}

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    win_start = np.concatenate(start_list, axis=0)
    win_end = np.concatenate(end_list, axis=0)

    print(f"\nAGGREGATED DATASET:")
    print(f"Total windows: {X.shape}")
    dist = safe_label_dist(y)
    print(f"Label dist: {dist}")

    n_total = len(X)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
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
        artifacts_dir / "dataset.npz",
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

    with open(artifacts_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(artifacts_dir / "feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    info = {
        "window_len": int(window_len),
        "stride": int(stride),
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

    with open(artifacts_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n✅ Dataset ready")
    print(f"Train: {X_train_sc.shape}, labels={safe_label_dist(y_train)}")
    print(f"Val:   {X_val_sc.shape}, labels={safe_label_dist(y_val)}")
    print(f"Test:  {X_test_sc.shape}, labels={safe_label_dist(y_test)}")

    result = {
        "dataset_npz": str(artifacts_dir / "dataset.npz"),
        "scaler_pkl": str(artifacts_dir / "scaler.pkl"),
        "feature_names_json": str(artifacts_dir / "feature_names.json"),
        "dataset_info_json": str(artifacts_dir / "dataset_info.json"),
        "info": info,
    }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("labels_path", help="Path to row_labels CSV")
    parser.add_argument("--cleaned-path", default="Step_5_Data/processed", help="Path to cleaned data CSV or directory")
    parser.add_argument("--artifacts-dir", default="Step_5_Data/artifacts", help="Output directory")
    args = parser.parse_args()
    main(args.cleaned_path, args.labels_path, args.artifacts_dir)
