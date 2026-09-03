#!/usr/bin/env python3
"""
retrain_real.py — Train and evaluate using real SITL flights only.

Uses the 13 real flights with heuristic labels, splits chronologically
into train/val/test. Acknowledges label noise as a limitation.

Usage:
    python Step_4_Detection/scripts/retrain_real.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

WINDOW_LEN = 30
STRIDE = 15
MIN_WINDOW_LABEL_RATIO = 0.5
ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = ROOT / "Step_5_Data" / "processed"
ARTIFACTS_DIR = ROOT / "Step_5_Data" / "artifacts"
MODELS_DIR = ROOT / "Step_5_Data" / "models"

FEATURE_COLS = [
    "x_m", "y_m", "alt_m", "rel_alt_m", "vel_m_s", "hdg_deg",
    "fix_type", "satellites_visible", "eph_m", "epv_m",
    "roll_deg", "pitch_deg", "yaw_deg",
    "rollspeed_radps", "pitchspeed_radps", "yawspeed_radps",
    "vibration_x", "vibration_y", "vibration_z",
    "clipping_0", "clipping_1", "clipping_2",
    "vel_ratio", "pos_horiz_ratio", "pos_vert_ratio",
    "vel_innov", "pos_horiz_innov", "pos_vert_innov",
    "battery_voltage", "battery_remaining_pct",
    "armed", "failsafe", "connection_ok", "is_stale_repeat",
]


def gps_to_local(df):
    df = df.copy()
    first_lat = df["lat_deg"].iloc[0]
    first_lon = df["lon_deg"].iloc[0]
    EARTH_RADIUS_M = 6_371_000.0
    df["x_m"] = np.radians(df["lat_deg"] - first_lat) * EARTH_RADIUS_M
    df["y_m"] = (
        np.radians(df["lon_deg"] - first_lon)
        * EARTH_RADIUS_M
        * np.cos(np.radians(first_lat))
    )
    return df


def load_real_flights():
    """Load ALL real flights, return per-flight (X, y, name)."""
    flights = []

    for csv_path in sorted(PROCESSED_DIR.glob("*_cleaned.csv")):
        # Skip synthetic files
        if "flight_" in csv_path.name and any(
            t in csv_path.name for t in ["flat_", "mountain_", "sea_"]
        ):
            continue
        if csv_path.stat().st_size < 1000:
            continue

        df = pd.read_csv(csv_path)
        if len(df) < WINDOW_LEN:
            continue

        # Get labels
        labels = None

        # Method 1: label column in CSV
        if "label" in df.columns:
            labels = df["label"].values.astype(int)

        # Method 2: *_labels.csv
        if labels is None:
            labels_csv = csv_path.parent / csv_path.name.replace("_cleaned.csv", "_labels.csv")
            if labels_csv.exists():
                ldf = pd.read_csv(labels_csv)
                if "label" in ldf.columns and len(ldf) == len(df):
                    labels = ldf["label"].values.astype(int)

        # Method 3: auto_segments.json
        if labels is None:
            segs_csv = csv_path.parent / csv_path.name.replace("_cleaned.csv", "_auto_segments.json")
            if segs_csv.exists():
                with open(segs_csv) as f:
                    segs = json.load(f)
                labels = np.zeros(len(df), dtype=int)
                for seg in segs:
                    mask = (df["time_s"] >= seg["start_s"]) & (df["time_s"] < seg["end_s"])
                    labels[mask] = seg["label"]

        if labels is None:
            print(f"  [SKIP] {csv_path.name}: no labels")
            continue

        # Convert GPS -> local
        if "lat_deg" in df.columns and "lon_deg" in df.columns:
            df = gps_to_local(df)

        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Build windows
        X, y = [], []
        for start in range(0, len(df) - WINDOW_LEN + 1, STRIDE):
            end = start + WINDOW_LEN
            win = df.iloc[start:end][FEATURE_COLS].values.astype(np.float32)
            win_y = labels[start:end]
            counts = np.bincount(win_y.astype(int), minlength=2)
            label = 1 if counts[1] / WINDOW_LEN >= MIN_WINDOW_LABEL_RATIO else 0
            X.append(win)
            y.append(label)

        if len(X) > 0:
            X_arr = np.array(X)
            y_arr = np.array(y)
            n_spoof = int(y_arr.sum())
            flights.append((X_arr, y_arr, csv_path.name))
            print(f"  {csv_path.name}: {len(X_arr)} windows (normal={len(X_arr)-n_spoof}, spoof={n_spoof})")

    return flights


def main():
    print("=" * 60)
    print("RETRAIN ON REAL FLIGHTS ONLY")
    print("=" * 60)

    print("\n[1/4] Loading real flights...")
    flights = load_real_flights()

    if not flights:
        print("ERROR: No flights loaded")
        return

    # Sort by name for reproducibility
    flights.sort(key=lambda f: f[2])

    total_windows = sum(len(f[0]) for f in flights)
    total_spoof = sum(int(f[1].sum()) for f in flights)
    print(f"\n  Total: {total_windows} windows ({total_windows - total_spoof} normal, {total_spoof} spoof)")

    # Chronological split: first 60% train, next 20% val, last 20% test
    # (splitting windows, not flights, to keep temporal order)
    all_X = np.concatenate([f[0] for f in flights])
    all_y = np.concatenate([f[1] for f in flights])

    n = len(all_X)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)

    X_train, y_train = all_X[:n_train], all_y[:n_train]
    X_val, y_val = all_X[n_train:n_train+n_val], all_y[n_train:n_train+n_val]
    X_test, y_test = all_X[n_train+n_val:], all_y[n_train+n_val:]

    print(f"\n  Train: {len(X_train)} (normal={int((y_train==0).sum())}, spoof={int((y_train==1).sum())})")
    print(f"  Val:   {len(X_val)} (normal={int((y_val==0).sum())}, spoof={int((y_val==1).sum())})")
    print(f"  Test:  {len(X_test)} (normal={int((y_test==0).sum())}, spoof={int((y_test==1).sum())})")

    # Scale
    print("\n[2/4] Training Random Forest...")
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    scaler.fit(X_train_flat)

    X_train_sc = scaler.transform(X_train_flat)
    X_val_sc = scaler.transform(X_val_flat)
    X_test_sc = scaler.transform(X_test_flat)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_train_sc, y_train)

    # Results
    print("\n  === TRAIN ===")
    print(classification_report(y_train, rf.predict(X_train_sc), zero_division=0))

    print("  === VAL ===")
    val_pred = rf.predict(X_val_sc)
    print(classification_report(y_val, val_pred, zero_division=0))

    print("[3/4] === TEST ===")
    test_pred = rf.predict(X_test_sc)
    print(classification_report(y_test, test_pred, zero_division=0))

    cm = confusion_matrix(y_test, test_pred)
    print(f"  Confusion Matrix:\n{cm}")

    # Feature importance
    print("\n[4/4] Feature importance (top 10):")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    n_features = min(10, len(indices), len(FEATURE_COLS))
    for i in range(n_features):
        idx = indices[i]
        if idx < len(FEATURE_COLS):
            print(f"  {FEATURE_COLS[idx]}: {importances[idx]:.4f}")
        else:
            print(f"  feature_{idx}: {importances[idx]:.4f}")

    spoof_f1 = f1_score(y_test, test_pred, pos_label=1, zero_division=0)
    normal_f1 = f1_score(y_test, test_pred, pos_label=0, zero_division=0)
    print(f"\n  === SUMMARY ===")
    print(f"  Normal F1: {normal_f1:.3f}")
    print(f"  Spoofed F1: {spoof_f1:.3f}")
    print(f"  Overall accuracy: {(test_pred == y_test).mean():.3f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "rf_model_real.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODELS_DIR / "scaler_real.pkl", "wb") as f:
        pickle.dump(scaler, f)

    info = {
        "approach": "real_flights_only_chronological_split",
        "total_windows": int(n),
        "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)),
        "test_windows": int(len(X_test)),
        "train_dist": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "val_dist": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
        "test_dist": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "test_normal_f1": float(normal_f1),
        "test_spoofed_f1": float(spoof_f1),
        "test_accuracy": float((test_pred == y_test).mean()),
        "note": "Labels from heuristic rules (H1-H8). Circular training/testing acknowledged as limitation.",
    }
    with open(ARTIFACTS_DIR / "real_retrain_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Model saved: {MODELS_DIR / 'rf_model_real.pkl'}")
    print(f"  Info saved: {ARTIFACTS_DIR / 'real_retrain_info.json'}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
