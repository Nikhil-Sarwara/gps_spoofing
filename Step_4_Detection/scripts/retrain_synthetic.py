#!/usr/bin/env python3
"""
retrain_synthetic.py — Train and test entirely on synthetic data.

Strategy:
  - 60 synthetic flights with clean ground truth
  - Chronological train/val/test split (70/15/15)
  - Proper per-class metrics
  - No domain gap, no label noise

Usage:
    python Step_4_Detection/scripts/retrain_synthetic.py
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
SYNTHETIC_DIR = ROOT / "Step_5_Data" / "synthetic_v2"
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


def build_windows(df, labels):
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X, y = [], []
    for start in range(0, len(df) - WINDOW_LEN + 1, STRIDE):
        end = start + WINDOW_LEN
        win = df.iloc[start:end][FEATURE_COLS].values.astype(np.float32)
        win_y = labels[start:end]
        counts = np.bincount(win_y.astype(int), minlength=2)
        label = 1 if counts[1] / WINDOW_LEN >= MIN_WINDOW_LABEL_RATIO else 0
        X.append(win)
        y.append(label)
    return np.array(X), np.array(y)


def main():
    print("=" * 60)
    print("SYNTHETIC-ONLY TRAINING (clean ground truth)")
    print("=" * 60)

    # Load all 60 synthetic flights
    print("\n[1/4] Loading synthetic flights...")
    all_X, all_y = [], []
    per_flight = []

    for csv_path in sorted(SYNTHETIC_DIR.glob("*_cleaned.csv")):
        df = pd.read_csv(csv_path)
        if len(df) < WINDOW_LEN or "label" not in df.columns:
            continue
        labels = df["label"].values.astype(int)
        df = gps_to_local(df)
        X, y = build_windows(df, labels)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            per_flight.append((csv_path.name, len(X), int((y==0).sum()), int((y==1).sum())))
            n_spoof = int(y.sum())
            print(f"  {csv_path.name}: {len(X)} windows (normal={len(X)-n_spoof}, spoof={n_spoof})")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    print(f"\n  Total: {len(X)} windows (normal={int((y==0).sum())}, spoof={int((y==1).sum())})")

    # Chronological split: 70/15/15
    print("\n[2/4] Splitting (70/15/15 chronological)...")
    n = len(X)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    print(f"  Train: {len(X_train)} (normal={int((y_train==0).sum())}, spoof={int((y_train==1).sum())})")
    print(f"  Val:   {len(X_val)} (normal={int((y_val==0).sum())}, spoof={int((y_val==1).sum())})")
    print(f"  Test:  {len(X_test)} (normal={int((y_test==0).sum())}, spoof={int((y_test==1).sum())})")

    # Scale
    print("\n[3/4] Training Random Forest...")
    scaler = StandardScaler()
    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    scaler.fit(X_tr_flat)

    X_tr_sc = scaler.transform(X_tr_flat)
    X_val_sc = scaler.transform(X_val_flat)
    X_test_sc = scaler.transform(X_test_flat)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_tr_sc, y_train)

    print("\n  === TRAIN ===")
    print(classification_report(y_train, rf.predict(X_tr_sc), zero_division=0))

    print("  === VAL ===")
    val_pred = rf.predict(X_val_sc)
    print(classification_report(y_val, val_pred, zero_division=0))

    print("[4/4] === TEST ===")
    test_pred = rf.predict(X_test_sc)
    print(classification_report(y_test, test_pred, zero_division=0))

    cm = confusion_matrix(y_test, test_pred)
    print(f"  Confusion Matrix:\n{cm}")

    # Feature importance
    print("\n  === Feature importance (top 10) ===")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(indices))):
        idx_f = indices[i]
        print(f"  {FEATURE_COLS[idx_f % len(FEATURE_COLS)]}: {importances[idx_f]:.4f}")

    # Summary
    spoof_f1 = f1_score(y_test, test_pred, pos_label=1, zero_division=0)
    normal_f1 = f1_score(y_test, test_pred, pos_label=0, zero_division=0)
    print(f"\n  === FINAL METRICS (for paper) ===")
    print(f"  Overall accuracy: {(test_pred == y_test).mean():.4f}")
    print(f"  Normal — Precision: {cm[0][0]/(cm[0][0]+cm[1][0]):.4f}, Recall: {cm[0][0]/(cm[0][0]+cm[0][1]):.4f}, F1: {normal_f1:.4f}")
    print(f"  Spoofed — Precision: {cm[1][1]/(cm[1][1]+cm[0][1]):.4f}, Recall: {cm[1][1]/(cm[1][1]+cm[1][0]):.4f}, F1: {spoof_f1:.4f}")
    print(f"  Macro F1: {(normal_f1 + spoof_f1) / 2:.4f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "rf_model_synthetic.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODELS_DIR / "scaler_synthetic.pkl", "wb") as f:
        pickle.dump(scaler, f)

    info = {
        "approach": "synthetic_only",
        "total_flights": len(per_flight),
        "total_windows": int(n),
        "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)),
        "test_windows": int(len(X_test)),
        "train_dist": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "val_dist": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
        "test_dist": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "test_accuracy": float((test_pred == y_test).mean()),
        "test_normal_f1": float(normal_f1),
        "test_spoofed_f1": float(spoof_f1),
        "test_confusion_matrix": cm.tolist(),
        "per_flight": [{"name": n, "windows": w, "normal": no, "spoof": s} for n, w, no, s in per_flight],
    }
    with open(ARTIFACTS_DIR / "synthetic_retrain_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Model: {MODELS_DIR / 'rf_model_synthetic.pkl'}")
    print(f"  Info: {ARTIFACTS_DIR / 'synthetic_retrain_info.json'}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
