#!/usr/bin/env python3
"""
retrain_hybrid.py — Train on synthetic spoofed + real normal flights.

Strategy:
  - Spoofed class: synthetic data (clean ground truth)
  - Normal class: real flights with NO spoof labels (clean normal data)
  - Test on all real flights (including those with spoof labels)
  - This bridges the domain gap while keeping clean training labels

Usage:
    python Step_4_Detection/scripts/retrain_hybrid.py
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
SYNTHETIC_DIR = ROOT / "Step_5_Data" / "synthetic"
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


def build_windows(df, labels):
    """Build sliding windows from a dataframe + row labels."""
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
    return np.array(X) if X else np.empty((0, WINDOW_LEN, len(FEATURE_COLS))), np.array(y) if y else np.empty((0,), dtype=int)


def load_synthetic_spoofed():
    """Load ALL synthetic flights — these have clean spoof/normal labels."""
    X_all, y_all = [], []
    for csv_path in sorted(SYNTHETIC_DIR.glob("*_cleaned.csv")):
        df = pd.read_csv(csv_path)
        if len(df) < WINDOW_LEN or "label" not in df.columns:
            continue
        labels = df["label"].values.astype(int)
        df = gps_to_local(df)
        X, y = build_windows(df, labels)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
    if not X_all:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(X_all), np.concatenate(y_all)


def load_real_normal_only():
    """Load real flights that have ZERO spoof labels — pure normal data."""
    X_all, y_all = [], []
    files = []
    for csv_path in sorted(PROCESSED_DIR.glob("*_cleaned.csv")):
        # Skip synthetic
        if "flight_" in csv_path.name and any(t in csv_path.name for t in ["flat_", "mountain_", "sea_"]):
            continue
        if csv_path.stat().st_size < 1000:
            continue

        df = pd.read_csv(csv_path)
        if len(df) < WINDOW_LEN:
            continue

        # Get labels
        labels = None
        if "label" in df.columns:
            labels = df["label"].values.astype(int)
        if labels is None:
            labels_csv = csv_path.parent / csv_path.name.replace("_cleaned.csv", "_labels.csv")
            if labels_csv.exists():
                ldf = pd.read_csv(labels_csv)
                if "label" in ldf.columns and len(ldf) == len(df):
                    labels = ldf["label"].values.astype(int)
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
            continue

        # ONLY include if this flight has ZERO spoof labels
        if labels.sum() == 0:
            if "lat_deg" in df.columns and "lon_deg" in df.columns:
                df = gps_to_local(df)
            X, y = build_windows(df, labels)
            if len(X) > 0:
                X_all.append(X)
                y_all.append(y)
                files.append(csv_path.name)
                print(f"  [NORMAL] {csv_path.name}: {len(X)} windows (all normal)")

    if not X_all:
        return np.empty((0,)), np.empty((0,)), []
    return np.concatenate(X_all), np.concatenate(y_all), files


def load_real_all_for_test():
    """Load ALL real flights for testing (including those with spoof labels)."""
    X_all, y_all = [], []
    files = []
    per_file_windows = []

    for csv_path in sorted(PROCESSED_DIR.glob("*_cleaned.csv")):
        if "flight_" in csv_path.name and any(t in csv_path.name for t in ["flat_", "mountain_", "sea_"]):
            continue
        if csv_path.stat().st_size < 1000:
            continue

        df = pd.read_csv(csv_path)
        if len(df) < WINDOW_LEN:
            continue

        labels = None
        if "label" in df.columns:
            labels = df["label"].values.astype(int)
        if labels is None:
            labels_csv = csv_path.parent / csv_path.name.replace("_cleaned.csv", "_labels.csv")
            if labels_csv.exists():
                ldf = pd.read_csv(labels_csv)
                if "label" in ldf.columns and len(ldf) == len(df):
                    labels = ldf["label"].values.astype(int)
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
            continue

        if "lat_deg" in df.columns and "lon_deg" in df.columns:
            df = gps_to_local(df)

        X, y = build_windows(df, labels)
        if len(X) > 0:
            X_all.append(X)
            y_all.append(y)
            files.append(csv_path.name)
            per_file_windows.append(len(X))
            n_spoof = int(y.sum())
            print(f"  [TEST] {csv_path.name}: {len(X)} windows (normal={len(X)-n_spoof}, spoof={n_spoof})")

    if not X_all:
        return np.empty((0,)), np.empty((0,)), [], []
    return np.concatenate(X_all), np.concatenate(y_all), files, per_file_windows


def main():
    print("=" * 60)
    print("HYBRID TRAINING: Synthetic Spoofed + Real Normal")
    print("=" * 60)

    # --- Load synthetic (spoofed class) ---
    print("\n[1/5] Loading synthetic data (spoofed class, clean labels)...")
    X_synth, y_synth = load_synthetic_spoofed()
    n_synth_spoof = int((y_synth == 1).sum())
    n_synth_normal = int((y_synth == 0).sum())
    print(f"  Synthetic: {len(X_synth)} windows (normal={n_synth_normal}, spoof={n_synth_spoof})")

    # --- Load real normal flights (normal class) ---
    print("\n[2/5] Loading real flights with ZERO spoof labels (normal class)...")
    X_real_norm, y_real_norm, real_norm_files = load_real_normal_only()
    print(f"  Real normal: {len(X_real_norm)} windows from {len(real_norm_files)} flights")

    # --- Combine for training ---
    print("\n[3/5] Building hybrid training set...")
    X_train = np.concatenate([X_synth, X_real_norm])
    y_train = np.concatenate([y_synth, y_real_norm])

    # Shuffle
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    # Split: 80% train, 20% val
    n = len(X_train)
    n_tr = int(0.8 * n)
    X_tr, y_tr = X_train[:n_tr], y_train[:n_tr]
    X_val, y_val = X_train[n_tr:], y_train[n_tr:]

    print(f"  Train: {len(X_tr)} (normal={int((y_tr==0).sum())}, spoof={int((y_tr==1).sum())})")
    print(f"  Val:   {len(X_val)} (normal={int((y_val==0).sum())}, spoof={int((y_val==1).sum())})")

    # --- Load ALL real flights for testing ---
    print("\n[4/5] Loading ALL real flights for testing...")
    X_test, y_test, test_files, per_file_windows = load_real_all_for_test()
    print(f"  Test: {len(X_test)} windows (normal={int((y_test==0).sum())}, spoof={int((y_test==1).sum())})")

    # --- Train ---
    print("\n[5/5] Training Random Forest...")
    scaler = StandardScaler()
    X_tr_flat = X_tr.reshape(len(X_tr), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    scaler.fit(X_tr_flat)

    X_tr_sc = scaler.transform(X_tr_flat)
    X_val_sc = scaler.transform(X_val_flat)
    X_test_sc = scaler.transform(X_test_flat)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    rf.fit(X_tr_sc, y_tr)

    # Results
    print("\n  === TRAIN ===")
    print(classification_report(y_tr, rf.predict(X_tr_sc), zero_division=0))

    print("  === VAL ===")
    print(classification_report(y_val, rf.predict(X_val_sc), zero_division=0))

    print("  === TEST (all real flights) ===")
    test_pred = rf.predict(X_test_sc)
    print(classification_report(y_test, test_pred, zero_division=0))

    cm = confusion_matrix(y_test, test_pred)
    print(f"  Confusion Matrix:\n{cm}")

    # Per-file breakdown
    print("\n  === PER-FILE BREAKDOWN ===")
    idx = 0
    for fname, n_w in zip(test_files, per_file_windows):
        file_pred = test_pred[idx:idx+n_w]
        file_true = y_test[idx:idx+n_w]
        n_correct = (file_pred == file_true).sum()
        n_spoof_gt = int(file_true.sum())
        n_spoof_det = int(((file_pred == 1) & (file_true == 1)).sum())
        n_fp = int(((file_pred == 1) & (file_true == 0)).sum())
        print(f"  {fname}: {n_w} windows, {n_correct}/{n_w} correct, "
              f"spoof_gt={n_spoof_gt}, detected={n_spoof_det}, FP={n_fp}")
        idx += n_w

    # Feature importance
    print("\n  === Feature importance (top 10) ===")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(10, len(FEATURE_COLS))):
        idx_f = indices[i]
        print(f"  {FEATURE_COLS[idx_f]}: {importances[idx_f]:.4f}")

    spoof_f1 = f1_score(y_test, test_pred, pos_label=1, zero_division=0)
    normal_f1 = f1_score(y_test, test_pred, pos_label=0, zero_division=0)
    print(f"\n  === SUMMARY ===")
    print(f"  Normal F1: {normal_f1:.3f}")
    print(f"  Spoofed F1: {spoof_f1:.3f}")
    print(f"  Overall accuracy: {(test_pred == y_test).mean():.3f}")

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "rf_model_hybrid.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODELS_DIR / "scaler_hybrid.pkl", "wb") as f:
        pickle.dump(scaler, f)

    info = {
        "approach": "hybrid_spoof_synthetic_normal_real",
        "synthetic_windows": int(len(X_synth)),
        "real_normal_windows": int(len(X_real_norm)),
        "train_windows": int(len(X_tr)),
        "val_windows": int(len(X_val)),
        "test_windows": int(len(X_test)),
        "train_dist": {str(k): int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))},
        "test_dist": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        "test_normal_f1": float(normal_f1),
        "test_spoofed_f1": float(spoof_f1),
        "test_accuracy": float((test_pred == y_test).mean()),
        "real_normal_files": real_norm_files,
        "test_files": test_files,
        "note": "Spoofed class from synthetic ground truth. Normal class from real flights with zero spoof labels. Test on all real flights.",
    }
    with open(ARTIFACTS_DIR / "hybrid_retrain_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Model: {MODELS_DIR / 'rf_model_hybrid.pkl'}")
    print(f"  Info: {ARTIFACTS_DIR / 'hybrid_retrain_info.json'}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
