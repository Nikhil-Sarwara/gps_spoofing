#!/usr/bin/env python3
"""
retrain_clean.py — Retrain with clean ground truth from synthetic data.

Strategy:
  - Train on synthetic data ONLY (clean ground truth from 00_augment_synthetic.py)
  - Test on real flights (ground truth from *_auto_segments.json or *_labels.csv)
  - Evaluate spoofed-class metrics properly

Usage:
    python Step_4_Detection/scripts/retrain_clean.py
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

# ---------------------------------------------------------------------------
# Constants (match pipeline exactly)
# ---------------------------------------------------------------------------
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


def gps_to_local(df: pd.DataFrame) -> pd.DataFrame:
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


def load_synthetic_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load synthetic flights with GROUND TRUTH labels."""
    X_all, y_all = [], []
    source_files = []

    for csv_path in sorted(SYNTHETIC_DIR.glob("*_cleaned.csv")):
        df = pd.read_csv(csv_path)
        if len(df) < WINDOW_LEN:
            continue

        # Labels are BUILT INTO the CSV (from 00_augment_synthetic.py)
        if "label" not in df.columns:
            print(f"  [SKIP] {csv_path.name}: no 'label' column")
            continue

        labels = df["label"].values.astype(int)

        # Convert GPS -> local
        df = gps_to_local(df)

        # Ensure features exist
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
            X_all.append(np.array(X))
            y_all.append(np.array(y))
            source_files.append(csv_path.name)
            n_spoof = sum(y)
            print(f"  [SYN] {csv_path.name}: {len(X)} windows (normal={len(X)-n_spoof}, spoof={n_spoof})")

    if not X_all:
        return np.empty((0,)), np.empty((0,)), []

    return np.concatenate(X_all), np.concatenate(y_all), source_files


def load_real_flights_for_test() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load real flights for TESTING only. Labels from *_labels.csv or segments."""
    X_all, y_all = [], []
    source_files = []
    per_file_windows = []  # track per-file window counts for breakdown

    # Find real flight CSVs (not synthetic, not zero-row)
    real_csvs = []
    for csv_path in sorted(PROCESSED_DIR.glob("*_cleaned.csv")):
        # Skip synthetic files that might be in processed/
        if "flight_" in csv_path.name and any(
            t in csv_path.name for t in ["flat_", "mountain_", "sea_"]
        ):
            continue
        if csv_path.stat().st_size < 1000:  # Skip tiny/empty files
            continue
        real_csvs.append(csv_path)

    for csv_path in real_csvs:
        df = pd.read_csv(csv_path)
        if len(df) < WINDOW_LEN:
            continue

        # Try to get labels
        labels = None

        # Method 1: label column already in CSV
        if "label" in df.columns:
            labels = df["label"].values.astype(int)

        # Method 2: *_labels.csv in same directory
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
            print(f"  [SKIP] {csv_path.name}: no labels found")
            continue

        # Convert GPS -> local
        if "lat_deg" in df.columns and "lon_deg" in df.columns:
            df = gps_to_local(df)

        # Ensure features
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
            X_all.append(np.array(X))
            y_all.append(np.array(y))
            source_files.append(csv_path.name)
            n_spoof = sum(y)
            per_file_windows.append(len(X))
            print(f"  [TEST] {csv_path.name}: {len(X)} windows (normal={len(X)-n_spoof}, spoof={n_spoof})")

    if not X_all:
        return np.empty((0,)), np.empty((0,)), [], []

    return np.concatenate(X_all), np.concatenate(y_all), source_files, per_file_windows


def main():
    print("=" * 60)
    print("RETRAIN WITH CLEAN GROUND TRUTH")
    print("=" * 60)

    # --- Load synthetic (training) ---
    print("\n[1/4] Loading synthetic data (ground truth labels)...")
    X_synth, y_synth, synth_files = load_synthetic_data()
    print(f"\n  Synthetic total: {len(X_synth)} windows")
    print(f"  Label dist: {dict(zip(*np.unique(y_synth, return_counts=True)))}")

    if len(X_synth) == 0:
        print("ERROR: No synthetic data loaded. Run 00_augment_synthetic.py first.")
        return

    # --- Load real flights (testing) ---
    print("\n[2/4] Loading real flights (test only)...")
    X_real, y_real, real_files = load_real_flights_for_test()
    print(f"\n  Real total: {len(X_real)} windows")
    if len(X_real) > 0:
        print(f"  Label dist: {dict(zip(*np.unique(y_real, return_counts=True)))}")

    # --- Split synthetic into train/val ---
    n = len(X_synth)
    n_train = int(0.8 * n)
    n_val = n - n_train

    X_train, y_train = X_synth[:n_train], y_synth[:n_train]
    X_val, y_val = X_synth[n_train:], y_synth[n_train:]

    print(f"\n  Train: {len(X_train)} windows (normal={int((y_train==0).sum())}, spoof={int((y_train==1).sum())})")
    print(f"  Val:   {len(X_val)} windows (normal={int((y_val==0).sum())}, spoof={int((y_val==1).sum())})")
    print(f"  Test:  {len(X_real)} windows (real flights)")

    # --- Scale ---
    print("\n[3/4] Training Random Forest...")
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    scaler.fit(X_train_flat)

    X_train_sc = scaler.transform(X_train_flat)
    X_val_sc = scaler.transform(X_val_flat)

    # --- Train RF ---
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train_sc, y_train)

    # Train results
    train_pred = rf.predict(X_train_sc)
    print("\n  === TRAIN RESULTS (synthetic) ===")
    print(classification_report(y_train, train_pred, zero_division=0))

    # Val results
    val_pred = rf.predict(X_val_sc)
    print("  === VAL RESULTS (synthetic) ===")
    print(classification_report(y_val, val_pred, zero_division=0))

    # --- Test on real flights ---
    print("[4/4] Testing on real flights...")
    if len(X_real) > 0:
        X_real_flat = X_real.reshape(len(X_real), -1)
        X_real_sc = scaler.transform(X_real_flat)
        real_pred = rf.predict(X_real_sc)
        real_proba = rf.predict_proba(X_real_sc)[:, 1]

        print("\n  === TEST RESULTS (real flights) ===")
        print(classification_report(y_real, real_pred, zero_division=0))

        cm = confusion_matrix(y_real, real_pred)
        print(f"  Confusion Matrix:\n{cm}")

        # Per-file breakdown
        print("\n  === PER-FILE BREAKDOWN ===")
        idx = 0
        for fname, n_w in zip(real_files, per_file_windows):
            file_pred = real_pred[idx:idx+n_w]
            file_true = y_real[idx:idx+n_w]
            n_correct = (file_pred == file_true).sum()
            n_spoof = file_true.sum()
            n_spoof_caught = ((file_pred == 1) & (file_true == 1)).sum()
            print(f"  {fname}: {n_w} windows, {n_correct}/{n_w} correct, "
                  f"spoof_gt={n_spoof}, spoof_detected={n_spoof_caught}")
            idx += n_w

        # Summary
        spoof_f1 = f1_score(y_real, real_pred, pos_label=1, zero_division=0)
        normal_f1 = f1_score(y_real, real_pred, pos_label=0, zero_division=0)
        print(f"\n  === SUMMARY ===")
        print(f"  Normal F1: {normal_f1:.3f}")
        print(f"  Spoofed F1: {spoof_f1:.3f}")
        print(f"  Overall accuracy: {(real_pred == y_real).mean():.3f}")
    else:
        print("  No real flight data available for testing.")

    # --- Save model ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "rf_model_clean.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(MODELS_DIR / "scaler_clean.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save dataset info
    info = {
        "approach": "train_on_synthetic_ground_truth",
        "synthetic_windows": int(len(X_synth)),
        "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)),
        "test_real_windows": int(len(X_real)),
        "train_label_dist": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "val_label_dist": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
        "test_label_dist": {str(k): int(v) for k, v in zip(*np.unique(y_real, return_counts=True))} if len(X_real) > 0 else {},
        "synthetic_source_files": synth_files,
        "real_test_files": real_files,
    }
    with open(ARTIFACTS_DIR / "clean_retrain_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n  Model saved: {MODELS_DIR / 'rf_model_clean.pkl'}")
    print(f"  Scaler saved: {MODELS_DIR / 'scaler_clean.pkl'}")
    print(f"  Info saved: {ARTIFACTS_DIR / 'clean_retrain_info.json'}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
