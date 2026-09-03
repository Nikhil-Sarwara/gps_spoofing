#!/usr/bin/env python3
"""
generate_paper_figures.py — Generate all figures needed for the paper.

Generates:
  1. rf_test_confusion_matrix.png — test set confusion matrix
  2. rf_feature_importance.png — top 15 feature importances
  3. rf_train_confusion_matrix.png — training set confusion matrix

Usage:
    python Step_4_Detection/scripts/generate_paper_figures.py
    python Step_4_Detection/scripts/generate_paper_figures.py --output-dir /path/to/paper/figures
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

WINDOW_LEN = 30
STRIDE = 15
MIN_WINDOW_LABEL_RATIO = 0.5
ROOT = Path(__file__).resolve().parent.parent.parent
SYNTHETIC_DIR = ROOT / "Step_5_Data" / "synthetic_v2"
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
        * EARTH_RADIUS_M * np.cos(np.radians(first_lat))
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


def plot_confusion_matrix(y_true, y_pred, title, filepath, class_names=("Normal", "Spoofed")):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    return cm


def plot_feature_importance(rf, feature_names, filepath, top_n=15):
    importances = rf.feature_importances_

    # Aggregate importance per feature across all timesteps
    n_features = len(feature_names)
    aggregated = np.zeros(n_features)
    for i in range(len(importances)):
        aggregated[i % n_features] += importances[i]

    # Sort and get top N
    indices = np.argsort(aggregated)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_importances = [aggregated[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances[::-1], align="center", color="#2196F3")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Aggregated Gini Importance")
    ax.set_title("Random Forest Feature Importance (Top 15)")
    fig.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for figures (default: Step_5_Data/models)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    # --- Load and prepare data ---
    print("\n[1/4] Loading synthetic data...")
    all_X, all_y = [], []
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

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    print(f"  Total: {len(X)} windows (normal={int((y==0).sum())}, spoof={int((y==1).sum())})")

    # --- Split ---
    n = len(X)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    # --- Scale ---
    print("\n[2/4] Training model...")
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

    # --- Generate figures ---
    print("\n[3/4] Generating confusion matrices...")

    # Test set confusion matrix
    test_pred = rf.predict(X_test_sc)
    test_cm = plot_confusion_matrix(
        y_test, test_pred,
        "Random Forest Test Set Confusion Matrix",
        output_dir / "rf_test_confusion_matrix.png",
    )
    print(f"  ✅ {output_dir / 'rf_test_confusion_matrix.png'}")
    print(f"     TN={test_cm[0][0]}, FP={test_cm[0][1]}, FN={test_cm[1][0]}, TP={test_cm[1][1]}")

    # Train set confusion matrix
    train_pred = rf.predict(X_tr_sc)
    train_cm = plot_confusion_matrix(
        y_train, train_pred,
        "Random Forest Training Set Confusion Matrix",
        output_dir / "rf_train_confusion_matrix.png",
    )
    print(f"  ✅ {output_dir / 'rf_train_confusion_matrix.png'}")

    # Feature importance
    print("\n[4/4] Generating feature importance plot...")
    plot_feature_importance(rf, FEATURE_COLS, output_dir / "rf_feature_importance.png")
    print(f"  ✅ {output_dir / 'rf_feature_importance.png'}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob("rf_*.png")):
        print(f"  {f}")
    print(f"\nTo copy to paper repo:")
    print(f"  cp {output_dir / 'rf_test_confusion_matrix.png'} ~/gps-spoofing-paper/figures/")
    print(f"  cp {output_dir / 'rf_test_confusion_matrix.png'} ~/gps-spoofing-paper/big\\ font\\ rf_confusion_matrix.png")
    print(f"  cp {output_dir / 'rf_feature_importance.png'} ~/gps-spoofing-paper/figures/")
    print(f"\nTest metrics:")
    print(classification_report(y_test, test_pred, target_names=["Normal", "Spoofed"]))


if __name__ == "__main__":
    main()
