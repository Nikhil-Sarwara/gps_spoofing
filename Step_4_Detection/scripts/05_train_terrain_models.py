"""05_train_terrain_models.py — Step 2: Per-Terrain Model Training

Reads all cleaned CSVs with terrain_type column, splits by terrain,
builds sliding windows, trains RF + CNN per terrain, saves artifacts.

CLI:
    python 05_train_terrain_models.py \\
        --processed-dirs Step_5_Data/processed \\
        --models-dir Step_5_Data/models/terrain \\
        --artifacts-dir Step_5_Data/artifacts \\
        --model-type both
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants — match global pipeline exactly
# ---------------------------------------------------------------------------
WINDOW_LEN              = 30
STRIDE                  = 15
MIN_WINDOW_LABEL_RATIO  = 0.5
TRAIN_RATIO             = 0.70
VAL_RATIO               = 0.15
# TEST_RATIO             = 0.15  (remainder)

EPOCHS                  = 50
LR                      = 0.001

TERRAINS                = ["flat", "mountain", "sea"]

EARTH_RADIUS_M          = 6_371_000.0

# ---------------------------------------------------------------------------
# WindowCNN — identical to 04_train_baseline.py
# ---------------------------------------------------------------------------
class WindowCNN(nn.Module):
    def __init__(self, num_features: int, num_classes: int = 1):
        super().__init__()
        self.conv1   = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gps_to_local(df: pd.DataFrame) -> pd.DataFrame:
    """Add x_m, y_m columns using local ENU frame from first row."""
    df = df.copy()
    first_lat = df["lat_deg"].iloc[0]
    first_lon = df["lon_deg"].iloc[0]
    df["x_m"] = np.radians(df["lat_deg"] - first_lat) * EARTH_RADIUS_M
    df["y_m"] = (
        np.radians(df["lon_deg"] - first_lon)
        * EARTH_RADIUS_M
        * np.cos(np.radians(first_lat))
    )
    return df


def get_row_labels(
    df: pd.DataFrame,
    csv_path: Path,
    processed_dir: Path,
) -> np.ndarray:
    """Priority: row_labels_auto.csv -> auto_segments.json -> all zeros."""
    n = len(df)

    # 1. row_labels_auto.csv in same processed dir
    labels_csv = processed_dir / "row_labels_auto.csv"
    if labels_csv.exists():
        try:
            ldf = pd.read_csv(labels_csv)
            # Try to match by filename stem
            stem = csv_path.stem.replace("_cleaned", "")
            if "log_name" in ldf.columns:
                subset = ldf[ldf["log_name"].str.contains(stem, na=False)]
                if len(subset) == n and "label" in subset.columns:
                    return subset["label"].values.astype(int)
            # Fallback: if single log in dir, use all rows
            if "label" in ldf.columns and len(ldf) == n:
                return ldf["label"].values.astype(int)
        except Exception as e:
            print(f"    [warn] row_labels_auto.csv read error: {e}")

    # 2. label column already in the CSV (synthetic files)
    if "label" in df.columns:
        return df["label"].values.astype(int)

    # 3. auto_segments.json in same processed dir
    # Check per-file segments first (synthetic: flat_flight_01_auto_segments.json)
    per_file_segs = csv_path.parent / csv_path.name.replace("_cleaned.csv", "_auto_segments.json")
    segs_path = per_file_segs if per_file_segs.exists() else processed_dir / "auto_segments.json"
    if segs_path.exists():
        try:
            with open(segs_path) as f:
                segs = json.load(f)
            if "time_s" not in df.columns:
                raise KeyError("time_s")
            time_s = df["time_s"].values.astype(float)
            labels = np.zeros(n, dtype=int)
            for seg in segs:
                mask = (time_s >= seg["start_s"]) & (time_s < seg["end_s"])
                labels[mask] = int(seg["label"])
            return labels
        except Exception as e:
            print(f"    [warn] auto_segments.json parse error for {csv_path.name}: {e}")

    # 4. Fallback
    print(f"    [warn] No labels found for {csv_path.name} — labelling all rows as 0 (normal)")
    return np.zeros(n, dtype=int)


def build_windows(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window builder — matches 03_make_windows.py logic."""
    X, y = [], []
    for start in range(0, len(df) - WINDOW_LEN + 1, STRIDE):
        end   = start + WINDOW_LEN
        win   = df.iloc[start:end][feature_cols].values.astype(np.float32)
        win_y = labels[start:end]
        counts = np.bincount(win_y, minlength=2)
        label  = 1 if counts[1] / WINDOW_LEN >= MIN_WINDOW_LABEL_RATIO else 0
        X.append(win)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 70/15/15 split — NO shuffle to prevent data leakage."""
    n       = len(X)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    return (
        X[:n_train], y[:n_train],
        X[n_train: n_train + n_val], y[n_train: n_train + n_val],
        X[n_train + n_val:], y[n_train + n_val:],
    )


def label_dist(y: np.ndarray) -> dict:
    counts = np.bincount(y, minlength=2)
    return {"0": int(counts[0]), "1": int(counts[1])}


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    title: str,
) -> None:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Spoof"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Spoof"])
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# RF trainer
# ---------------------------------------------------------------------------

def train_rf_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    out_dir: Path,
) -> tuple[float, float]:
    """Train RF, save model + confusion matrix. Returns (val_accuracy, val_f1)."""
    X_tr = X_train.reshape(X_train.shape[0], -1)
    X_v  = X_val.reshape(X_val.shape[0], -1)

    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    rf.fit(X_tr, y_train)

    val_pred = rf.predict(X_v)
    acc = accuracy_score(y_val, val_pred)
    f1  = f1_score(y_val, val_pred, zero_division=0)

    with open(out_dir / "rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    save_confusion_matrix(y_val, val_pred, out_dir / "rf_val_cm.png",
                          f"RF Val CM ({out_dir.name})")
    return float(acc), float(f1)


# ---------------------------------------------------------------------------
# CNN trainer
# ---------------------------------------------------------------------------

def train_cnn_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    num_features: int,
    out_dir: Path,
) -> tuple[float, float]:
    """Train CNN, save model + confusion matrix. Returns (val_accuracy, val_f1)."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    batch_size = min(32, len(X_train))

    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.FloatTensor(X.transpose(0, 2, 1)),
            torch.FloatTensor(y).unsqueeze(1),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)

    model     = WindowCNN(num_features).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss    = float("inf")
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += criterion(model(bx), by).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final val predictions
    model.eval()
    val_pred, val_true = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            preds = (model(bx).cpu().numpy() > 0.5).astype(int).flatten()
            val_pred.extend(preds)
            val_true.extend(by.numpy().astype(int).flatten())

    val_pred = np.array(val_pred)
    val_true = np.array(val_true)

    acc = accuracy_score(val_true, val_pred)
    f1  = f1_score(val_true, val_pred, zero_division=0)

    torch.save(model.state_dict(), out_dir / "cnn_model.pth")
    save_confusion_matrix(val_true, val_pred, out_dir / "cnn_val_cm.png",
                          f"CNN Val CM ({out_dir.name})")
    return float(acc), float(f1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_terrain_data(
    processed_dirs: list[Path],
    feature_cols: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Load + window all cleaned CSVs, split by terrain_type.

    Returns:
        dict: terrain -> (X_windows, y_labels, source_files)
    """
    terrain_data = {t: [] for t in TERRAINS}
    terrain_files = {t: [] for t in TERRAINS}

    for proc_dir in processed_dirs:
        proc_dir = Path(proc_dir)
        if not proc_dir.exists():
            print(f"  [warn] processed dir not found: {proc_dir}")
            continue

        csv_paths = sorted(proc_dir.glob("*_cleaned.csv"))
        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path, low_memory=False)
            except Exception as e:
                print(f"  [warn] Could not read {csv_path.name}: {e}")
                continue

            if len(df) == 0:
                print(f"  [skip] {csv_path.name}: 0 rows")
                continue

            if "terrain_type" not in df.columns:
                print(f"  [skip] {csv_path.name}: no terrain_type column (run Step 1 first)")
                continue

            # Convert GPS -> local frame
            if "lat_deg" in df.columns and "lon_deg" in df.columns:
                df = gps_to_local(df)

            # Fill missing feature columns with 0
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0.0

            # Force numeric
            for col in feature_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

            # Get row labels
            row_labels = get_row_labels(df, csv_path, proc_dir)

            # Majority terrain for this file
            terrain = df["terrain_type"].mode().iloc[0]
            if terrain not in TERRAINS:
                print(f"  [skip] {csv_path.name}: unknown terrain '{terrain}'")
                continue

            X, y = build_windows(df, row_labels, feature_cols)
            if len(X) == 0:
                print(f"  [skip] {csv_path.name}: 0 windows produced")
                continue

            terrain_data[terrain].append((X, y))
            terrain_files[terrain].append(csv_path.name)
            print(f"  [load] {csv_path.name:55s} terrain={terrain:8s} "
                  f"windows={len(X)} (0={int((y==0).sum())}, 1={int((y==1).sum())})")

    # Concatenate per terrain
    result = {}
    for terrain in TERRAINS:
        chunks = terrain_data[terrain]
        if not chunks:
            result[terrain] = (np.empty((0,), dtype=np.float32),
                               np.empty((0,), dtype=int),
                               [])
            continue
        Xs = np.concatenate([c[0] for c in chunks], axis=0)
        ys = np.concatenate([c[1] for c in chunks], axis=0)
        result[terrain] = (Xs, ys, terrain_files[terrain])

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Train per-terrain GPS spoofing detectors")
    parser.add_argument("--processed-dirs", nargs="+",
                        default=["Step_5_Data/processed", "Step_5_Data/synthetic"],
                        help="Directories containing *_cleaned.csv files")
    parser.add_argument("--models-dir",    default="Step_5_Data/models/terrain")
    parser.add_argument("--artifacts-dir", default="Step_5_Data/artifacts")
    parser.add_argument("--model-type",    default="both",
                        choices=["rf", "cnn", "both"])
    args = parser.parse_args(argv)

    # Resolve paths relative to project root
    # 4_Detection_Engine/scripts/05_train_terrain_models.py
    root = Path(__file__).resolve().parent.parent.parent
    processed_dirs = [root / d for d in args.processed_dirs]
    models_dir     = root / args.models_dir
    artifacts_dir  = root / args.artifacts_dir

    # Load feature names
    feat_path = artifacts_dir / "feature_names.json"
    if not feat_path.exists():
        print(f"  [error] feature_names.json not found at: {feat_path}")
        return
        
    with open(feat_path) as f:
        feature_cols: list[str] = json.load(f)
    n_features = len(feature_cols)
    print(f"\nFeatures ({n_features}): {feature_cols[:5]} ... {feature_cols[-3:]}")

    # Load all data
    print("\n" + "="*60)
    print("Loading cleaned CSVs...")
    print("="*60)
    terrain_data = load_all_terrain_data(processed_dirs, feature_cols)

    # Per-terrain training
    summary = {}
    print("\n" + "="*60)
    print("Training terrain models...")
    print("="*60)

    for terrain in TERRAINS:
        X, y, source_files = terrain_data[terrain]
        out_dir = models_dir / terrain
        out_dir.mkdir(parents=True, exist_ok=True)

        n_windows = len(X)
        print(f"\n[{terrain.upper()}] {n_windows} windows from {len(source_files)} files")

        # --- Insufficient data guard ---
        if n_windows < 10:
            print(f"  ⚠ Skipping {terrain}: only {n_windows} windows (need ≥10)")
            info = {
                "terrain_type": terrain, "status": "insufficient_data",
                "n_windows": n_windows, "source_csv_files": source_files,
                "window_len": WINDOW_LEN, "stride": STRIDE, "num_features": n_features,
            }
            with open(out_dir / "dataset_info.json", "w") as f:
                json.dump(info, f, indent=2)
            summary[terrain] = {"n_windows": n_windows, "rf_val_f1": None,
                                "cnn_val_f1": None, "status": "skip"}
            continue

        # --- Split ---
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)
        print(f"  Split: train={len(X_train)} val={len(X_val)} test={len(X_test)}")
        print(f"  Train labels: {label_dist(y_train)}")
        print(f"  Val   labels: {label_dist(y_val)}")
        print(f"  Test  labels: {label_dist(y_test)}")

        # --- Scaler (fit on train only) ---
        scaler = StandardScaler()
        n_t = len(X_train)
        X_train_2d = X_train.reshape(n_t, -1)
        scaler.fit(X_train_2d)

        def scale(arr):
            n = len(arr)
            return scaler.transform(arr.reshape(n, -1)).reshape(n, WINDOW_LEN, n_features)

        X_train_s = scale(X_train)
        X_val_s   = scale(X_val)
        X_test_s  = scale(X_test)

        with open(out_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # --- RF ---
        rf_acc = rf_f1 = None
        if args.model_type in ("rf", "both"):
            print(f"  Training RF...")
            rf_acc, rf_f1 = train_rf_model(X_train_s, y_train, X_val_s, y_val, out_dir)
            print(f"  RF  val → acc={rf_acc:.3f}  f1={rf_f1:.3f}")

        # --- CNN (only if ≥30 windows) ---
        cnn_acc = cnn_f1 = None
        cnn_status = "skipped_insufficient_data"
        if args.model_type in ("cnn", "both"):
            if n_windows >= 30:
                print(f"  Training CNN...")
                cnn_acc, cnn_f1 = train_cnn_model(
                    X_train_s, y_train, X_val_s, y_val, n_features, out_dir
                )
                print(f"  CNN val → acc={cnn_acc:.3f}  f1={cnn_f1:.3f}")
                cnn_status = "trained"
            else:
                print(f"  CNN skipped: {n_windows} windows < 30")

        # --- dataset_info.json ---
        info = {
            "terrain_type":       terrain,
            "status":             "trained",
            "window_len":         WINDOW_LEN,
            "stride":             STRIDE,
            "num_features":       n_features,
            "feature_names":      feature_cols,
            "n_total_windows":    n_windows,
            "n_train":            int(len(X_train)),
            "n_val":              int(len(X_val)),
            "n_test":             int(len(X_test)),
            "label_dist": {
                "train": label_dist(y_train),
                "val":   label_dist(y_val),
                "test":  label_dist(y_test),
            },
            "rf_val_accuracy":    rf_acc,
            "rf_val_f1":          rf_f1,
            "cnn_val_accuracy":   cnn_acc,
            "cnn_val_f1":         cnn_f1,
            "cnn_status":         cnn_status,
            "source_csv_files":   source_files,
        }
        with open(out_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)

        status = "trained" if cnn_status == "trained" else "rf_only"
        summary[terrain] = {
            "n_windows":  n_windows,
            "rf_val_f1":  rf_f1,
            "cnn_val_f1": cnn_f1,
            "status":     status,
        }

    # --- terrain_training_report.json ---
    trained = {t: s for t, s in summary.items() if s["status"] != "skip"}
    default_model = max(trained, key=lambda t: trained[t]["n_windows"]) if trained else "flat"

    report = {
        "generated_at":            datetime.now(timezone.utc).isoformat(),
        "total_source_csvs":       sum(len(terrain_data[t][2]) for t in TERRAINS),
        "terrain_summary":         summary,
        "recommended_default_model": default_model,
        "note":                    "Terrain with most training data recommended as fallback default",
    }
    report_path = models_dir / "terrain_training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Summary table ---
    w = 60
    hdr = f"{chr(9556)}{chr(9552)*w}{chr(9559)}"
    ttl = f"{chr(9553)}{'TERRAIN MODEL TRAINING SUMMARY':^{w}}{chr(9553)}"
    sep1 = f"{chr(9568)}{chr(9552)*11}{chr(9574)}{chr(9552)*9}{chr(9574)}{chr(9552)*10}{chr(9574)}{chr(9552)*10}{chr(9574)}{chr(9552)*15}{chr(9571)}"
    col = f"{chr(9553)}{'Terrain':^11}{chr(9553)}{'Windows':^9}{chr(9553)}{'RF Val F1':^10}{chr(9553)}{'CNN Val F1':^10}{chr(9553)}{'Status':^15}{chr(9553)}"
    sep2 = f"{chr(9568)}{chr(9552)*11}{chr(9579)}{chr(9552)*9}{chr(9579)}{chr(9552)*10}{chr(9579)}{chr(9552)*10}{chr(9579)}{chr(9552)*15}{chr(9571)}"
    bot = f"{chr(9562)}{chr(9552)*11}{chr(9577)}{chr(9552)*9}{chr(9577)}{chr(9552)*10}{chr(9577)}{chr(9552)*10}{chr(9577)}{chr(9552)*15}{chr(9565)}"

    print(f"\n{hdr}")
    print(ttl)
    print(sep1)
    print(col)
    print(sep2)
    for terrain in TERRAINS:
        s      = summary.get(terrain, {"n_windows": 0, "rf_val_f1": None, "cnn_val_f1": None, "status": "skip"})
        rf_str = f"{s['rf_val_f1']:.3f}" if s["rf_val_f1"] is not None else "  —  "
        cn_str = f"{s['cnn_val_f1']:.3f}" if s["cnn_val_f1"] is not None else "  —  "
        print(f"{chr(9553)}{terrain:^11}{chr(9553)}{s['n_windows']:^9}{chr(9553)}{rf_str:^10}{chr(9553)}{cn_str:^10}{chr(9553)}{s['status']:^15}{chr(9553)}")
    print(bot)
    print(f"\n  Recommended default model : {default_model}")
    print(f"  Report saved               : {report_path}")
    print(f"  Models saved to            : {models_dir}\n")


if __name__ == "__main__":
    main()
