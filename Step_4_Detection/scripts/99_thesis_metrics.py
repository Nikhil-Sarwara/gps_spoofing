"""
99_thesis_metrics.py — Thesis Improvement & Validation Script

This script performs:
1. Leave-One-Mission-Out (LOMO) Cross-Validation for the General RF model.
2. Training and evaluation of an ML-based Terrain Identifier (matching the thesis claim).
3. Generation of high-quality Confusion Matrices and ROC curves for the thesis.
4. Generation of a multi-pane Telemetry Trace Figure showing a spoofing attack in action.

Usage:
    python Step_4_Detection/scripts/99_thesis_metrics.py
"""

import json
import os
import pickle
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler

# Constants
WINDOW_LEN = 30
STRIDE = 15
MIN_WINDOW_LABEL_RATIO = 0.5
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "Step_5_Data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
FIGURES_DIR = ROOT / "Step_7_Research" / "thesis" / "figures"
MODELS_DIR = DATA_DIR / "models"

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Feature names from artifacts
with open(DATA_DIR / "artifacts" / "feature_names.json") as f:
    FEATURE_COLS = json.load(f)

def load_mission_data():
    """Load all cleaned CSVs and group by mission (filename)."""
    missions = {}
    
    csv_paths = list(PROCESSED_DIR.glob("*_cleaned_cleaned.csv")) + \
                list(SYNTHETIC_DIR.glob("*_cleaned.csv"))
    
    # Filter out empty or very small files
    csv_paths = [p for p in csv_paths if p.stat().st_size > 5000]
    
    print(f"Loading {len(csv_paths)} flight logs...")
    
    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        if "terrain_type" not in df.columns:
            # Try to infer from filename
            if "flat" in path.name: terrain = "flat"
            elif "mountain" in path.name: terrain = "mountain"
            elif "sea" in path.name: terrain = "sea"
            else: terrain = "flat"
        else:
            terrain = df["terrain_type"].mode().iloc[0]
            
        # Get labels
        labels = np.zeros(len(df), dtype=int)
        if "label" in df.columns:
            labels = df["label"].values
        else:
            # Look for segments
            stem = path.name.replace("_cleaned_cleaned.csv", "").replace("_cleaned.csv", "")
            segs_path = path.parent / f"{stem}_auto_segments.json"
            if segs_path.exists():
                with open(segs_path) as f:
                    segs = json.load(f)
                for seg in segs:
                    mask = (df["time_s"] >= seg["start_s"]) & (df["time_s"] < seg["end_s"])
                    labels[mask] = seg["label"]
        
        # Convert GPS -> local frame if x_m/y_m missing
        if "x_m" not in df.columns and "lat_deg" in df.columns:
            first_lat = df["lat_deg"].iloc[0]
            first_lon = df["lon_deg"].iloc[0]
            EARTH_RADIUS_M = 6_371_000.0
            df["x_m"] = np.radians(df["lat_deg"] - first_lat) * EARTH_RADIUS_M
            df["y_m"] = (
                np.radians(df["lon_deg"] - first_lon)
                * EARTH_RADIUS_M
                * np.cos(np.radians(first_lat))
            )

        # Ensure all feature columns exist
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
            missions[path.name] = {
                "X": np.array(X),
                "y": np.array(y),
                "terrain": terrain,
                "df": df,  # Keep df for trace plotting later
                "labels": labels
            }
            
    return missions

def run_lomo_cv(missions):
    """Leave-One-Mission-Out Cross-Validation for General RF."""
    print("\nStarting LOMO Cross-Validation for General RF...")
    mission_names = sorted(list(missions.keys()))
    # Remove duplicates like '...cleaned.csv' vs '...cleaned_cleaned.csv'
    unique_mission_roots = {}
    for name in mission_names:
        root = name.replace("_cleaned_cleaned.csv", "").replace("_cleaned.csv", "")
        if root not in unique_mission_roots:
            unique_mission_roots[root] = name
    
    selected_roots = sorted(list(unique_mission_roots.keys()))
    
    # For speed and stability, we'll pick a representative set if too many, 
    # but let's try 15 missions (5 per terrain).
    selected_missions = []
    for t in ["flat", "mountain", "sea"]:
        t_missions = [r for r in selected_roots if missions[unique_mission_roots[r]]["terrain"] == t]
        selected_missions.extend([unique_mission_roots[r] for r in t_missions[:5]])
        
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for i, test_mission in enumerate(selected_missions):
        print(f"  [{i+1}/{len(selected_missions)}] Testing on {test_mission}...")
        
        # Split
        X_train_list = [missions[m]["X"] for m in selected_missions if m != test_mission]
        y_train_list = [missions[m]["y"] for m in selected_missions if m != test_mission]
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = missions[test_mission]["X"]
        y_test = missions[test_mission]["y"]
        
        # Flatten for RF
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Train
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        rf.fit(X_train_flat, y_train)
        
        # Predict
        y_pred = rf.predict(X_test_flat)
        y_proba = rf.predict_proba(X_test_flat)[:, 1]
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        
    print("\nLOMO Results (General RF):")
    print(classification_report(all_y_true, all_y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(all_y_true, all_y_pred)
    plot_cm(cm, ["Normal", "Spoof"], "LOMO General RF", "lomo_rf_cm.png")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_y_true, all_y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (LOMO RF)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / "lomo_rf_roc.png")
    plt.close()
    
    return all_y_true, all_y_pred

def train_terrain_identifier(missions):
    """Train an ML model to identify terrain (matching thesis claim)."""
    print("\nTraining ML-based Terrain Identifier...")
    
    X_list = []
    y_list = []
    terrain_map = {"flat": 0, "mountain": 1, "sea": 2}
    
    for m in missions.values():
        X_list.append(m["X"])
        y_list.extend([terrain_map[m["terrain"]]] * len(m["X"]))
        
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    
    # Flatten
    X_flat = X.reshape(X.shape[0], -1)
    
    # Train-test split (random for terrain ID as we want to see if it generalizes across all samples)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("Terrain Identifier Results:")
    print(classification_report(y_test, y_pred, target_names=["flat", "mountain", "sea"]))
    
    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, ["Flat", "Mountain", "Sea"], "Terrain Classifier", "terrain_classifier_cm.png")

def plot_cm(cm, labels, title, filename):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", 
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename)
    plt.close()

def generate_telemetry_trace(missions):
    """Generate a multi-pane trace figure for a spoofing attack."""
    print("\nGenerating Telemetry Trace Figure...")
    
    # Find a mission with an anomaly
    attack_mission = None
    for name, m in missions.items():
        if 1 in m["y"]:
            attack_mission = name
            break
            
    if not attack_mission:
        print("No attack mission found for trace plotting.")
        return
        
    m = missions[attack_mission]
    df = m["df"]
    labels = m["labels"]
    
    # Train a quick model on OTHER data to get probabilities for THIS mission
    X_train_list = [missions[name]["X"] for name in missions if name != attack_mission]
    y_train_list = [missions[name]["y"] for name in missions if name != attack_mission]
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    
    X_test_flat = m["X"].reshape(m["X"].shape[0], -1)
    probs = rf.predict_proba(X_test_flat)[:, 1]
    
    # Align probabilities with original timeline (one prob per window)
    times = df["time_s"].values
    window_times = []
    for start in range(0, len(df) - WINDOW_LEN + 1, STRIDE):
        window_times.append(times[start + WINDOW_LEN // 2])
        
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Pane 1: Velocity Innovation (EKF)
    ax1.plot(df["time_s"], df["vel_innov"], label="Vel Innovation (m/s)", color="blue")
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label="Threshold")
    ax1.set_ylabel("EKF Innovation")
    ax1.legend(loc="upper right")
    ax1.set_title(f"Spoofing Attack Signature Analysis: {attack_mission}")
    
    # Pane 2: Vibration Z (Noise indicator)
    ax2.plot(df["time_s"], df["vibration_z"], label="Vibration Z", color="green")
    ax2.set_ylabel("Vibration")
    ax2.legend(loc="upper right")
    
    # Pane 3: Anomaly Probability
    ax3.plot(window_times, probs, label="ML Anomaly Prob", color="red", lw=2)
    ax3.fill_between(window_times, 0, probs, color='red', alpha=0.1)
    ax3.axhline(y=0.5, color='black', linestyle='--', label="Decision Boundary")
    ax3.set_ylabel("Probability")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc="upper right")
    
    # Highlight attack region
    attack_mask = labels == 1
    if any(attack_mask):
        start_attack = df["time_s"][attack_mask].min()
        end_attack = df["time_s"][attack_mask].max()
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(start_attack, end_attack, color='red', alpha=0.1, label="Actual Attack")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "telemetry_trace_attack.png")
    plt.close()
    print(f"Saved trace to {FIGURES_DIR / 'telemetry_trace_attack.png'}")

if __name__ == "__main__":
    missions = load_mission_data()
    if not missions:
        print("No missions loaded. Check your data directories.")
    else:
        run_lomo_cv(missions)
        train_terrain_identifier(missions)
        generate_telemetry_trace(missions)
        print("\nAll thesis improvements metrics and figures generated successfully.")
