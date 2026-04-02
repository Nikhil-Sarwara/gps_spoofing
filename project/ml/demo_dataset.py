#!/usr/bin/env python3
"""
PX4 Anomaly Detector Demo - Dataset Replay

- Loads cleaned CSV + row labels + window dataset + trained CNN
- Reconstructs windows with their time ranges and labels
- Prints a structured, readable summary for each window:
    [index] time_start–time_end | label | mode | failsafe | prob
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

import torch
import torch.nn as nn

ARTIFACTS_DIR = Path("ml/artifacts")
MODELS_DIR = Path("ml/models")
GPS_PROCESSED_CSV = Path("gps_logs/processed/gps_log_20260402_184048_cleaned.csv")
ROW_LABELS_CSV = Path("gps_logs/processed/row_labels.csv")

console = Console()


class WindowCNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 32, 3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x))


def load_artifacts():
    data = np.load(ARTIFACTS_DIR / "dataset.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    win_start_train = data["win_start_train"]
    win_end_train = data["win_end_train"]
    win_start_val = data["win_start_val"]
    win_end_val = data["win_end_val"]
    win_start_test = data["win_start_test"]
    win_end_test = data["win_end_test"]

    with open(ARTIFACTS_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "win_start_train": win_start_train,
        "win_end_train": win_end_train,
        "win_start_val": win_start_val,
        "win_end_val": win_end_val,
        "win_start_test": win_start_test,
        "win_end_test": win_end_test,
        "feature_names": feature_names,
        "scaler": scaler,
    }


def load_model(num_features: int) -> WindowCNN:
    model = WindowCNN(num_features)
    state_dict = torch.load(MODELS_DIR / "cnn_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_raw_and_labels():
    df_raw = pd.read_csv(GPS_PROCESSED_CSV)
    df_labels = pd.read_csv(ROW_LABELS_CSV)
    df = pd.merge(df_raw, df_labels[["time_s", "label"]], on="time_s", how="left")
    df["label"] = df["label"].fillna(0).astype(int)
    return df


def summarize_split(name, X, y, win_start, win_end):
    n = len(X)
    normal = int((y == 0).sum())
    anomaly = int((y == 1).sum())
    console.print(
        f"[bold cyan]{name}[/bold cyan]: {n} windows "
        f"(normal={normal}, anomaly={anomaly}) "
        f"time range: {win_start[0]:.2f}s → {win_end[-1]:.2f}s"
    )


def run_demo():
    console.rule("[bold green]PX4 Anomaly Detector - Dataset Demo")

    artifacts = load_artifacts()
    df = load_raw_and_labels()

    num_features = artifacts["X_train"].shape[2]
    model = load_model(num_features)

    summarize_split(
        "Train",
        artifacts["X_train"],
        artifacts["y_train"],
        artifacts["win_start_train"],
        artifacts["win_end_train"],
    )
    summarize_split(
        "Val",
        artifacts["X_val"],
        artifacts["y_val"],
        artifacts["win_start_val"],
        artifacts["win_end_val"],
    )
    summarize_split(
        "Test",
        artifacts["X_test"],
        artifacts["y_test"],
        artifacts["win_start_test"],
        artifacts["win_end_test"],
    )

    X_all = np.concatenate(
        [artifacts["X_train"], artifacts["X_val"], artifacts["X_test"]], axis=0
    )
    y_all = np.concatenate(
        [artifacts["y_train"], artifacts["y_val"], artifacts["y_test"]], axis=0
    )
    win_start_all = np.concatenate(
        [
            artifacts["win_start_train"],
            artifacts["win_start_val"],
            artifacts["win_start_test"],
        ],
        axis=0,
    )
    win_end_all = np.concatenate(
        [
            artifacts["win_end_train"],
            artifacts["win_end_val"],
            artifacts["win_end_test"],
        ],
        axis=0,
    )

    sort_idx = np.argsort(win_start_all)
    X_all = X_all[sort_idx]
    y_all = y_all[sort_idx]
    win_start_all = win_start_all[sort_idx]
    win_end_all = win_end_all[sort_idx]

    with torch.no_grad():
        X_torch = torch.FloatTensor(X_all.transpose(0, 2, 1))
        probs = model(X_torch).numpy().flatten()

    console.print()
    console.rule("[bold yellow]Window-by-window view")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", justify="right")
    table.add_column("t_start (s)")
    table.add_column("t_end (s)")
    table.add_column("Label")
    table.add_column("Anom prob")
    table.add_column("Dominant mode")
    table.add_column("Failsafe")
    table.add_column("Armed")

    for i in range(len(X_all)):
        t_start = float(win_start_all[i])
        t_end = float(win_end_all[i])
        label = int(y_all[i])
        prob = float(probs[i])

        rows = df[(df["time_s"] >= t_start) & (df["time_s"] <= t_end)]
        if len(rows) == 0:
            mode = "-"
            failsafe = "-"
            armed = "-"
        else:
            mode = rows["mode"].value_counts().index[0]
            failsafe = rows["failsafe"].max()
            armed = rows["armed"].max()

        label_text = "[green]NORMAL[/green]" if label == 0 else "[red]ANOMALY[/red]"
        prob_text = f"{prob:.3f}"

        table.add_row(
            str(i),
            f"{t_start:.2f}",
            f"{t_end:.2f}",
            label_text,
            prob_text,
            str(mode),
            str(int(failsafe) if failsafe != "-" else "-"),
            str(int(armed) if armed != "-" else "-"),
        )

    console.print(table)

    mean_normal = probs[y_all == 0].mean() if (y_all == 0).any() else 0.0
    mean_anom = probs[y_all == 1].mean() if (y_all == 1).any() else 0.0

    summary_text = (
        f"[bold]Summary[/bold]\n"
        f"- Total windows: {len(X_all)}\n"
        f"- Normal windows: {(y_all == 0).sum()} (mean prob={mean_normal:.3f})\n"
        f"- Anomaly windows: {(y_all == 1).sum()} (mean prob={mean_anom:.3f})\n"
        f"- Threshold idea: >0.3 could be considered suspicious in this dataset."
    )
    console.print()
    console.print(Panel(summary_text, title="Model Behaviour", style="bold blue"))


if __name__ == "__main__":
    run_demo()
