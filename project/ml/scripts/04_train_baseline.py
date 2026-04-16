from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001


class WindowCNN(nn.Module):
    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


def train_rf(X_train, y_train, X_val, y_val, models_dir, n_estimators=100, random_state=42):
    print("Training Random Forest...")

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced"
    )
    rf.fit(X_train_flat, y_train)

    train_pred = rf.predict(X_train_flat)
    val_pred = rf.predict(X_val_flat)

    print("RF Results:")
    print("Train:", classification_report(y_train, train_pred, zero_division=0))
    print("Val:  ", classification_report(y_val, val_pred, zero_division=0))

    with open(models_dir / "rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)

    return rf, train_pred, val_pred


def train_cnn(X_train, y_train, X_val, y_val, X_test, y_test, num_features, models_dir):
    print("\nTraining 1D CNN...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train.transpose(0, 2, 1)),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val.transpose(0, 2, 1)),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test.transpose(0, 2, 1)),
        torch.FloatTensor(y_test).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = WindowCNN(num_features).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_pred = []
        val_true = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
                val_pred.extend((pred.cpu().numpy() > 0.5).astype(int).flatten())
                val_true.extend(batch_y.cpu().numpy().astype(int).flatten())

        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: "
                f"train_loss={train_loss / max(len(train_loader), 1):.4f}, "
                f"val_loss={val_loss:.4f}"
            )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()

    val_pred = []
    val_true = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            val_pred.extend((pred.cpu().numpy() > 0.5).astype(int).flatten())
            val_true.extend(batch_y.numpy().astype(int).flatten())

    test_pred = []
    test_true = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)
            test_pred.extend((pred.cpu().numpy() > 0.5).astype(int).flatten())
            test_true.extend(batch_y.numpy().astype(int).flatten())

    print("\nCNN Results:")
    print("Val: ", classification_report(val_true, val_pred, zero_division=0))
    print("Test:", classification_report(test_true, test_pred, zero_division=0))

    torch.save(model.state_dict(), models_dir / "cnn_model.pth")

    return model, np.array(val_pred), np.array(test_pred)


def plot_confusion(y_true, y_pred, title, filename, models_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(models_dir / filename)
    plt.close()

    print(f"\n{title} confusion matrix:")
    print(cm)


def main(
    artifacts_dir: Union[str, Path],
    models_dir: Union[str, Path],
    training_config: Optional[dict] = None,
) -> dict:
    """
    Train ML models for GPS spoofing detection.

    Args:
        artifacts_dir: Directory containing dataset.npz and feature_names.json
        models_dir: Output directory for trained models
        training_config: Optional training configuration

    Returns:
        Dict with paths to trained models
    """
    training_config = training_config or {}

    artifacts_dir = Path(artifacts_dir)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = artifacts_dir / "dataset.npz"
    features_path = artifacts_dir / "feature_names.json"

    data = np.load(dataset_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    with open(features_path) as f:
        feature_names = json.load(f)

    num_features = X_train.shape[2]

    print(
        f"Dataset loaded: train={X_train.shape}, "
        f"val={X_val.shape}, test={X_test.shape}"
    )
    print(f"Features ({num_features}): {feature_names[:5]}...")

    n_estimators = training_config.get("n_estimators", 100)
    random_state = training_config.get("random_state", 42)

    rf, rf_train_pred, rf_val_pred = train_rf(
        X_train, y_train, X_val, y_val, models_dir, n_estimators, random_state
    )

    cnn_model, cnn_val_pred, cnn_test_pred = train_cnn(
        X_train, y_train, X_val, y_val, X_test, y_test, num_features, models_dir
    )

    plot_confusion(y_train, rf_train_pred, "RF Train", "rf_train_cm.png", models_dir)
    plot_confusion(y_val, rf_val_pred, "RF Val", "rf_val_cm.png", models_dir)
    plot_confusion(y_val, cnn_val_pred, "CNN Val", "cnn_val_cm.png", models_dir)
    plot_confusion(y_test, cnn_test_pred, "CNN Test", "cnn_test_cm.png", models_dir)

    print("\n✅ Training complete!")
    print("Models saved:")
    print(f"- {models_dir / 'rf_model.pkl'}")
    print(f"- {models_dir / 'cnn_model.pth'}")

    return {
        "rf_model": str(models_dir / "rf_model.pkl"),
        "cnn_model": str(models_dir / "cnn_model.pth"),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="ml/artifacts", help="Artifacts directory")
    parser.add_argument("--models-dir", default="ml/models", help="Models output directory")
    args = parser.parse_args()
    main(args.artifacts_dir, args.models_dir)
