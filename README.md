# GPS Spoofing Detection System

A machine learning-based GPS anomaly/spoofing detection system for PX4 drones.

## Project Structure

```
gps_spoofing/
├── project/
│   ├── config/
│   │   └── pipeline.yaml       # Centralized configuration
│   ├── gps_monitor/            # MAVLink telemetry collector
│   │   ├── config.py           # Connection settings
│   │   ├── main.py             # Entry point
│   │   ├── mavlink_client.py   # MAVLink communication
│   │   ├── state_model.py      # Telemetry state
│   │   └── console_ui.py      # Terminal UI
│   ├── ml/                    # ML pipeline & anomaly detection
│   │   ├── config/             # ML configs
│   │   ├── scripts/            # Pipeline scripts
│   │   │   ├── 01_clean_log.py       # Data cleaning
│   │   │   ├── 02_auto_label.py      # Auto-labeling (heuristics)
│   │   │   ├── 03_make_windows.py    # Create ML windows
│   │   │   └── 04_train_baseline.py  # Train models
│   │   ├── artifacts/          # Dataset & scaler
│   │   ├── models/            # Trained models
│   │   ├── pipeline.py        # Unified pipeline runner
│   │   ├── live_inference.py  # Real-time inference
│   │   └── demo_dataset.py    # Dataset visualization
│   ├── ui/                     # Streamlit web dashboard
│   └── tests/                  # Unit tests
├── PX4-Autopilot/              # PX4 firmware submodule
└── README.md
```

## Quick Start

### 1. Collect GPS Data

```bash
cd project
python -m gps_monitor.main
```

### 2. Automated Pipeline (Recommended)

Run the entire pipeline with one command:

```bash
# Process all raw logs
python -m ml.pipeline process-batch

# Train models
python -m ml.pipeline train

# Run full pipeline
python -m ml.pipeline full
```

### 3. Or Run Steps Individually

```bash
# Clean data
python ml/scripts/01_clean_log.py gps_logs/raw/your_log.csv

# Auto-label anomalies (heuristic-based)
python ml/scripts/02_auto_label.py gps_logs/processed/your_log_cleaned.csv

# Create ML windows
python ml/scripts/03_make_windows.py gps_logs/processed/row_labels_auto.csv

# Train model
python ml/scripts/04_train_baseline.py
```

### 4. Run Inference

```bash
# Live inference
python ml/live_inference.py

# Dashboard
cd ui && streamlit run app.py
```

## Automated Labeling

The `02_auto_label.py` script uses heuristics to automatically detect spoofing/anomalies:

| Detection Method | Description |
|-----------------|-------------|
| Position Jumps | Sudden unrealistic movement (>30 m/s) |
| Speed Anomalies | Velocity inconsistencies |
| GPS Quality | Low satellites, high eph/epv |
| Stale Data | Repeated identical readings |
| Mode Anomalies | Failsafe, hex mode codes |

Configure thresholds in `config/pipeline.yaml`.

## Configuration

All settings are in `config/pipeline.yaml`:

```yaml
data:
  raw_dir: "gps_logs/raw"
  processed_dir: "gps_logs/processed"
  artifacts_dir: "ml/artifacts"
  models_dir: "ml/models"

auto_label:
  enabled: true
  max_normal_speed_ms: 30.0
  min_satellites: 6

windows:
  length: 30      # Samples (3s @ 10Hz)
  stride: 15     # 50% overlap
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  gps_monitor    │────▶│  ml.pipeline    │────▶│     ui/         │
│  (MAVLink UDP) │     │  (Auto-label)   │     │  (Streamlit)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
  Raw CSV logs          Labeled windows          Live dashboard
                         Train RF/CNN            Anomaly alerts
```
