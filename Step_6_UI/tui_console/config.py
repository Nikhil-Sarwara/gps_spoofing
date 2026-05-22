# TUI Configuration - Component definitions
import os
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
PID_DIR = PROJECT_ROOT / ".pids"
PID_DIR.mkdir(exist_ok=True)
LOG_DIR = PROJECT_ROOT / "Step_8_Archive" / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "Step_5_Data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Step_5_Data" / "processed"
MODELS_DIR = PROJECT_ROOT / "Step_5_Data" / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "Step_5_Data" / "artifacts"

# VENV paths
VENV_BIN = PROJECT_ROOT / "venv" / "bin"

# Available Gazebo simulation worlds
GZ_WORLDS = {
    "windy":       {"label": "🌬️  Windy          (wind disturbance)",   "terrain": "flat"},
    "default":     {"label": "🏙️  Default         (flat open ground)",   "terrain": "flat"},
    "baylands":    {"label": "🌊  Baylands        (bay / sea terrain)",   "terrain": "sea"},
    "forest":      {"label": "🌲  Forest          (dense trees)",         "terrain": "mountain"},
    "lawn":        {"label": "🌿  Lawn            (suburban grass)",      "terrain": "flat"},
    "aruco":       {"label": "🔲  ArUco           (indoor markers)",      "terrain": "flat"},
    "kthspacelab": {"label": "🏢  KTH Space Lab   (indoor lab)",          "terrain": "flat"},
    "underwater":  {"label": "🐠  Underwater      (underwater env)",      "terrain": "sea"},
}

# Default world on first launch
DEFAULT_GZ_WORLD = "windy"

# Component configurations - ALL open in new terminal window
COMPONENTS = {
    "dashboard": {
        "name": "Streamlit Dashboard",
        "description": "Web UI (opens browser)",
        "command": "PYTHONPATH=. ./venv/bin/streamlit run Step_6_UI/web_ui/app.py --server.port 8501",
        "pid_file": PID_DIR / "dashboard.pid",
        "log_file": LOG_DIR / "dashboard.log",
        "workdir": PROJECT_ROOT,
        "spawn_method": "terminal",
        "open_browser": True,
    },
    "gps_monitor": {
        "name": "GPS Monitor",
        "description": "MAVLink telemetry collector",
        "command": "PYTHONPATH=. ./venv/bin/python -m Step_2_Monitoring.main",
        "pid_file": PID_DIR / "gps_monitor.pid",
        "log_file": LOG_DIR / "gps_monitor.log",
        "workdir": PROJECT_ROOT,
        "spawn_method": "terminal",
    },
    "live_inference": {
        "name": "ML Live Inference",
        "description": "Real-time anomaly detection",
        "command": "PYTHONPATH=. ./venv/bin/python -m Step_4_Detection.live_inference --connection udp:127.0.0.1:14562 --model-path Step_5_Data/models/rf_model.pkl --scaler-path Step_5_Data/artifacts/scaler.pkl --terrain-models-dir Step_5_Data/models/terrain --preferred-model-type rf --anomaly-threshold 0.6",
        "pid_file": PID_DIR / "live_inference.pid",
        "log_file": LOG_DIR / "live_inference.log",
        "workdir": PROJECT_ROOT,
        "spawn_method": "terminal",
    },
    "px4_sim": {
        "name": "PX4 SITL Simulation",
        "description": "Gazebo drone simulation",
        "command": "./start_px4.sh",
        "pid_file": PID_DIR / "px4_sim.pid",
        "log_file": LOG_DIR / "px4_sim.log",
        "workdir": PROJECT_ROOT,
        "spawn_method": "terminal",
    },
}

# GPS Spoofing attack presets — launched from the local TUI subprocess
SPOOF_ATTACKS = {
    "drift": {
        "name": "Drift Attack  (3 m/s, 30 s)",
        "command": "./venv/bin/python gps_spoofer.py --mode drift --intensity 3.0 --duration 30",
    },
    "ramp": {
        "name": "Ramp Drift    (0 -> 5 m/s, 60 s)",
        "command": "./venv/bin/python gps_spoofer.py --mode ramp_drift --intensity 5.0 --duration 60",
    },
    "jump_drift": {
        "name": "Jump + Drift  (15 m jump, 2 m/s drift)",
        "command": "./venv/bin/python gps_spoofer.py --mode jump_drift --intensity 2.0 --duration 45",
    },
    "static": {
        "name": "Static Freeze (500 m offset, 20 s)",
        "command": "./venv/bin/python gps_spoofer.py --mode static --intensity 500 --duration 20",
    },
    "teleport": {
        "name": "Teleport      (2 km jump, 15 s)",
        "command": "./venv/bin/python gps_spoofer.py --mode teleport --intensity 2000 --duration 15",
    },
    "noise": {
        "name": "Noise Inject  (50 m stddev, 60 s)",
        "command": "./venv/bin/python gps_spoofer.py --mode noise --intensity 50.0 --duration 60",
    },
}

# ML Pipeline configurations
ML_PIPELINE = {
    "process": {
        "name": "Process Data",
        "description": "Clean raw GPS logs",
        "command": "PYTHONPATH=. ./venv/bin/python -m Step_4_Detection.pipeline process-batch",
    },
    "train": {
        "name": "Train Model",
        "description": "Train ML model",
        "command": "PYTHONPATH=. ./venv/bin/python -m Step_4_Detection.pipeline train",
    },
    "full": {
        "name": "Full Pipeline",
        "description": "Process + Train",
        "command": "PYTHONPATH=. ./venv/bin/python -m Step_4_Detection.pipeline full",
    },
    "train_terrain": {
        "name": "Train Terrain Models",
        "description": "Train per-terrain RF+CNN models (Step 2)",
        "command": "PYTHONPATH=. ./venv/bin/python Step_4_Detection/scripts/05_train_terrain_models.py --processed-dirs Step_5_Data/processed --models-dir Step_5_Data/models/terrain --artifacts-dir Step_5_Data/artifacts",
    },
    "full_terrain": {
        "name": "Full Terrain Pipeline",
        "description": "Process + label terrain + train all models",
        "command": "PYTHONPATH=. ./venv/bin/python -m Step_4_Detection.pipeline full-terrain",
    },
}


def get_terrain_model_status() -> dict:
    """
    Returns a dict summarising trained terrain models.
    Keys: terrain names. Values: dict with 'status', 'rf_val_f1', 'cnn_val_f1'.
    Returns empty dict if terrain_training_report.json not found.
    """
    import json
    report_path = PROJECT_ROOT / "Step_5_Data" / "models" / "terrain" / "terrain_training_report.json"
    if not report_path.exists():
        return {}
    try:
        with open(report_path) as f:
            report = json.load(f)
        return report.get("terrain_summary", {})
    except Exception:
        return {}


def get_component(component_id):
    return COMPONENTS.get(component_id)


def get_all_components():
    return COMPONENTS


def get_component_ids():
    return list(COMPONENTS.keys())


def get_dashboard_url():
    return "http://localhost:8501"
