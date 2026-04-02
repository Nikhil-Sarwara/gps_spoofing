# GPS Spoofing Detection System

A machine learning-based GPS anomaly/spoofing detection system for PX4 drones.

## Project Structure

```
gps_spoofing/
├── project/
│   ├── gps_monitor/     # MAVLink telemetry collector
│   ├── ml/              # ML pipeline & anomaly detection
│   ├── ui/              # Streamlit web dashboard
│   └── tests/           # Unit tests
├── PX4-Autopilot/       # PX4 firmware submodule
└── README.md
```

## Quick Start

```bash
# 1. Collect GPS data
cd project && python -m gps_monitor.main

# 2. Clean and process data
python ml/scripts/01_clean_log.py
python ml/scripts/02_label_segments.py
python ml/scripts/03_make_windows.py

# 3. Train model
python ml/scripts/04_train_baseline.py

# 4. Run live inference
python ml/live_inference.py

# 5. Start dashboard
cd ui && streamlit run app.py
```

## Architecture

1. **gps_monitor**: Captures MAVLink telemetry via UDP @ 10Hz
2. **ml**: Trains ML models, performs sliding-window anomaly detection
3. **ui**: Real-time Streamlit dashboard for monitoring
