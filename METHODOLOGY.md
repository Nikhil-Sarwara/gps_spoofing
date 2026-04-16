# GPS Spoofing Detection Methodology

## Table of Contents
1. [Problem Definition](#1-problem-definition)
2. [System Architecture](#2-system-architecture)
3. [Data Collection](#3-data-collection)
4. [Data Processing Pipeline](#4-data-processing-pipeline)
5. [Automated Labeling Heuristics](#5-automated-labeling-heuristics)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine Learning Approach](#7-machine-learning-approach)
8. [Training and Evaluation](#8-training-and-evaluation)
9. [Live Inference Pipeline](#9-live-inference-pipeline)
10. [Workflow Summary](#10-workflow-summary)

---

## 1. Problem Definition

### Objective
Detect GPS spoofing attacks on PX4-powered drones in real-time using machine learning.

### Threat Model
GPS spoofing attacks involve:
- Broadcasting false GPS signals to mislead the drone's position estimation
- Gradual position drift attacks (slow injection)
- Sudden position jumps (rapid injection)
- Complete GPS takeover

### Success Criteria
- Detect spoofing within 3 seconds (30 samples at 10Hz)
- Minimize false positives during normal flight
- Real-time inference capability (< 100ms latency)

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GPS SPOOFING DETECTION SYSTEM               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     │
│  │  PX4 SITL /  │────▶│  gps_monitor │────▶│   ml.pipeline   │     │
│  │  Real Drone  │     │  (MAVLink)   │     │  (Auto-label)  │     │
│  └──────────────┘     └──────────────┘     └────────┬────────┘     │
│                                                        │              │
│                      ┌──────────────┐                  │              │
│                      │   ml/live    │◀─────────────────┘              │
│                      │  _inference  │                                │
│                      └──────┬───────┘                                │
│                             │                                        │
│                             ▼                                        │
│                      ┌──────────────┐                                │
│                      │     ui/      │                                │
│                      │   Streamlit  │                                │
│                      └──────────────┘                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| `gps_monitor/` | Collect MAVLink telemetry | pymavlink |
| `ml/scripts/` | Process & label data | pandas, numpy |
| `ml/models/` | Trained ML models | scikit-learn, PyTorch |
| `ml/live_inference.py` | Real-time detection | joblib |
| `ui/` | Monitoring dashboard | Streamlit |

---

## 3. Data Collection

### Collection Method
MAVLink protocol over UDP connection (`udp:127.0.0.1:14560`)

### Message Types Captured
| MAVLink Message | Data Captured |
|----------------|---------------|
| `GLOBAL_POSITION_INT` | lat, lon, alt, velocity, heading |
| `GPS_RAW_INT` | fix_type, satellites, eph, epv |
| `RAW_IMU` / `SCALED_IMU2` | accelerometer, gyroscope |
| `HEARTBEAT` | armed state, flight mode |
| `SYS_STATUS` | battery voltage, remaining % |
| `VFR_HUD` | airspeed, groundspeed |

### Sampling Rate
- **10 Hz** (100ms intervals) - ensures consistency with ML training
- GPS logs saved as CSV to `gps_logs/raw/`

### Raw Data Schema
```csv
time_s,lat_deg,lon_deg,alt_m,rel_alt_m,vel_m_s,hdg_deg,
fix_type,satellites_visible,eph_m,epv_m,
ax_mps2,ay_mps2,az_mps2,gx_radps,gy_radps,gz_radps,
battery_voltage,battery_remaining_pct,armed,mode,failsafe,
connection_ok,last_update_iso
```

---

## 4. Data Processing Pipeline

### Step 1: Cleaning (`01_clean_log.py`)

| Operation | Description |
|-----------|-------------|
| Type conversion | Parse all columns as numeric |
| Missing data | Drop rows without `time_s` or critical GPS columns |
| Fix type filter | Keep only rows with `fix_type >= 3` (3D fix) |
| Connection filter | Keep only `connection_ok == 1` |
| Duplicate removal | Drop exact duplicate rows |
| Stale detection | Mark repeated identical readings |
| Time gap filter | Remove gaps > 5 seconds |

### Step 2: Automated Labeling (`02_auto_label.py`)

See Section 5 for heuristic rules.

### Step 3: Window Creation (`03_make_windows.py`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Window length | 30 samples | 3 seconds (enough for pattern recognition) |
| Stride | 15 samples | 50% overlap for smooth detection |
| Label rule | `anomaly_ratio > 0.5` | Window is anomaly if >50% rows are anomaly |

### Step 4: Feature Engineering

Transform GPS coordinates to local ENU frame:
```python
x_m = (lat - first_lat) * 111320  # meters
y_m = (lon - first_lon) * cos(first_lat) * 111320  # meters
```

### Step 5: Normalization

StandardScaler fitted on training data only:
```python
scaler.fit(X_train_flat)
X_train_scaled = scaler.transform(X_train_flat)
```

---

## 5. Automated Labeling Heuristics

The auto-labeler detects anomalies using multiple heuristics:

### 5.1 Position Jump Detection
```python
speed = haversine_distance(lat1, lon1, lat2, lon2) / dt
if speed > max_normal_speed_ms (30.0):
    label = ANOMALY
```

### 5.2 Speed Anomaly Detection
```python
vel_change = |vel[i] - vel[i-1]|
if vel_change > sudden_speed_change_ms (10.0):
    label = ANOMALY

accel = |vel.diff() / dt.diff()|
if accel > max_normal_accel_ms2 (15.0):
    label = ANOMALY
```

### 5.3 GPS Quality Degradation
```python
if satellites_visible < min_satellites (6):
    label = ANOMALY
if eph > max_eph_m (5.0):  # Horizontal error
    label = ANOMALY
if epv > max_epv_m (10.0):  # Vertical error
    label = ANOMALY
```

### 5.4 Stale Data Detection
```python
# Consecutive identical readings
stale = (row == row.shift(1)).all(axis=1)
# Long stale runs indicate GPS issues
if stale_run >= stale_threshold (2):
    label = ANOMALY
```

### 5.5 Mode Anomaly Detection
```python
if failsafe == 1:
    label = ANOMALY
if mode.startswith("Mode(0x"):  # Unknown mode
    label = ANOMALY
```

### Confidence Scoring
Each anomaly row stores the reason:
```
"pos_jump:45.2m/s; high_eph;"
```

---

## 6. Feature Engineering

### Input Features (16 total)

| Feature | Description | Units |
|---------|-------------|-------|
| `x_m` | East position (local frame) | meters |
| `y_m` | North position (local frame) | meters |
| `alt_m` | Altitude MSL | meters |
| `rel_alt_m` | Relative altitude | meters |
| `vel_m_s` | Ground speed | m/s |
| `hdg_deg` | Heading | degrees |
| `fix_type` | GPS fix quality (0-5) | enum |
| `satellites_visible` | Number of satellites | count |
| `eph_m` | Horizontal position error | meters |
| `epv_m` | Vertical position error | meters |
| `battery_voltage` | Battery voltage | volts |
| `battery_remaining_pct` | Battery remaining | % |
| `armed` | Armed state | bool |
| `failsafe` | Failsafe active | bool |
| `connection_ok` | MAVLink connection | bool |
| `is_stale_repeat` | Stale data flag | bool |

### Feature Rationale

| Attack Signature | Detected By |
|-----------------|-------------|
| Sudden position jump | `x_m`, `y_m`, `vel_m_s` |
| Gradual drift | `x_m`, `y_m` trend analysis |
| Satellite blackout | `satellites_visible`, `fix_type` |
| Accuracy degradation | `eph_m`, `epv_m` |
| Signal stalling | `is_stale_repeat` |
| Failsafe trigger | `failsafe` flag |

---

## 7. Machine Learning Approach

### Models Implemented

#### 7.1 Random Forest (Primary)
```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
```

**Why RF?**
- Fast inference (< 10ms)
- Handles imbalanced data well
- No need for feature scaling at inference
- Interpretable feature importance

#### 7.2 1D CNN (Alternative)
```python
WindowCNN:
├── Conv1D(16, 32, kernel=3)
├── ReLU
├── Conv1D(32, 64, kernel=3)
├── ReLU
├── AdaptiveAvgPool1d(1)
├── Linear(64, 32)
├── Dropout(0.3)
├── Linear(32, 1)
└── Sigmoid
```

**Why CNN?**
- Captures temporal patterns
- Handles variable-length windows

### Window Processing for RF
```python
# Flatten window for RF
X_window: (30, 16) → X_flat: (1, 480)
# Apply pre-fitted scaler
X_scaled = scaler.transform(X_flat)
# Predict
proba = model.predict_proba(X_scaled)[0, 1]
```

---

## 8. Training and Evaluation

### Data Split
```
Train: 70% | Val: 15% | Test: 15%
```

### Training Configuration
```yaml
training:
  model_type: "rf"
  n_estimators: 100
  random_state: 42

windows:
  length: 30
  stride: 15
```

### Evaluation Metrics
| Metric | Purpose |
|--------|---------|
| Precision | Minimize false alarms |
| Recall | Detect actual attacks |
| F1-Score | Balance precision/recall |
| Confusion Matrix | Visualize errors |

### Current Dataset Status
From `dataset_info.json`:
- **Total windows**: 122
- **Anomaly windows**: 5 (4.1%)
- **Highly imbalanced** - needs more spoofing data

---

## 9. Live Inference Pipeline

### Architecture
```
MAVLink UDP ──▶ gps_monitor ──▶ live_inference ──▶ CSV Log
                    │               │
                    │          Anomaly Alert
                    │               │
                    └───────────────▼────▶ Streamlit UI
```

### Inference Loop (`live_inference.py`)
```python
detector = LiveAnomalyDetector(model_path, scaler_path)
detector.buffer = deque(maxlen=30)  # Sliding window

while True:
    msg = master.recv_match(type="GLOBAL_POSITION_INT")
    features = detector.extract_features(msg.to_dict())
    detector.buffer.append(features)

    if len(buffer) == 30:
        proba = detector.predict()
        log_result(proba)
```

### Detection Threshold
```python
if proba > 0.5:
    status = "ANOMALY"
    alert()
```

---

## 10. Workflow Summary

### Development Workflow
```
┌─────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT PHASE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. SIMULATION                                               │
│     cd PX4-Autopilot                                        │
│     source .px4-venv/bin/activate                           │
│     PX4_GZ_WORLD=baylands make px4_sitl gz_x500           │
│                                                              │
│  2. DATA COLLECTION                                         │
│     cd project                                              │
│     python -m gps_monitor.main                              │
│     → Logs to gps_logs/raw/                                │
│                                                              │
│  3. PIPELINE (Automated)                                    │
│     python -m ml.pipeline full                              │
│     → Cleans → Auto-labels → Windows → Trains               │
│                                                              │
│  4. EVALUATION                                              │
│     python ml/demo_dataset.py                                │
│     → Confusion matrices, per-window analysis               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   DEPLOYMENT PHASE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  5. LIVE INFERENCE                                          │
│     python ml/live_inference.py                             │
│     → Real-time detection with alerts                       │
│                                                              │
│  6. MONITORING                                              │
│     cd ui && streamlit run app.py                          │
│     → Dashboard with live metrics                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Configuration-Driven
All thresholds and paths in `config/pipeline.yaml`:
```yaml
auto_label:
  max_normal_speed_ms: 30.0
  min_satellites: 6
  max_eph_m: 5.0

windows:
  length: 30
  stride: 15

training:
  n_estimators: 100
```

---

## Future Enhancements

1. **Synthetic Spoofing Data**
   - Use PX4 SITL with injected GPS offsets
   - Generate labeled data without real attacks

2. **Online Learning**
   - Update model with confirmed true/false positives
   - Adapt to environment-specific patterns

3. **Multi-Sensor Fusion**
   - Integrate vision-based position estimation
   - Compare GPS with optical flow

4. **Attack Classification**
   - Distinguish spoofing types (gradual vs sudden)
   - Estimate spoofing direction/offset

---

## References

- MAVLink Protocol: https://mavlink.io/en/
- PX4 Flight Stack: https://px4.io/
- scikit-learn: https://scikit-learn.org/
- Haversine Formula: Calculate distance between GPS coordinates
