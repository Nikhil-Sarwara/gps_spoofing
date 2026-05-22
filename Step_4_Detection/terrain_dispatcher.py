"""terrain_dispatcher.py — Step 3: TerrainModelDispatcher

Dynamically routes GPS spoofing inference to the best terrain-specific model
(flat / mountain / sea) based on the UAV's current GPS position, falling back
to the global RF model when a terrain-specific model is unavailable.

Usage:
    from terrain_dispatcher import TerrainModelDispatcher

    dispatcher = TerrainModelDispatcher(
        terrain_models_dir="Step_5_Data/models/terrain",
        global_model_path="Step_5_Data/models/rf_model.pkl",
        global_scaler_path="Step_5_Data/artifacts/scaler.pkl",
    )
    dispatcher.update_terrain(lat, lon, alt_m)   # called per GPS fix
    dispatcher.push_features(feat_vec)           # (34,) feature vector
    result = dispatcher.predict()                # dict or None
"""
from __future__ import annotations

import json
import logging
import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_LEN      = 30
TERRAINS        = ["flat", "mountain", "sea"]
DEBOUNCE_COUNT = 5      # consecutive identical terrain calls to confirm a switch
EARTH_RADIUS_M  = 6_371_000.0
NUM_FEATURES    = 34

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WindowCNN — copied verbatim from 04_train_baseline.py (self-contained)
# ---------------------------------------------------------------------------
class WindowCNN(nn.Module):
    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.conv1   = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(32, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc2(x))


# ---------------------------------------------------------------------------
# TerrainClassifier import — graceful fallback if not importable
# ---------------------------------------------------------------------------
try:
    from terrain_classifier import TerrainClassifier
except ImportError:
    TerrainClassifier = None
    logger.warning("TerrainClassifier not importable — terrain will stay fixed.")


# ---------------------------------------------------------------------------
# TerrainModelDispatcher
# ---------------------------------------------------------------------------
class TerrainModelDispatcher:
    """Wraps terrain classifier + all terrain-specific RF/CNN models.

    On every GPS fix:
      1. Call update_terrain(lat, lon, alt_m)  — debounced terrain switch
      2. Call push_features(feat_vec)           — append (34,) row to buffer
      3. Call predict()                          — returns dict or None
    """

    def __init__(
        self,
        terrain_models_dir,
        global_model_path,
        global_scaler_path,
        preferred_model_type="rf",
        anomaly_threshold=0.5,
    ):
        self.terrain_models_dir   = Path(terrain_models_dir)
        self.global_model_path    = Path(global_model_path)
        self.global_scaler_path   = Path(global_scaler_path)
        self.preferred_model_type = preferred_model_type.lower()
        self.anomaly_threshold    = anomaly_threshold

        # Shared rolling buffer — NOT cleared on terrain switch
        self.buffer = deque(maxlen=WINDOW_LEN)

        # Debounce state
        self._debounce_candidate = ""
        self._debounce_count     = 0

        # Active model slots (set by _set_active_terrain)
        self.active_terrain_label = "global_fallback"
        self.active_scaler        = None
        self.active_model_type    = "global_fallback"
        self.active_rf_model      = None
        self.active_cnn_model     = None
        self.active_info          = {}
        self.is_fallback          = True
        self._switched            = False

        # Model registry
        self._models = {}

        # Global fallback
        self._global_model  = None
        self._global_scaler = None

        # Recommended default from training report
        self._recommended_default = "flat"

        # ── Area-specific model (Baylands square) ────────────────────
        self._baylands_model = None
        self._baylands_scaler = None
        self._baylands_bbox = None
        _baylands_dir = self.terrain_models_dir.parent / "area_specific"
        if (_baylands_dir / "rf_model.pkl").exists():
            try:
                self._baylands_model = joblib.load(_baylands_dir / "rf_model.pkl")
                self._baylands_scaler = joblib.load(_baylands_dir / "scaler.pkl")
                with open(_baylands_dir / "model_info.json") as f:
                    info = json.load(f)
                self._baylands_bbox = info.get("bbox")
                logger.info("Area-specific model loaded (Baylands: %s)", self._baylands_bbox)
            except Exception as e:
                logger.warning("Area-specific model load failed: %s", e)
        self.home_gps        = None   # (lat, lon) of first GPS fix
        self.last_msg        = None   # last GLOBAL_POSITION_INT
        self.last_attitude   = None   # last ATTITUDE msg
        self.last_estimator  = None   # last ESTIMATOR_STATUS msg
        self.last_vibration  = None   # last VIBRATION msg
        self.last_health     = None   # last HEARTBEAT msg
        self.last_sys_status = None   # last SYS_STATUS msg
        self._stale_count    = 0

        # Terrain classifier
        self._classifier = TerrainClassifier() if TerrainClassifier is not None else None

        # ── Smoothing & ground-gate state ────────────────────────────
        self._ema_alpha      = 0.3
        self._ema_proba      = None
        self._stable_window  = deque(maxlen=15)

        # --- Load everything ---
        self._load_report()
        self._load_all_terrain_models()
        self._load_global_fallback()
        self._set_active_terrain(self._recommended_default, force=True)

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _load_report(self):
        report_path = self.terrain_models_dir / "terrain_training_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            self._recommended_default = report.get("recommended_default_model", "flat")
            logger.info("Terrain report loaded. Recommended default: %s",
                        self._recommended_default)
        else:
            logger.warning("terrain_training_report.json not found — defaulting to 'flat'")

    def _load_all_terrain_models(self):
        for terrain in TERRAINS:
            entry = {
                "rf":          None,
                "cnn":         None,
                "scaler":      None,
                "info":        {},
                "available":   False,
                "active_type": None,
            }
            terrain_dir = self.terrain_models_dir / terrain
            info_path   = terrain_dir / "dataset_info.json"

            if not info_path.exists():
                logger.warning("[%s] dataset_info.json missing — marking unavailable", terrain)
                self._models[terrain] = entry
                continue

            with open(info_path) as f:
                info = json.load(f)
            entry["info"] = info

            if info.get("status") == "insufficient_data":
                logger.warning("[%s] Insufficient data (n=%s) — skipping",
                               terrain, info.get("n_windows", 0))
                self._models[terrain] = entry
                continue

            # Scaler
            scaler_path = terrain_dir / "scaler.pkl"
            if not scaler_path.exists():
                logger.warning("[%s] scaler.pkl missing — marking unavailable", terrain)
                self._models[terrain] = entry
                continue
            entry["scaler"] = joblib.load(scaler_path)

            # RF
            rf_path = terrain_dir / "rf_model.pkl"
            if rf_path.exists():
                entry["rf"] = joblib.load(rf_path)
                logger.info("[%s] RF model loaded (val F1=%.3f)",
                            terrain, info.get("rf_val_f1") or 0.0)

            # CNN (state-dict only)
            cnn_path = terrain_dir / "cnn_model.pth"
            if cnn_path.exists() and info.get("cnn_status") == "trained":
                try:
                    cnn = WindowCNN(num_features=NUM_FEATURES)
                    cnn.load_state_dict(
                        torch.load(cnn_path, map_location="cpu", weights_only=True)
                    )
                    cnn.eval()
                    entry["cnn"] = cnn
                    logger.info("[%s] CNN model loaded (val F1=%.3f)",
                                terrain, info.get("cnn_val_f1") or 0.0)
                except Exception as exc:
                    logger.warning("[%s] CNN load failed (%s) — will use RF", terrain, exc)

            # Decide active model type for this terrain
            if self.preferred_model_type == "cnn" and entry["cnn"] is not None:
                entry["active_type"] = "cnn"
            elif entry["rf"] is not None:
                if self.preferred_model_type == "cnn":
                    logger.warning("[%s] CNN unavailable — falling back to RF", terrain)
                entry["active_type"] = "rf"
            else:
                logger.warning("[%s] No trained model available", terrain)
                self._models[terrain] = entry
                continue

            entry["available"] = True
            self._models[terrain] = entry

    def _load_global_fallback(self):
        if self.global_model_path.exists() and self.global_scaler_path.exists():
            self._global_model  = joblib.load(self.global_model_path)
            self._global_scaler = joblib.load(self.global_scaler_path)
            logger.info("Global fallback RF loaded from %s", self.global_model_path)
        else:
            logger.warning("Global fallback model not found at %s",
                           self.global_model_path)

    def _set_active_terrain(self, terrain, force=False):
        """Swap active model/scaler to terrain. Walks fallback chain if needed."""
        entry = self._models.get(terrain, {})

        if entry.get("available"):
            self.active_terrain_label = terrain
            self.active_scaler        = entry["scaler"]
            self.active_model_type    = entry["active_type"]
            self.active_rf_model      = entry["rf"]
            self.active_cnn_model     = entry["cnn"]
            self.active_info          = entry["info"]
            self.is_fallback          = False
            if not force:
                logger.info("Terrain switched to '%s' (model=%s)",
                            terrain, self.active_model_type)
            return

        # Fallback 1: recommended_default
        default = self._recommended_default
        if default != terrain and self._models.get(default, {}).get("available"):
            logger.warning("[%s] model unavailable — using recommended default '%s'",
                           terrain, default)
            self._set_active_terrain(default, force=force)
            self.is_fallback = True
            return

        # Fallback 2: any other available terrain
        for t in TERRAINS:
            if t != terrain and self._models.get(t, {}).get("available"):
                logger.warning("[%s] model unavailable — using '%s'", terrain, t)
                self._set_active_terrain(t, force=force)
                self.is_fallback = True
                return

        # Fallback 3: global RF
        if self._global_model is not None:
            logger.warning(
                "[%s] No terrain model available — using global RF fallback", terrain
            )
            self.active_terrain_label = "global_fallback"
            self.active_scaler        = self._global_scaler
            self.active_model_type    = "global_rf"
            self.active_rf_model      = self._global_model
            self.active_cnn_model     = None
            self.active_info          = {}
            self.is_fallback          = True
            return

        raise RuntimeError(
            "No terrain model and no global fallback found. "
            "Check Step_5_Data/models/terrain/ and Step_5_Data/models/rf_model.pkl."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Feature extraction  (ported from LiveAnomalyDetector)
    # ------------------------------------------------------------------
    def _gps_to_local(self, lat, lon):
        """Convert absolute GPS to local X/Y metres relative to home."""
        if self.home_gps is None:
            self.home_gps = (lat, lon)
            return 0.0, 0.0
        d_lat = np.radians(lat - self.home_gps[0]) * EARTH_RADIUS_M
        d_lon = (
            np.radians(lon - self.home_gps[1])
            * EARTH_RADIUS_M
            * np.cos(np.radians(self.home_gps[0]))
        )
        return d_lat, d_lon

    def extract_features(
        self,
        gps_msg,
        attitude_msg=None,
        estimator_msg=None,
        health_msg=None,
        vibration_msg=None,
    ) -> np.ndarray:
        """Extract a (34,) float32 feature vector from MAVLink messages."""
        import math
        from pymavlink import mavutil as _mavu

        # ── GPS / position ───────────────────────────────────
        lat_raw   = gps_msg.lat / 1e7
        lon_raw   = gps_msg.lon / 1e7
        x_m, y_m  = self._gps_to_local(lat_raw, lon_raw)
        alt_m     = gps_msg.alt / 1000.0
        rel_alt_m = gps_msg.relative_alt / 1000.0
        vel_m_s   = (gps_msg.vx**2 + gps_msg.vy**2 + gps_msg.vz**2)**0.5 / 100.0
        hdg_deg   = gps_msg.hdg / 100.0 if gps_msg.hdg != 65535 else 0.0
        fix_type  = getattr(gps_msg, "fix_type", 3)
        sats      = getattr(gps_msg, "satellites_visible", 10)
        eph_m     = getattr(gps_msg, "eph", 0) / 100.0 if hasattr(gps_msg, "eph") else 0.0
        epv_m     = getattr(gps_msg, "epv", 0) / 100.0 if hasattr(gps_msg, "epv") else 0.0

        # ── Attitude ─────────────────────────────────────────
        roll_deg = pitch_deg = yaw_deg = 0.0
        rollspeed = pitchspeed = yawspeed = 0.0
        _att = attitude_msg or self.last_attitude
        if _att is not None:
            roll_deg   = math.degrees(getattr(_att, "roll",       0))
            pitch_deg  = math.degrees(getattr(_att, "pitch",      0))
            yaw_deg    = math.degrees(getattr(_att, "yaw",        0))
            rollspeed  = getattr(_att, "rollspeed",  0.0)
            pitchspeed = getattr(_att, "pitchspeed", 0.0)
            yawspeed   = getattr(_att, "yawspeed",   0.0)

        # ── Vibration ──────────────────────────────────────
        vib_x = vib_y = vib_z = 0.0
        clipping_0 = clipping_1 = clipping_2 = 0
        _vib = vibration_msg or self.last_vibration
        if _vib is not None:
            vib_x      = getattr(_vib, "vibration_x", 0.0)
            vib_y      = getattr(_vib, "vibration_y", 0.0)
            vib_z      = getattr(_vib, "vibration_z", 0.0)
            clipping_0 = getattr(_vib, "clipping_0",  0)
            clipping_1 = getattr(_vib, "clipping_1",  0)
            clipping_2 = getattr(_vib, "clipping_2",  0)

        # ── Estimator ──────────────────────────────────────
        vel_ratio = pos_horiz_ratio = pos_vert_ratio = 0.0
        vel_innov = pos_horiz_innov = pos_vert_innov = 0.0
        _est = estimator_msg or self.last_estimator
        if _est is not None:
            vel_ratio       = getattr(_est, "vel_ratio",       0.0)
            pos_horiz_ratio = getattr(_est, "pos_horiz_ratio", 0.0)
            pos_vert_ratio  = getattr(_est, "pos_vert_ratio",  0.0)
            vel_innov       = getattr(_est, "vel_innov",       0.0)
            pos_horiz_innov = getattr(_est, "pos_horiz_innov", 0.0)
            pos_vert_innov  = getattr(_est, "pos_vert_innov",  0.0)

        # ── Health / battery / armed ─────────────────────────
        batt_v = 12.0; batt_pct = 80.0; armed = 0; failsafe = 0
        
        # Battery from SYS_STATUS
        _sys = self.last_sys_status
        if _sys is not None:
            batt_v    = getattr(_sys, "voltage_battery",  12000) / 1000.0
            batt_pct  = getattr(_sys, "battery_remaining", 80)
            
        # Armed from HEARTBEAT
        _hlt = health_msg or self.last_health
        if _hlt is not None:
            base_mode = getattr(_hlt, "base_mode", 0)
            armed     = 1 if (base_mode & _mavu.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)          else 0
            failsafe  = 1 if (base_mode & _mavu.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY) else 0

        # ── Stale detection ────────────────────────────────
        conn_ok = 1
        if self.last_msg and gps_msg.lat == self.last_msg.lat:
            self._stale_count += 1
        else:
            self._stale_count = 0
        is_stale     = 1 if self._stale_count >= 2 else 0
        self.last_msg = gps_msg

        # ── Assemble vector (must match FEATURE_NAMES order, 34 dims) ──
        return np.array(
            [
                x_m, y_m, alt_m, rel_alt_m, vel_m_s, hdg_deg,
                fix_type, sats, eph_m, epv_m,
                roll_deg, pitch_deg, yaw_deg,
                rollspeed, pitchspeed, yawspeed,
                vib_x, vib_y, vib_z,
                clipping_0, clipping_1, clipping_2,
                vel_ratio, pos_horiz_ratio, pos_vert_ratio,
                vel_innov, pos_horiz_innov, pos_vert_innov,
                batt_v, batt_pct,
                armed, failsafe, conn_ok, is_stale,
            ],
            dtype=np.float32,
        )

    def push_features(self, feature_vec):
        """Append one (34,) feature vector to the rolling buffer."""
        self.buffer.append(np.asarray(feature_vec, dtype=np.float32))

    def predict(self):
        """Return prediction dict, or None if buffer has fewer than 30 samples."""
        if len(self.buffer) < WINDOW_LEN:
            return None

        window = np.stack(list(self.buffer), axis=0)   # (30, 34)

        try:
            last_sample = window[-1]
            rel_alt     = last_sample[3]
            speed       = last_sample[4]
            is_armed    = last_sample[30] > 0.5

            if not is_armed:
                raw_proba = 0.01
            elif rel_alt < 5.0 and speed < 3.0:
                raw_proba = 0.01
            else:
                # Check if we're in the Baylands area — use area-specific model
                if self._baylands_model is not None and self._in_baylands():
                    raw_proba = self._run_baylands_model(window)
                else:
                    raw_proba = self._run_model(window)

            if self._ema_proba is None:
                self._ema_proba = raw_proba
            else:
                self._ema_proba = self._ema_alpha * raw_proba + (1 - self._ema_alpha) * self._ema_proba

        except Exception as exc:
            logger.error("predict() model error: %s", exc)
            return None

        switched       = self._switched
        self._switched = False  # consume flag

        result = {
            "anom_proba":       round(float(self._ema_proba), 6),
            "is_anomaly":       float(self._ema_proba) >= self.anomaly_threshold,
            "terrain":          self.active_terrain_label,
            "model_type":       self.active_model_type,
            "switched_terrain": switched,
            "confidence":       round(float(self._ema_proba), 6),
        }
        logger.debug("predict() terrain=%s proba=%.4f anomaly=%s",
                     result["terrain"], result["anom_proba"], result["is_anomaly"])
        return result

    def update_terrain(self, lat, lon, alt_m):
        """Classify GPS position and debounce-switch the active model.

        Returns the current active terrain string (post any switch).
        """
        if self._classifier is None:
            return self.active_terrain_label

        try:
            new_terrain = self._classifier.classify_point(lat, lon, alt_m)
        except Exception as exc:
            logger.warning("TerrainClassifier.classify_point failed: %s", exc)
            return self.active_terrain_label

        # Debounce: only switch after DEBOUNCE_COUNT consecutive identical calls
        if new_terrain == self._debounce_candidate:
            self._debounce_count += 1
        else:
            self._debounce_candidate = new_terrain
            self._debounce_count     = 1

        if (
            self._debounce_count >= DEBOUNCE_COUNT
            and new_terrain != self.active_terrain_label
        ):
            old = self.active_terrain_label
            self._set_active_terrain(new_terrain)
            self._switched       = True
            self._debounce_count = 0    # reset after confirmed switch
            logger.info("Terrain debounce confirmed: %s -> %s", old, new_terrain)

        return self.active_terrain_label

    def get_active_terrain(self):
        """Return current terrain label."""
        return self.active_terrain_label

    def get_model_info(self):
        """Return metadata dict about the currently active model."""
        info = self.active_info
        return {
            "terrain":            self.active_terrain_label,
            "model_type":         self.active_model_type,
            "rf_val_f1":          info.get("rf_val_f1"),
            "cnn_val_f1":         info.get("cnn_val_f1"),
            "n_training_windows": info.get("n_train"),
            "is_fallback":        self.is_fallback,
        }

    # ------------------------------------------------------------------
    # Batch / offline inference
    # ------------------------------------------------------------------

    def predict_csv(self, csv_path, window_len=WINDOW_LEN, stride=15):
        """Offline inference over a cleaned CSV.

        Calls update_terrain() per window using median lat/lon/alt_m.
        Returns DataFrame: window_idx, start_row, end_row, terrain,
        model_type, anom_proba, is_anomaly, switched_terrain.
        """
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, low_memory=False)

        # Derive x_m, y_m if missing
        if "x_m" not in df.columns and "lat_deg" in df.columns:
            first_lat = df["lat_deg"].iloc[0]
            first_lon = df["lon_deg"].iloc[0]
            df["x_m"] = np.radians(df["lat_deg"] - first_lat) * EARTH_RADIUS_M
            df["y_m"] = (
                np.radians(df["lon_deg"] - first_lon)
                * EARTH_RADIUS_M
                * np.cos(np.radians(first_lat))
            )

        # Load feature column order
        feat_json = Path(__file__).resolve().parent.parent / "Step_5_Data" / "artifacts" / "feature_names.json"
        if feat_json.exists():
            with open(feat_json) as f:
                feature_cols = json.load(f)
        else:
            feature_cols = [
                c for c in df.columns
                if c not in ("terrain_type", "lat_deg", "lon_deg",
                             "time_s", "label", "Unnamed: 0")
            ]

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        rows = []
        self.buffer.clear()   # clean offline pass

        for win_idx, start in enumerate(range(0, len(df) - window_len + 1, stride)):
            end   = start + window_len
            chunk = df.iloc[start:end]

            # Update terrain from median GPS of this window
            if "lat_deg" in df.columns:
                lat_med = float(chunk["lat_deg"].median())
                lon_med = float(chunk["lon_deg"].median())
                alt_med = float(chunk["alt_m"].median()) if "alt_m" in chunk.columns else 50.0
                self.update_terrain(lat_med, lon_med, alt_med)

            # Fill buffer with exactly this window
            self.buffer.clear()
            for _, row in chunk.iterrows():
                self.buffer.append(
                    row[feature_cols].values.astype(np.float32)
                )

            result = self.predict()
            if result is None:
                continue

            rows.append({
                "window_idx":       win_idx,
                "start_row":        start,
                "end_row":          end - 1,
                "terrain":          result["terrain"],
                "model_type":       result["model_type"],
                "anom_proba":       result["anom_proba"],
                "is_anomaly":       result["is_anomaly"],
                "switched_terrain": result["switched_terrain"],
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal model runner
    # ------------------------------------------------------------------

    def _run_model(self, window):
        """Route (30, 34) window to active model. Returns anomaly probability."""
        if self.active_model_type == "cnn" and self.active_cnn_model is not None:
            return self._predict_cnn(window)
        return self._predict_rf(window)

    def _in_baylands(self) -> bool:
        """Check if current GPS position is inside the Baylands training area."""
        if self._baylands_bbox is None or self.home_gps is None:
            return False
        lat, lon = self.home_gps
        bb = self._baylands_bbox
        return (bb["lat_min"] <= lat <= bb["lat_max"] and 
                bb["lon_min"] <= lon <= bb["lon_max"])

    def _run_baylands_model(self, window) -> float:
        """Use area-specific model for Baylands region."""
        X_flat = window.reshape(1, -1)
        X_scaled = self._baylands_scaler.transform(X_flat)
        if hasattr(self._baylands_model, "predict_proba"):
            return float(self._baylands_model.predict_proba(X_scaled)[0, 1])
        return float(self._baylands_model.predict(X_scaled)[0])

    def _predict_rf(self, window):
        """window (30, 34) → RF predict_proba → float.

        The terrain scaler was fitted on the FLATTENED window (1020 features),
        matching the training pipeline in 05_train_terrain_models.py where
        X_train.reshape(n, -1) is passed to scaler.fit().
        """
        X_flat    = window.reshape(1, -1)                  # (1, 1020)
        X_flat_sc = self.active_scaler.transform(X_flat)   # (1, 1020)
        model     = self.active_rf_model
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X_flat_sc)[0, 1])
        return float(model.predict(X_flat_sc)[0])

    def _predict_cnn(self, window):
        """window (30, 34) → CNN sigmoid output → float.

        CNN scaler was also fitted on flattened (1020,) rows during training.
        We scale the flat representation then reshape back for the CNN.
        """
        X_flat    = window.reshape(1, -1)                        # (1, 1020)
        X_flat_sc = self.active_scaler.transform(X_flat)         # (1, 1020)
        window_sc = X_flat_sc.reshape(WINDOW_LEN, NUM_FEATURES)  # (30, 34)
        x = torch.FloatTensor(window_sc).unsqueeze(0)            # (1, 30, 34)
        x = x.permute(0, 2, 1)                                   # (1, 34, 30)
        with torch.no_grad():
            prob = self.active_cnn_model(x).item()
        return float(prob)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ROOT = Path(__file__).resolve().parent.parent

    def _make_dispatcher(**kw):
        return TerrainModelDispatcher(
            terrain_models_dir = ROOT / "Step_5_Data/models/terrain",
            global_model_path  = ROOT / "Step_5_Data/models/rf_model.pkl",
            global_scaler_path = ROOT / "Step_5_Data/artifacts/scaler.pkl",
            **kw,
        )

    print("\n" + "="*60)
    print(" TERRAIN MODEL DISPATCHER — UNIT TESTS")
    print("="*60)

    # ------------------------------------------------------------------
    # Test 1: Init loads without error
    # ------------------------------------------------------------------
    print("\n[TEST 1] Init — all terrain models load without error...")
    try:
        d = _make_dispatcher(preferred_model_type="rf")
        print(f"  PASS — active terrain : {d.get_active_terrain()}")
        print(f"         model info     : {d.get_model_info()}")
    except Exception as e:
        print(f"  FAIL: {e}")
        raise

    # ------------------------------------------------------------------
    # Test 2: push_features 30× → predict() returns a dict
    # ------------------------------------------------------------------
    print("\n[TEST 2] push 30 feature vectors → predict() returns dict...")
    d.buffer.clear()
    for _ in range(30):
        d.push_features(np.zeros(NUM_FEATURES, dtype=np.float32))
    res2 = d.predict()
    if isinstance(res2, dict) and "anom_proba" in res2:
        print(f"  PASS — result: {res2}")
    else:
        print(f"  FAIL — got: {res2}")

    # ------------------------------------------------------------------
    # Test 3: update_terrain 5× flat coords → active terrain == 'flat'
    # ------------------------------------------------------------------
    print("\n[TEST 3] update_terrain 5× flat coords → active_terrain == 'flat'...")

    class _FakeFlat:
        def classify_point(self, lat, lon, alt_m):
            return "flat"

    d3 = _make_dispatcher()
    d3._classifier          = _FakeFlat()
    d3._debounce_candidate  = ""
    d3._debounce_count      = 0
    d3.active_terrain_label = "sea"   # start somewhere else

    for _ in range(DEBOUNCE_COUNT):
        d3.update_terrain(0.0, 0.0, 50.0)

    if d3.get_active_terrain() == "flat":
        print(f"  PASS — active terrain: {d3.get_active_terrain()}")
    else:
        print(f"  FAIL — expected 'flat', got '{d3.get_active_terrain()}'")

    # ------------------------------------------------------------------
    # Test 4: predict_csv on a real cleaned CSV → non-empty DataFrame
    # ------------------------------------------------------------------
    print("\n[TEST 4] predict_csv() on a cleaned CSV → non-empty DataFrame...")
    candidates = (
        list((ROOT / "Step_5_Data/processed").glob("*_cleaned.csv")) +
        list((ROOT / "Step_5_Data/synthetic").glob("*_cleaned.csv"))
    )
    candidates = [p for p in candidates if p.stat().st_size > 2000]
    if candidates:
        d4     = _make_dispatcher()
        df_out = d4.predict_csv(candidates[0])
        if len(df_out) > 0:
            print(f"  PASS — {len(df_out)} windows from '{candidates[0].name}'")
            print(df_out[["window_idx","terrain","model_type",
                          "anom_proba","is_anomaly"]].head(4).to_string(index=False))
        else:
            print(f"  FAIL — empty DataFrame")
    else:
        print("  SKIP — no cleaned CSVs found")

    # ------------------------------------------------------------------
    # Test 5: Debounce — alternating terrains do NOT trigger a switch
    # ------------------------------------------------------------------
    print("\n[TEST 5] Debounce — alternating terrains should NOT switch...")

    class _Alternating:
        def __init__(self):
            self._i = 0
        def classify_point(self, lat, lon, alt_m):
            self._i += 1
            return "flat" if self._i % 2 == 0 else "mountain"

    d5 = _make_dispatcher()
    d5._classifier = _Alternating()
    start = d5.get_active_terrain()
    for _ in range(10):
        d5.update_terrain(0.0, 0.0, 50.0)

    if d5.get_active_terrain() == start:
        print(f"  PASS — terrain stayed '{start}' (debounce held)")
    else:
        print(f"  FAIL — unexpectedly switched to '{d5.get_active_terrain()}'")

    print("\n" + "="*60)
    print(" ALL TESTS COMPLETE")
    print("="*60 + "\n")
