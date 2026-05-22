import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# Ensure 4_Detection_Engine is on sys.path
_ENGINE_ROOT = Path(__file__).resolve().parent
if str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

import joblib
import numpy as np
from pymavlink import mavutil

# Import from the local directory
from terrain_dispatcher import TerrainModelDispatcher

WINDOW_LEN = 30
TARGET_RATE_HZ = 10.0
EARTH_RADIUS_M = 6371000.0

FEATURE_NAMES = [
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


class LiveAnomalyDetector:
    def __init__(self, model_path, scaler_path):
        print(f"[INFO] Loading model: {model_path}")
        self.model = joblib.load(model_path)
        print(f"[INFO] Loading scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        self.buffer = deque(maxlen=WINDOW_LEN)
        self.home_gps = None
        self.last_msg = None
        self.last_attitude = None
        self.last_estimator = None
        self.last_vibration = None
        self.last_health = None
        self._stale_count = 0

    def gps_to_local(self, lat, lon):
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

    def extract_features(self, gps_msg, attitude_msg=None, estimator_msg=None, health_msg=None, vibration_msg=None):
        lat_raw = gps_msg.lat / 1e7
        lon_raw = gps_msg.lon / 1e7
        x_m, y_m = self.gps_to_local(lat_raw, lon_raw)
        alt_m = gps_msg.alt / 1000.0
        rel_alt_m = gps_msg.relative_alt / 1000.0
        vel_m_s = (
            (gps_msg.vx ** 2 + gps_msg.vy ** 2 + gps_msg.vz ** 2) ** 0.5
            / 100.0
        )
        hdg_deg = gps_msg.hdg / 100.0 if gps_msg.hdg != 65535 else 0.0
        fix_type = getattr(gps_msg, "fix_type", 3)
        sats = getattr(gps_msg, "satellites_visible", 10)
        eph_m = getattr(gps_msg, "eph", 0) / 100.0 if hasattr(gps_msg, "eph") else 0.0
        epv_m = getattr(gps_msg, "epv", 0) / 100.0 if hasattr(gps_msg, "epv") else 0.0

        roll_deg = 0.0
        pitch_deg = 0.0
        yaw_deg = 0.0
        rollspeed = 0.0
        pitchspeed = 0.0
        yawspeed = 0.0
        if attitude_msg is not None:
            import math
            roll_deg = math.degrees(getattr(attitude_msg, "roll", 0))
            pitch_deg = math.degrees(getattr(attitude_msg, "pitch", 0))
            yaw_deg = math.degrees(getattr(attitude_msg, "yaw", 0))
            rollspeed = getattr(attitude_msg, "rollspeed", 0)
            pitchspeed = getattr(attitude_msg, "pitchspeed", 0)
            yawspeed = getattr(attitude_msg, "yawspeed", 0)

        vib_x = 0.0
        vib_y = 0.0
        vib_z = 0.0
        clipping_0 = 0
        clipping_1 = 0
        clipping_2 = 0
        if vibration_msg is not None:
            vib_x = getattr(vibration_msg, "vibration_x", 0.0)
            vib_y = getattr(vibration_msg, "vibration_y", 0.0)
            vib_z = getattr(vibration_msg, "vibration_z", 0.0)
            clipping_0 = getattr(vibration_msg, "clipping_0", 0)
            clipping_1 = getattr(vibration_msg, "clipping_1", 0)
            clipping_2 = getattr(vibration_msg, "clipping_2", 0)
        elif self.last_vibration is not None:
            vib_x = getattr(self.last_vibration, "vibration_x", 0.0)
            vib_y = getattr(self.last_vibration, "vibration_y", 0.0)
            vib_z = getattr(self.last_vibration, "vibration_z", 0.0)
            clipping_0 = getattr(self.last_vibration, "clipping_0", 0)
            clipping_1 = getattr(self.last_vibration, "clipping_1", 0)
            clipping_2 = getattr(self.last_vibration, "clipping_2", 0)

        vel_ratio = 0.0
        pos_horiz_ratio = 0.0
        pos_vert_ratio = 0.0
        vel_innov = 0.0
        pos_horiz_innov = 0.0
        pos_vert_innov = 0.0
        if estimator_msg is not None:
            vel_ratio = getattr(estimator_msg, "vel_ratio", 0.0)
            pos_horiz_ratio = getattr(estimator_msg, "pos_horiz_ratio", 0.0)
            pos_vert_ratio = getattr(estimator_msg, "pos_vert_ratio", 0.0)
            vel_innov = getattr(estimator_msg, "vel_innov", 0.0)
            pos_horiz_innov = getattr(estimator_msg, "pos_horiz_innov", 0.0)
            pos_vert_innov = getattr(estimator_msg, "pos_vert_innov", 0.0)
        elif self.last_estimator is not None:
            vel_ratio = getattr(self.last_estimator, "vel_ratio", 0.0)
            pos_horiz_ratio = getattr(self.last_estimator, "pos_horiz_ratio", 0.0)
            pos_vert_ratio = getattr(self.last_estimator, "pos_vert_ratio", 0.0)
            vel_innov = getattr(self.last_estimator, "vel_innov", 0.0)
            pos_horiz_innov = getattr(self.last_estimator, "pos_horiz_innov", 0.0)
            pos_vert_innov = getattr(self.last_estimator, "pos_vert_innov", 0.0)

        batt_v = 12.0
        batt_pct = 80.0
        armed = 0
        failsafe = 0
        if health_msg is not None:
            from pymavlink import mavutil
            batt_v = getattr(health_msg, "voltage_battery", 12000) / 1000.0
            batt_pct = getattr(health_msg, "battery_remaining", 80)
            base_mode = getattr(health_msg, "base_mode", 0)
            armed = 1 if (base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 0
            failsafe = 1 if (base_mode & mavutil.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY) else 0
        elif self.last_health is not None:
            from pymavlink import mavutil
            batt_v = getattr(self.last_health, "voltage_battery", 12000) / 1000.0
            batt_pct = getattr(self.last_health, "battery_remaining", 80)
            base_mode = getattr(self.last_health, "base_mode", 0)
            armed = 1 if (base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 0
            failsafe = 1 if (base_mode & mavutil.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY) else 0

        conn_ok = 1
        is_stale = 0
        if self.last_msg and gps_msg.lat == self.last_msg.lat:
            self._stale_count += 1
        else:
            self._stale_count = 0
        is_stale = 1 if self._stale_count >= 2 else 0
        self.last_msg = gps_msg

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

    def predict(self):
        if len(self.buffer) < WINDOW_LEN:
            return None
        window = np.stack(self.buffer, axis=0)
        window_sc = self.scaler.transform(window)
        X_flat = window_sc.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_flat)[0, 1]
        return None


def run_live_inference(
    connection_string: str,
    model_path: str,
    scaler_path: str,
    log_dir: str,
    terrain_models_dir: str = "Step_5_Data/models/terrain",
    preferred_model_type: str = "rf",
    anomaly_threshold: float = 0.5,
) -> None:
    detector = TerrainModelDispatcher(
        terrain_models_dir=terrain_models_dir,
        global_model_path=model_path,
        global_scaler_path=scaler_path,
        preferred_model_type=preferred_model_type,
        anomaly_threshold=anomaly_threshold,
    )

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        datetime.now(timezone.utc).strftime("live_%Y%m%d_%H%M%S.csv"),
    )
    print(f"[INFO] Logging to {log_path}")

    log_file = open(log_path, mode="w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        [
            "timestamp", "unix_time", "anom_proba",
            "lat_deg", "lon_deg", "alt_m", "rel_alt_m", "vel_m_s",
            "roll_deg", "pitch_deg", "yaw_deg",
            "rollspeed", "pitchspeed", "yawspeed",
            "terrain", "model_type", "switched_terrain",
        ]
    )

    print(f"[INFO] Connecting to {connection_string}")
    master = mavutil.mavlink_connection(connection_string, dialect='common')
    master.wait_heartbeat()
    print("[INFO] Heartbeat received. Starting inference loop...")
    print("[DBG] Listening for messages (first 5 will print type)...")
    debug_count = 0

    message_types = ["GLOBAL_POSITION_INT", "ATTITUDE", "ESTIMATOR_STATUS", "SYS_STATUS", "HEARTBEAT", "VIBRATION"]
    gps_count = 0
    pred_count = 0

    try:
        while True:
            msg = master.recv_match(type=message_types, blocking=True, timeout=1.0)
            if not msg:
                # If we've received no messages after heartbeat, print a warning
                if debug_count == 0 and gps_count == 0:
                    # Check if ANY messages are arriving (unfiltered)
                    raw = master.recv_match(blocking=True, timeout=0.5)
                    if raw:
                        print(f"[DBG] Message received but type filter missed: {raw.get_type()}")
                        msg = raw  # process it anyway if it's one of our types
                    else:
                        continue
                else:
                    continue

            msg_type = msg.get_type()
            if debug_count < 5:
                print(f"[DBG] msg type: {msg_type}")
                debug_count += 1

            if msg_type == "VIBRATION":
                detector.last_vibration = msg
            elif msg_type == "HEARTBEAT":
                detector.last_health = msg
            elif msg_type == "SYS_STATUS":
                detector.last_sys_status = msg
            elif msg_type == "GLOBAL_POSITION_INT":
                gps_count += 1
                if gps_count == 1:
                    print(f"[LIVE] First GLOBAL_POSITION_INT received! Starting detection...")
                if gps_count % 10 == 0:
                    print(f"[LIVE] gps={gps_count} | buf={len(detector.buffer)}/30 | preds={pred_count}")
                lat_deg = msg.lat / 1e7
                lon_deg = msg.lon / 1e7
                alt_m   = msg.alt / 1000.0
                detector.update_terrain(lat_deg, lon_deg, alt_m)

                feat_vec = detector.extract_features(
                    msg,
                    attitude_msg=detector.last_attitude,
                    estimator_msg=detector.last_estimator,
                    health_msg=detector.last_health,
                    vibration_msg=detector.last_vibration,
                )
                detector.push_features(feat_vec)

                result = detector.predict()
                if result is not None:
                    pred_count += 1
                    proba      = result["anom_proba"]
                    terrain    = result["terrain"]
                    model_type = result["model_type"]
                    switched   = result["switched_terrain"]
                    now        = time.time()
                    rel_alt_m  = msg.relative_alt / 1000.0
                    vel_m_s    = (msg.vx**2 + msg.vy**2 + msg.vz**2)**0.5 / 100.0
                    status     = "ANOMALY" if proba > anomaly_threshold else "normal"
                    
                    # Debug print for features if anomaly detected
                    if status == "ANOMALY":
                        rel_alt_m  = msg.relative_alt / 1000.0
                        vel_m_s    = (msg.vx**2 + msg.vy**2 + msg.vz**2)**0.5 / 100.0
                        # armed is usually in HEARTBEAT (last_health)
                        from pymavlink import mavutil as _mavu
                        base_mode = getattr(detector.last_health, "base_mode", 0)
                        armed = 1 if (base_mode & _mavu.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 0
                        print(f"  [DEBUG] alt={alt_m:.1f} rel_alt={rel_alt_m:.1f} speed={vel_m_s:.1f} armed={armed}")

                    print(f"[{now:.2f}] terrain={terrain} model={model_type} p(anom): {proba:.3f} -> {status}")

                    roll = 0.0
                    pitch = 0.0
                    yaw = 0.0
                    rs = 0.0
                    ps = 0.0
                    ys = 0.0
                    if detector.last_attitude:
                        import math
                        roll = math.degrees(getattr(detector.last_attitude, "roll", 0))
                        pitch = math.degrees(getattr(detector.last_attitude, "pitch", 0))
                        yaw = math.degrees(getattr(detector.last_attitude, "yaw", 0))
                        rs = getattr(detector.last_attitude, "rollspeed", 0)
                        ps = getattr(detector.last_attitude, "pitchspeed", 0)
                        ys = getattr(detector.last_attitude, "yawspeed", 0)

                    log_writer.writerow(
                        [
                            datetime.now(timezone.utc).isoformat(),
                            f"{now:.3f}",
                            f"{proba:.6f}",
                            f"{lat_deg:.7f}",
                            f"{lon_deg:.7f}",
                            f"{alt_m:.3f}",
                            f"{rel_alt_m:.3f}",
                            f"{vel_m_s:.3f}",
                            f"{roll:.2f}",
                            f"{pitch:.2f}",
                            f"{yaw:.2f}",
                            f"{rs:.4f}",
                            f"{ps:.4f}",
                            f"{ys:.4f}",
                            terrain,
                            model_type,
                            int(switched),
                        ]
                    )
                    log_file.flush()

            elif msg_type == "ATTITUDE":
                detector.last_attitude = msg

            elif msg_type == "ESTIMATOR_STATUS":
                detector.last_estimator = msg

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try:
            log_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", default="udp:127.0.0.1:14562")
    parser.add_argument("--model-path", "--model", default="Step_5_Data/models/rf_model.pkl")
    parser.add_argument("--scaler-path", "--scaler", default="Step_5_Data/artifacts/scaler.pkl")
    parser.add_argument("--log-dir", default="Step_8_Archive/logs")
    parser.add_argument("--terrain-models-dir", default="Step_5_Data/models/terrain")
    parser.add_argument("--preferred-model-type", default="rf", choices=["rf", "cnn"])
    parser.add_argument("--anomaly-threshold", type=float, default=0.6)
    args = parser.parse_args()
    run_live_inference(
        args.connection,
        args.model_path,
        args.scaler_path,
        args.log_dir,
        terrain_models_dir=args.terrain_models_dir,
        preferred_model_type=args.preferred_model_type,
        anomaly_threshold=args.anomaly_threshold,
    )
