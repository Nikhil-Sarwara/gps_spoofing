import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
from pymavlink import mavutil

WINDOW_LEN = 30
TARGET_RATE_HZ = 10.0
EARTH_RADIUS_M = 6371000.0

FEATURE_NAMES = [
    "x_m", "y_m", "alt_m", "rel_alt_m", "vel_m_s", "hdg_deg",
    "fix_type", "satellites_visible", "eph_m", "epv_m",
    "roll_deg", "pitch_deg", "yaw_deg",
    "rollspeed_radps", "pitchspeed_radps", "yawspeed_radps",
    "vibration_x", "vibration_y", "vibration_z",
    "vel_ratio", "pos_horiz_ratio", "pos_vert_ratio",
    "vel_innov", "pos_horiz_innov", "pos_vert_innov",
    "battery_voltage", "battery_remaining_pct",
    "armed", "failsafe", "connection_ok", "is_stale",
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

    def extract_features(self, gps_msg, attitude_msg=None, estimator_msg=None, health_msg=None):
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

        batt_v = 12.0
        batt_pct = 80.0
        armed = 1
        failsafe = 0
        if health_msg is not None:
            batt_v = getattr(health_msg, "voltage_battery", 12000) / 1000.0
            batt_pct = getattr(health_msg, "battery_remaining", 80)
            base_mode = getattr(health_msg, "base_mode", 0)
            armed = 1 if (base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 0
            failsafe = 1 if (base_mode & mavutil.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY) else 0

        conn_ok = 1
        is_stale = 0
        if self.last_msg and gps_msg.lat == self.last_msg.lat:
            is_stale = 1
        self.last_msg = gps_msg

        return np.array(
            [
                x_m, y_m, alt_m, rel_alt_m, vel_m_s, hdg_deg,
                fix_type, sats, eph_m, epv_m,
                roll_deg, pitch_deg, yaw_deg,
                rollspeed, pitchspeed, yawspeed,
                vib_x, vib_y, vib_z,
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


def run_live_inference(connection_string, model_path, scaler_path, log_dir):
    detector = LiveAnomalyDetector(model_path, scaler_path)

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
        ]
    )

    print(f"[INFO] Connecting to {connection_string}")
    master = mavutil.mavlink_connection(connection_string)
    master.wait_heartbeat()
    print("[INFO] Heartbeat received. Starting inference loop...")

    message_types = ["GLOBAL_POSITION_INT", "ATTITUDE", "ESTIMATOR_STATUS", "SYS_STATUS", "HEARTBEAT"]

    try:
        while True:
            msg = master.recv_match(type=message_types, blocking=True, timeout=1.0)
            if not msg:
                continue

            msg_type = msg.get_type()

            if msg_type == "GLOBAL_POSITION_INT":
                feat_vec = detector.extract_features(
                    msg,
                    attitude_msg=detector.last_attitude,
                    estimator_msg=detector.last_estimator,
                )
                detector.buffer.append(feat_vec)

                proba = detector.predict()
                if proba is not None:
                    now = time.time()
                    lat_deg = msg.lat / 1e7
                    lon_deg = msg.lon / 1e7
                    alt_m = msg.alt / 1000.0
                    rel_alt_m = msg.relative_alt / 1000.0
                    vel_m_s = (msg.vx ** 2 + msg.vy ** 2 + msg.vz ** 2) ** 0.5 / 100.0
                    status = "ANOMALY" if proba > 0.5 else "normal"
                    print(f"[{now:.2f}] p(anom): {proba:.3f} -> {status}")

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
    parser.add_argument("--connection", default="udp:127.0.0.1:14560")
    parser.add_argument("--model", default="ml/models/rf_model.pkl")
    parser.add_argument("--scaler", default="ml/artifacts/scaler.pkl")
    parser.add_argument("--log-dir", default="ui/logs")
    args = parser.parse_args()
    run_live_inference(args.connection, args.model, args.scaler, args.log_dir)
