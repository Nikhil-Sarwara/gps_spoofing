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


class LiveAnomalyDetector:
    def __init__(self, model_path, scaler_path):
        print(f"[INFO] Loading model: {model_path}")
        self.model = joblib.load(model_path)
        print(f"[INFO] Loading scaler: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        self.buffer = deque(maxlen=WINDOW_LEN)
        self.home_gps = None
        self.last_msg = None

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

    def extract_features(self, msg_dict):
        lat_raw = msg_dict["lat"] / 1e7
        lon_raw = msg_dict["lon"] / 1e7
        x_m, y_m = self.gps_to_local(lat_raw, lon_raw)
        alt_m = msg_dict["alt"] / 1000.0
        rel_alt_m = msg_dict["relative_alt"] / 1000.0
        vel_m_s = (
            (msg_dict["vx"] ** 2 + msg_dict["vy"] ** 2 + msg_dict["vz"] ** 2) ** 0.5
            / 100.0
        )
        hdg_deg = msg_dict["hdg"] / 100.0 if msg_dict["hdg"] != 65535 else 0.0
        fix_type = msg_dict.get("fix_type", 3)
        sats = msg_dict.get("satellites_visible", 10)
        eph_m = msg_dict.get("eph", 0) / 100.0
        epv_m = msg_dict.get("epv", 0) / 100.0
        batt_v = 12.0
        batt_pct = 80.0
        armed = 1
        failsafe = msg_dict.get("failsafe", 0)
        conn_ok = 1
        is_stale = 0
        if self.last_msg and msg_dict["lat"] == self.last_msg["lat"]:
            is_stale = 1
        self.last_msg = msg_dict
        return np.array(
            [
                x_m, y_m, alt_m, rel_alt_m, vel_m_s, hdg_deg,
                fix_type, sats, eph_m, epv_m,
                batt_v, batt_pct, armed, failsafe,
                conn_ok, is_stale,
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
        ]
    )

    print(f"[INFO] Connecting to {connection_string}")
    master = mavutil.mavlink_connection(connection_string)
    master.wait_heartbeat()
    print("[INFO] Heartbeat received. Starting inference loop...")

    try:
        while True:
            msg = master.recv_match(
                type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0
            )
            if not msg:
                continue

            feat_vec = detector.extract_features(msg.to_dict())
            detector.buffer.append(feat_vec)

            proba = detector.predict()
            if proba is not None:
                now = time.time()
                lat_deg = msg.lat / 1e7
                lon_deg = msg.lon / 1e7
                alt_m = msg.alt / 1000.0
                rel_alt_m = msg.relative_alt / 1000.0
                vel_m_s = (
                    (msg.vx ** 2 + msg.vy ** 2 + msg.vz ** 2) ** 0.5 / 100.0
                )
                status = "ANOMALY" if proba > 0.5 else "normal"
                print(f"[{now:.2f}] p(anom): {proba:.3f} -> {status}")
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
                    ]
                )
                log_file.flush()

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
