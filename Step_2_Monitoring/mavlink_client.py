# gps_monitor/mavlink_client.py

from __future__ import annotations

import csv
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from pymavlink import mavutil

from .config import LOG_DIR, LOG_HZ, MAVLINK_CONNECTION_URL, MESSAGE_TYPES
from .state_model import TelemetryState


class MAVLinkClient:
    def __init__(self, state: TelemetryState) -> None:
        self.state = state
        self.master: Optional[mavutil.mavfile] = None

        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._logger_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._log_fp = None
        self._csv_writer: Optional[csv.writer] = None
        self._log_start_monotonic: Optional[float] = None

    def connect(self, heartbeat_timeout: float = 10.0) -> None:
        try:
            self.master = mavutil.mavlink_connection(MAVLINK_CONNECTION_URL)
            self.master.wait_heartbeat(timeout=heartbeat_timeout)
            self.state.connection_ok = True
            self.state.last_error = None
        except Exception as exc:
            self.state.connection_ok = False
            self.state.last_error = f"Connection failed: {exc}"
            raise

    def start(self) -> None:
        if self.master is None:
            self.connect()

        self._stop_event.clear()
        self._open_log_file()

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="mavlink-reader",
            daemon=True,
        )
        self._logger_thread = threading.Thread(
            target=self._logger_loop,
            name="mavlink-logger",
            daemon=True,
        )

        self._reader_thread.start()
        self._logger_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)

        if self._logger_thread and self._logger_thread.is_alive():
            self._logger_thread.join(timeout=2.0)

        self._close_log_file()

        if self.master is not None:
            try:
                self.master.close()
            except Exception:
                pass

        self.state.connection_ok = False

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def snapshot(self) -> TelemetryState:
        with self._lock:
            return TelemetryState(
                gps=self.state.gps.__class__(**self.state.gps.__dict__),
                imu=self.state.imu.__class__(**self.state.imu.__dict__),
                estimator=self.state.estimator.__class__(**self.state.estimator.__dict__),
                health=self.state.health.__class__(**self.state.health.__dict__),
                start_time=self.state.start_time,
                connection_ok=self.state.connection_ok,
                last_error=self.state.last_error,
            )

    def _reader_loop(self) -> None:
        assert self.master is not None

        while not self._stop_event.is_set():
            try:
                msg = self.master.recv_match(
                    type=MESSAGE_TYPES,
                    blocking=True,
                    timeout=1.0,
                )

                if msg is None:
                    continue

                msg_type = msg.get_type()
                if msg_type == "BAD_DATA":
                    continue

                with self._lock:
                    self._handle_message(msg_type, msg)

                self.state.connection_ok = True
                self.state.last_error = None

            except Exception as exc:
                self.state.connection_ok = False
                self.state.last_error = f"Reader error: {exc}"
                time.sleep(0.5)

    def _handle_message(self, msg_type: str, msg) -> None:
        now = datetime.now()

        if msg_type == "GLOBAL_POSITION_INT":
            self.state.gps.last_update = now
            self.state.gps.lat_deg = self._safe_div(msg.lat, 1e7)
            self.state.gps.lon_deg = self._safe_div(msg.lon, 1e7)
            self.state.gps.alt_m = self._safe_div(msg.alt, 1000.0)
            self.state.gps.rel_alt_m = self._safe_div(msg.relative_alt, 1000.0)

            vx = self._safe_div(msg.vx, 100.0)
            vy = self._safe_div(msg.vy, 100.0)
            vz = self._safe_div(msg.vz, 100.0)
            if None not in (vx, vy, vz):
                self.state.gps.vel_m_s = math.sqrt(vx * vx + vy * vy + vz * vz)

            hdg = self._safe_div(msg.hdg, 100.0)
            if hdg is not None and hdg != 655.35:
                self.state.gps.hdg_deg = hdg

        elif msg_type == "GPS_RAW_INT":
            self.state.gps.last_update = now
            self.state.gps.fix_type = getattr(msg, "fix_type", None)
            self.state.gps.satellites_visible = self._none_if_uint8_max(
                getattr(msg, "satellites_visible", None)
            )
            self.state.gps.eph = self._gps_accuracy_to_m(getattr(msg, "eph", None))
            self.state.gps.epv = self._gps_accuracy_to_m(getattr(msg, "epv", None))

        elif msg_type == "ATTITUDE":
            self.state.imu.roll_deg = math.degrees(self._safe_div(getattr(msg, "roll", None), 1.0))
            self.state.imu.pitch_deg = math.degrees(self._safe_div(getattr(msg, "pitch", None), 1.0))
            self.state.imu.yaw_deg = math.degrees(self._safe_div(getattr(msg, "yaw", None), 1.0))
            self.state.imu.rollspeed_radps = getattr(msg, "rollspeed", None)
            self.state.imu.pitchspeed_radps = getattr(msg, "pitchspeed", None)
            self.state.imu.yawspeed_radps = getattr(msg, "yawspeed", None)

        elif msg_type == "VIBRATION":
            self.state.imu.vibration_x = getattr(msg, "vibration_x", None)
            self.state.imu.vibration_y = getattr(msg, "vibration_y", None)
            self.state.imu.vibration_z = getattr(msg, "vibration_z", None)
            self.state.imu.clipping_0 = getattr(msg, "clipping_0", 0)
            self.state.imu.clipping_1 = getattr(msg, "clipping_1", 0)
            self.state.imu.clipping_2 = getattr(msg, "clipping_2", 0)

        elif msg_type == "ESTIMATOR_STATUS":
            self.state.estimator.vel_ratio = self._none_if_too_large(
                getattr(msg, "vel_ratio", None)
            )
            self.state.estimator.pos_horiz_ratio = self._none_if_too_large(
                getattr(msg, "pos_horiz_ratio", None)
            )
            self.state.estimator.pos_vert_ratio = self._none_if_too_large(
                getattr(msg, "pos_vert_ratio", None)
            )
            self.state.estimator.mag_ratio = self._none_if_too_large(
                getattr(msg, "mag_ratio", None)
            )
            self.state.estimator.hagl_ratio = self._none_if_too_large(
                getattr(msg, "hagl_ratio", None)
            )
            self.state.estimator.vel_innov = self._none_if_too_large(
                getattr(msg, "vel_innov", None)
            )
            self.state.estimator.pos_horiz_innov = self._none_if_too_large(
                getattr(msg, "pos_horiz_innov", None)
            )
            self.state.estimator.pos_vert_innov = self._none_if_too_large(
                getattr(msg, "pos_vert_innov", None)
            )

        elif msg_type == "HEARTBEAT":
            base_mode = getattr(msg, "base_mode", 0)
            self.state.health.armed = bool(
                base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            self.state.health.mode = mavutil.mode_string_v10(msg)
            self.state.health.failsafe = bool(
                base_mode & mavutil.mavlink.MAV_MODE_FLAG_DECODE_POSITION_SAFETY
            )

        elif msg_type == "SYS_STATUS":
            voltage_mv = getattr(msg, "voltage_battery", None)
            self.state.health.battery_voltage = (
                None if voltage_mv in (None, 0, 65535) else voltage_mv / 1000.0
            )

            batt_remaining = getattr(msg, "battery_remaining", None)
            self.state.health.battery_remaining_pct = (
                None if batt_remaining in (None, -1, 255) else batt_remaining
            )

        elif msg_type == "ATTITUDE":
            pass

        elif msg_type == "VFR_HUD":
            airspeed = getattr(msg, "airspeed", None)
            groundspeed = getattr(msg, "groundspeed", None)

            if self.state.gps.vel_m_s is None:
                if groundspeed is not None and groundspeed >= 0:
                    self.state.gps.vel_m_s = float(groundspeed)
                elif airspeed is not None and airspeed >= 0:
                    self.state.gps.vel_m_s = float(airspeed)

            heading = getattr(msg, "heading", None)
            if heading is not None and heading != 65535:
                self.state.gps.hdg_deg = float(heading)

            alt = getattr(msg, "alt", None)
            if alt is not None and self.state.gps.alt_m is None:
                self.state.gps.alt_m = float(alt)

    def _logger_loop(self) -> None:
        interval = 1.0 / LOG_HZ

        while not self._stop_event.is_set():
            tick_start = time.monotonic()

            with self._lock:
                row = self._build_csv_row()

            try:
                if self._csv_writer is not None:
                    self._csv_writer.writerow(row)
                    self._log_fp.flush()
            except Exception as exc:
                self.state.last_error = f"Logging error: {exc}"

            elapsed = time.monotonic() - tick_start
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

    def _open_log_file(self) -> None:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"gps_log_{ts}.csv")

        self._log_fp = open(log_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._log_fp)
        self._log_start_monotonic = time.monotonic()

        self._csv_writer.writerow(
            [
                "time_s",
                "lat_deg",
                "lon_deg",
                "alt_m",
                "rel_alt_m",
                "vel_m_s",
                "hdg_deg",
                "fix_type",
                "satellites_visible",
                "eph_m",
                "epv_m",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "rollspeed_radps",
                "pitchspeed_radps",
                "yawspeed_radps",
                "vibration_x",
                "vibration_y",
                "vibration_z",
                "clipping_0",
                "clipping_1",
                "clipping_2",
                "vel_ratio",
                "pos_horiz_ratio",
                "pos_vert_ratio",
                "vel_innov",
                "pos_horiz_innov",
                "pos_vert_innov",
                "battery_voltage",
                "battery_remaining_pct",
                "armed",
                "mode",
                "failsafe",
                "connection_ok",
                "last_update_iso",
            ]
        )

    def _close_log_file(self) -> None:
        if self._log_fp is not None:
            try:
                self._log_fp.close()
            except Exception:
                pass
        self._log_fp = None
        self._csv_writer = None

    def _build_csv_row(self) -> list:
        gps = self.state.gps
        imu = self.state.imu
        est = self.state.estimator
        health = self.state.health

        time_s = None
        if self._log_start_monotonic is not None:
            time_s = round(time.monotonic() - self._log_start_monotonic, 3)

        return [
            time_s,
            gps.lat_deg,
            gps.lon_deg,
            gps.alt_m,
            gps.rel_alt_m,
            gps.vel_m_s,
            gps.hdg_deg,
            gps.fix_type,
            gps.satellites_visible,
            gps.eph,
            gps.epv,
            imu.roll_deg,
            imu.pitch_deg,
            imu.yaw_deg,
            imu.rollspeed_radps,
            imu.pitchspeed_radps,
            imu.yawspeed_radps,
            imu.vibration_x,
            imu.vibration_y,
            imu.vibration_z,
            imu.clipping_0,
            imu.clipping_1,
            imu.clipping_2,
            est.vel_ratio,
            est.pos_horiz_ratio,
            est.pos_vert_ratio,
            est.vel_innov,
            est.pos_horiz_innov,
            est.pos_vert_innov,
            health.battery_voltage,
            health.battery_remaining_pct,
            int(health.armed),
            health.mode,
            int(health.failsafe),
            int(self.state.connection_ok),
            gps.last_update.isoformat() if gps.last_update else None,
        ]

    @staticmethod
    def _safe_div(value, denom: float) -> Optional[float]:
        if value is None:
            return None
        return float(value) / denom

    @staticmethod
    def _mg_to_mps2(value) -> Optional[float]:
        if value is None:
            return None
        return float(value) * 9.80665 / 1000.0

    @staticmethod
    def _mrad_to_rad(value) -> Optional[float]:
        if value is None:
            return None
        return float(value) / 1000.0

    @staticmethod
    def _gps_accuracy_to_m(value) -> Optional[float]:
        if value in (None, 65535):
            return None
        return float(value) / 100.0

    @staticmethod
    def _none_if_uint8_max(value) -> Optional[int]:
        if value in (None, 255):
            return None
        return int(value)

    @staticmethod
    def _none_if_too_large(value, threshold: float = 1e6) -> Optional[float]:
        if value is None or abs(value) > threshold:
            return None
        return float(value)
