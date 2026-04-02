# gps_monitor/console_ui.py

from __future__ import annotations

import os
import time
from typing import Optional

from .state_model import TelemetryState


class ConsoleUI:
    def __init__(self, refresh_hz: float = 4.0) -> None:
        self.refresh_hz = refresh_hz

    def run(self, state_provider, stop_checker) -> None:
        interval = 1.0 / self.refresh_hz

        try:
            while not stop_checker():
                state: TelemetryState = state_provider()
                self.render(state)
                time.sleep(interval)
        except KeyboardInterrupt:
            pass

    def render(self, state: TelemetryState) -> None:
        self._clear()

        gps = state.gps
        imu = state.imu
        health = state.health

        print("=" * 72)
        print("GPS Monitor")
        print("=" * 72)

        print(
            f"Connection: {'OK' if state.connection_ok else 'DOWN':<6} "
            f"| Uptime: {state.uptime_s():7.1f} s "
            f"| Mode: {health.mode:<15} "
            f"| Armed: {'YES' if health.armed else 'NO'}"
        )

        print(
            f"Failsafe: {'YES' if health.failsafe else 'NO':<3} "
            f"| Battery: {self._fmt(health.battery_voltage, '.2f', ' V'):>10} "
            f"| Remaining: {self._fmt(health.battery_remaining_pct, 'd', ' %'):>8}"
        )

        print("-" * 72)
        print("GPS")
        print("-" * 72)
        print(f"Last update : {gps.last_update.isoformat() if gps.last_update else 'N/A'}")
        print(f"Latitude    : {self._fmt(gps.lat_deg, '.7f')}")
        print(f"Longitude   : {self._fmt(gps.lon_deg, '.7f')}")
        print(f"Altitude    : {self._fmt(gps.alt_m, '.2f', ' m')}")
        print(f"Rel Alt     : {self._fmt(gps.rel_alt_m, '.2f', ' m')}")
        print(f"Velocity    : {self._fmt(gps.vel_m_s, '.2f', ' m/s')}")
        print(f"Heading     : {self._fmt(gps.hdg_deg, '.2f', ' deg')}")
        print(f"Fix Type    : {self._fmt(gps.fix_type)}")
        print(f"Satellites  : {self._fmt(gps.satellites_visible)}")
        print(f"EPH         : {self._fmt(gps.eph, '.2f', ' m')}")
        print(f"EPV         : {self._fmt(gps.epv, '.2f', ' m')}")

        print("-" * 72)
        print("IMU")
        print("-" * 72)
        print(
            f"Accel [m/s²]: "
            f"x={self._fmt(imu.ax_mps2, '.3f'):>10} "
            f"y={self._fmt(imu.ay_mps2, '.3f'):>10} "
            f"z={self._fmt(imu.az_mps2, '.3f'):>10}"
        )
        print(
            f"Gyro [rad/s]: "
            f"x={self._fmt(imu.gx_radps, '.3f'):>10} "
            f"y={self._fmt(imu.gy_radps, '.3f'):>10} "
            f"z={self._fmt(imu.gz_radps, '.3f'):>10}"
        )

        print("-" * 72)
        print(f"Last error: {state.last_error or 'None'}")
        print("Press Ctrl+C to stop.")
        print("=" * 72)

    @staticmethod
    def _fmt(value: Optional[float], fmt: Optional[str] = None, suffix: str = "") -> str:
        if value is None:
            return "N/A"
        if fmt == "d":
            return f"{int(value)}{suffix}"
        if fmt:
            return f"{value:{fmt}}{suffix}"
        return f"{value}{suffix}"

    @staticmethod
    def _clear() -> None:
        os.system("cls" if os.name == "nt" else "clear")
