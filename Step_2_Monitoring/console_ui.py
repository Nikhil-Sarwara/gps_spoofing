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
            f"Attitude [deg]: "
            f"roll={self._fmt(imu.roll_deg, '.2f'):>8} "
            f"pitch={self._fmt(imu.pitch_deg, '.2f'):>8} "
            f"yaw={self._fmt(imu.yaw_deg, '.2f'):>8}"
        )
        print(
            f"Ang.Rate [rad/s]: "
            f"roll={self._fmt(imu.rollspeed_radps, '.3f'):>8} "
            f"pitch={self._fmt(imu.pitchspeed_radps, '.3f'):>8} "
            f"yaw={self._fmt(imu.yawspeed_radps, '.3f'):>8}"
        )
        print(
            f"Vibration: "
            f"x={self._fmt(imu.vibration_x, '.3f'):>8} "
            f"y={self._fmt(imu.vibration_y, '.3f'):>8} "
            f"z={self._fmt(imu.vibration_z, '.3f'):>8}"
        )
        print(
            f"Clipping: "
            f"c0={imu.clipping_0 or 0} "
            f"c1={imu.clipping_1 or 0} "
            f"c2={imu.clipping_2 or 0}"
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
