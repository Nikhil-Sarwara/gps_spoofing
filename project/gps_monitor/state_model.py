# gps_monitor/state_model.py

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class GPSState:
    last_update: Optional[datetime] = None
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_m: Optional[float] = None
    rel_alt_m: Optional[float] = None
    vel_m_s: Optional[float] = None
    hdg_deg: Optional[float] = None
    fix_type: Optional[int] = None
    satellites_visible: Optional[int] = None
    eph: Optional[float] = None
    epv: Optional[float] = None


@dataclass
class IMUState:
    ax_mps2: Optional[float] = None
    ay_mps2: Optional[float] = None
    az_mps2: Optional[float] = None
    gx_radps: Optional[float] = None
    gy_radps: Optional[float] = None
    gz_radps: Optional[float] = None


@dataclass
class VehicleHealth:
    battery_voltage: Optional[float] = None
    battery_remaining_pct: Optional[int] = None
    armed: bool = False
    mode: str = "UNKNOWN"
    failsafe: bool = False


@dataclass
class TelemetryState:
    gps: GPSState = field(default_factory=GPSState)
    imu: IMUState = field(default_factory=IMUState)
    health: VehicleHealth = field(default_factory=VehicleHealth)
    start_time: datetime = field(default_factory=datetime.now)
    connection_ok: bool = False
    last_error: Optional[str] = None

    def uptime_s(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
