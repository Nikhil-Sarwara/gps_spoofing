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
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    rollspeed_radps: Optional[float] = None
    pitchspeed_radps: Optional[float] = None
    yawspeed_radps: Optional[float] = None
    vibration_x: Optional[float] = None
    vibration_y: Optional[float] = None
    vibration_z: Optional[float] = None
    clipping_0: Optional[int] = None
    clipping_1: Optional[int] = None
    clipping_2: Optional[int] = None


@dataclass
class EstimatorState:
    vel_ratio: Optional[float] = None
    pos_horiz_ratio: Optional[float] = None
    pos_vert_ratio: Optional[float] = None
    mag_ratio: Optional[float] = None
    hagl_ratio: Optional[float] = None
    vel_innov: Optional[float] = None
    pos_horiz_innov: Optional[float] = None
    pos_vert_innov: Optional[float] = None
    mag_innov: Optional[float] = None
    hagl_innov: Optional[float] = None


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
    estimator: EstimatorState = field(default_factory=EstimatorState)
    health: VehicleHealth = field(default_factory=VehicleHealth)
    start_time: datetime = field(default_factory=datetime.now)
    connection_ok: bool = False
    last_error: Optional[str] = None

    def uptime_s(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
