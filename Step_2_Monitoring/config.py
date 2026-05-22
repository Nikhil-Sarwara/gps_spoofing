# gps_monitor/config.py

MAVLINK_CONNECTION_URL = "udp:127.0.0.1:14561"

# Log at exactly 10 Hz (for ML consistency)
LOG_HZ = 10.0

# CSV logging folder
LOG_DIR = "Step_5_Data/raw"

# All MAVLink messages we want to capture for structured state
MESSAGE_TYPES = [
    "GLOBAL_POSITION_INT",  # GPS position + velocity
    "GPS_RAW_INT",          # GPS quality + satellite count
    "HEARTBEAT",            # Armed + flight mode
    "SYS_STATUS",           # Battery + system health
    "ATTITUDE",             # Orientation angles + angular rates
    "VIBRATION",            # Vibration levels (indicator of GPS issues)
    "ESTIMATOR_STATUS",     # EKF state innovation (GPS consistency)
    "VFR_HUD",              # Airspeed + climb rate
]
