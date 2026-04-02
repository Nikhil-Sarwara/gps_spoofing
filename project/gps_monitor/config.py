# gps_monitor/config.py

MAVLINK_CONNECTION_URL = "udp:127.0.0.1:14560"

# Log at exactly 10 Hz (for ML consistency)
LOG_HZ = 10.0

# CSV logging folder
LOG_DIR = "gps_logs"

# All MAVLink messages we want to capture for structured state
MESSAGE_TYPES = [
    "GLOBAL_POSITION_INT",  # GPS position + velocity
    "GPS_RAW_INT",          # GPS quality + satellite count
    "RAW_IMU",              # Accelerometer + gyro
    "SCALED_IMU2",          # Backup IMU if RAW_IMU missing
    "HEARTBEAT",            # Armed + flight mode
    "SYS_STATUS",           # Battery + system health
    "ATTITUDE",             # Roll/pitch/yaw rates
    "VFR_HUD",              # Airspeed + climb rate
]
