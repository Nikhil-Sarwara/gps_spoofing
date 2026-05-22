#!/usr/bin/env python3
"""
setup_gps_spoof_params.py — PX4 GPS param verifier + setter for GPS spoofing.

GPS_1_CONFIG=320 is baked into SITL rcS so it applies automatically on every
fresh boot.  This script verifies the live values and force-sets any that are
wrong (covers the case where parameters.bson was restored from an old cache).

Usage:
    source venv/bin/activate
    python setup_gps_spoof_params.py
"""
import sys
import time

try:
    from pymavlink import mavutil
except ImportError:
    print("[ERROR] pymavlink not found. Run: pip install pymavlink")
    sys.exit(1)

CONNECTION  = "udpout:127.0.0.1:14550"

# param name (str) → expected value
EXPECTED = {
    "GPS_1_CONFIG":     320.0,
    "GPS_1_PROTOCOL":   1.0,
    "EKF2_GPS_CHECK":   0.0,
    "EKF2_REQ_EPH":     100.0,
    "EKF2_REQ_NSATS":   1.0,
    "EKF2_GPS_POS_NSE": 0.1,
    "EKF2_GPS_P_GATE":  10.0,
    "EKF2_GPS_V_GATE":  10.0,
}


def connect():
    print(f"[*] Connecting to PX4 at {CONNECTION} ...")
    conn = mavutil.mavlink_connection(
        CONNECTION,
        source_system=255,
        source_component=190,
        dialect="common",
    )
    # Drain until we get a heartbeat from sysid=1, component=1 (autopilot)
    deadline = time.time() + 15
    while time.time() < deadline:
        hb = conn.wait_heartbeat(blocking=True, timeout=3)
        if hb and conn.target_system == 1:
            break
        if hb:
            print(f"[~] Skipping heartbeat from sysid={conn.target_system} (waiting for sysid=1)")
    else:
        print("[ERROR] No heartbeat from PX4 autopilot (sysid=1). Is SITL running?")
        sys.exit(1)

    print(f"[+] Connected → sysid={conn.target_system}  compid={conn.target_component}\n")
    return conn


def read_param(conn, name: str, retries: int = 3) -> float | None:
    """Request a single param by name, return float value or None on timeout."""
    for attempt in range(retries):
        conn.mav.param_request_read_send(
            conn.target_system,
            conn.target_component,
            name.encode("utf-8"),
            -1,
        )
        deadline = time.time() + 2.0
        while time.time() < deadline:
            msg = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg and msg.param_id.rstrip("\x00") == name:
                return float(msg.param_value)
        time.sleep(0.1)
    return None


def set_param(conn, name: str, value: float) -> bool:
    """Set a param and confirm with ACK, return True on success."""
    conn.mav.param_set_send(
        conn.target_system,
        conn.target_component,
        name.encode("utf-8"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    deadline = time.time() + 3.0
    while time.time() < deadline:
        msg = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if msg and msg.param_id.rstrip("\x00") == name:
            return True
    return False


def main():
    conn = connect()

    # Kick off a full param list fetch first — this primes the param server
    # and ensures subsequent param_request_read calls are answered promptly.
    print("[*] Requesting full param list to prime PX4 param server...")
    conn.mav.param_request_list_send(conn.target_system, conn.target_component)
    # Drain for 3 s to let params stream in
    deadline = time.time() + 3.0
    received = 0
    while time.time() < deadline:
        msg = conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.2)
        if msg:
            received += 1
    print(f"[+] Received {received} param frames during pre-fetch.\n")

    print("[*] Verifying GPS spoofing parameters...")
    all_ok = True
    needs_set = {}

    for name, expected in EXPECTED.items():
        actual = read_param(conn, name)
        if actual is None:
            print(f"  ? NO ACK  {name:<25} — will force-set")
            needs_set[name] = expected
            all_ok = False
        else:
            ok = abs(actual - expected) < 0.5
            status = "✓ OK" if ok else "✗ WRONG"
            print(f"  {status}  {name:<25} = {actual:<10.1f}  (expected {expected})")
            if not ok:
                needs_set[name] = expected
                all_ok = False

    if needs_set:
        print(f"\n[*] Force-setting {len(needs_set)} param(s)...")
        for name, value in needs_set.items():
            ok = set_param(conn, name, value)
            status = "✓ set" if ok else "✗ set FAILED"
            print(f"  {status}  {name} = {value}")

        # Re-verify after setting
        print("\n[*] Re-verifying after force-set...")
        all_ok = True
        for name, expected in needs_set.items():
            actual = read_param(conn, name)
            if actual is not None and abs(actual - expected) < 0.5:
                print(f"  ✓ OK  {name:<25} = {actual:.1f}")
            else:
                print(f"  ✗ STILL WRONG  {name:<25} got {actual}")
                all_ok = False

    conn.close()

    print()
    if all_ok:
        print("╔" + "═" * 56 + "╗")
        print("║  ✓ All params correct — PX4 ready for GPS spoofing.    ║")
        print("║    Arm in QGC, then launch any attack from the TUI.    ║")
        print("╚" + "═" * 56 + "╝")
        sys.exit(0)
    else:
        print("╔" + "═" * 56 + "╗")
        print("║  ✗ Some params could not be set.                        ║")
        print("║    Restart PX4 SITL and click Setup PX4 again.         ║")
        print("╚" + "═" * 56 + "╝")
        sys.exit(1)


if __name__ == "__main__":
    main()
