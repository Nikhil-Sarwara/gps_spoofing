#!/usr/bin/env python3
"""
gps_spoofer.py — GPS Spoofing Attack Injector for PX4 v1.17 SITL + Gazebo

How it works
------------
  Writes a North/East offset (metres) to /tmp/px4_gps_spoof.txt at <rate> Hz.
  A patched GZBridge::navSatCallback() reads this file each GPS tick and adds
  the offset to the Gazebo NavSat position before publishing to EKF2 / QGC.

  This is the only reliable method for Gazebo SITL — the MAVLink GPS_INPUT
  message (#232) is NOT parsed by the gz_bridge build of PX4 v1.17.

Attack Modes
------------
  drift       — Constant NE drift at <intensity> m/s
  ramp_drift  — Ramp 0 → <intensity> m/s over the full duration
  jump_drift  — Instant 15 m jump then drift at <intensity> m/s
  static      — Freeze at a fixed <intensity>-metre offset
  teleport    — Instant <intensity>-metre jump, held
  noise       — Gaussian noise, stddev = <intensity> m

Usage
-----
  python gps_spoofer.py --mode drift      --intensity 3.0  --duration 30
  python gps_spoofer.py --mode ramp_drift --intensity 5.0  --duration 60
  python gps_spoofer.py --mode jump_drift --intensity 2.0  --duration 45
  python gps_spoofer.py --mode static     --intensity 100  --duration 20
  python gps_spoofer.py --mode teleport   --intensity 500  --duration 15
  python gps_spoofer.py --mode noise      --intensity 50   --duration 60
"""

import argparse
import math
import os
import random
import sys
import time

# ── constants ────────────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000.0
HZ             = 10
DT             = 1.0 / HZ
SPOOF_FILE     = "/tmp/px4_gps_spoof.txt"


# ── coordinate math ──────────────────────────────────────────────────────────

def offset_latlon(lat, lon, d_north_m, d_east_m):
    """Return (lat, lon) shifted by d_north_m north and d_east_m east."""
    d_lat = math.degrees(d_north_m / EARTH_RADIUS_M)
    d_lon = math.degrees(
        d_east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat)))
    )
    return lat + d_lat, lon + d_lon


# ── spoof file I/O ───────────────────────────────────────────────────────────

def write_offset(d_north_m: float, d_east_m: float) -> None:
    """Atomically write offset so GZBridge never reads a partial file."""
    tmp = SPOOF_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{d_north_m:.6f} {d_east_m:.6f}\n")
    os.replace(tmp, SPOOF_FILE)


def clear_offset() -> None:
    """Zero the offset and remove the file so GZBridge returns to real GPS."""
    write_offset(0.0, 0.0)
    try:
        os.remove(SPOOF_FILE)
    except FileNotFoundError:
        pass


# ── attack runner ─────────────────────────────────────────────────────────────

def run_attack(mode: str, intensity: float, duration: float,
               lat: float = None, lon: float = None) -> None:
    """
    Run a GPS spoofing attack by writing North/East offsets to SPOOF_FILE.

    Parameters
    ----------
    mode      : attack mode string
    intensity : metres (drift: m/s, noise: stddev m, others: total m)
    duration  : seconds
    lat, lon  : optional fixed target for static/teleport modes
    """

    d_north = 0.0
    d_east  = 0.0
    angle   = random.uniform(0, 2 * math.pi)

    # Mode-specific setup
    if mode == "teleport":
        tp_north = intensity * math.cos(angle)
        tp_east  = intensity * math.sin(angle)
        print(f"[*] Teleport target: N{tp_north:+.1f} m  E{tp_east:+.1f} m")

    elif mode == "jump_drift":
        jump_north = 15.0 * math.cos(angle)
        jump_east  = 15.0 * math.sin(angle)
        print(f"[*] Jump+Drift: jump N{jump_north:+.1f} m  E{jump_east:+.1f} m, "
              f"then {intensity} m/s drift")

    elif mode == "static":
        if lat is not None and lon is not None:
            # convert absolute lat/lon to offset from Melbourne fallback
            base_lat, base_lon = -37.8136, 144.9631
            d_north_static = math.radians(lat - base_lat) * EARTH_RADIUS_M
            d_east_static  = (math.radians(lon - base_lon) *
                              EARTH_RADIUS_M * math.cos(math.radians(base_lat)))
        else:
            d_north_static = intensity * math.cos(angle)
            d_east_static  = intensity * math.sin(angle)
        print(f"[*] Static freeze: N{d_north_static:+.1f} m  E{d_east_static:+.1f} m")

    # 3-second warmup at zero offset (EKF settles on the real position)
    print("[*] Warmup: 3 s at real GPS position...")
    clear_offset()
    time.sleep(3.0)
    print("[+] Warmup done. Attack starting...\n")

    print(f"[*] Mode={mode}  intensity={intensity} m  "
          f"duration={duration} s  rate={HZ} Hz")
    print("[*] Ctrl+C to stop early.\n")
    print(f"[+] Writing offsets to {SPOOF_FILE}")

    t_start  = time.time()
    t_end    = t_start + duration
    count    = 0
    last_log = t_start

    try:
        while time.time() < t_end:
            elapsed = time.time() - t_start

            # ── compute offset for this tick ──────────────────────────────
            if mode == "drift":
                d_north += intensity * DT
                d_east  += intensity * DT * 0.5

            elif mode == "ramp_drift":
                rate     = intensity * min(1.0, elapsed / max(duration, 1e-9))
                d_north += rate * DT
                d_east  += rate * DT * 0.5

            elif mode == "jump_drift":
                d_north += intensity * DT
                d_east  += intensity * DT * 0.5

            elif mode == "static":
                d_north = d_north_static
                d_east  = d_east_static

            elif mode == "teleport":
                d_north = tp_north
                d_east  = tp_east

            elif mode == "noise":
                d_north = random.gauss(0, intensity)
                d_east  = random.gauss(0, intensity)

            # ── for jump_drift, add the initial jump ──────────────────────
            if mode == "jump_drift":
                write_offset(jump_north + d_north, jump_east + d_east)
            else:
                write_offset(d_north, d_east)

            count += 1

            # ── progress log every 2 s ────────────────────────────────────
            if time.time() - last_log >= 2.0:
                total_offset = math.sqrt(d_north**2 + d_east**2)
                print(
                    f"  [{elapsed:5.1f}s] → N{d_north:+.1f} m  E{d_east:+.1f} m  "
                    f"offset={total_offset:.1f} m  ticks={count}"
                )
                last_log = time.time()

            time.sleep(DT)

    except KeyboardInterrupt:
        print("\n[!] Stopped early by user.")

    finally:
        clear_offset()
        print(f"\n[+] Done. {count} spoof ticks written over {time.time()-t_start:.0f} s.")
        print("[+] GPS returned to real position (offset cleared).")


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GPS Spoofer — Gazebo SITL offset injection for PX4 v1.17"
    )
    p.add_argument(
        "--mode", default="drift",
        choices=["drift", "ramp_drift", "jump_drift", "static", "teleport", "noise"],
        help="Attack mode (default: drift)",
    )
    p.add_argument("--intensity", type=float, default=3.0,
                   help="Attack strength in metres (default: 3.0)")
    p.add_argument("--duration",  type=float, default=30.0,
                   help="Duration in seconds (default: 30)")
    p.add_argument("--lat",       type=float, default=None,
                   help="Custom target latitude  (static mode only)")
    p.add_argument("--lon",       type=float, default=None,
                   help="Custom target longitude (static mode only)")
    # kept for CLI compatibility — no longer used (Gazebo file approach)
    p.add_argument("--connection", default=None,
                   help="(deprecated — ignored, Gazebo file injection used)")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        run_attack(
            mode=args.mode,
            intensity=args.intensity,
            duration=args.duration,
            lat=args.lat,
            lon=args.lon,
        )
    except KeyboardInterrupt:
        clear_offset()
        print("\n[!] Interrupted. Offset cleared.")
        sys.exit(0)


if __name__ == "__main__":
    main()
