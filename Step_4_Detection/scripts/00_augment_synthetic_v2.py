#!/usr/bin/env python3
"""
00_augment_synthetic_v2.py — Realistic synthetic GPS telemetry generator.

Improvements over v1:
  - Subtle spoofing attacks (slow drifts, small offsets, gradual capture)
  - More noise/variation in normal flight (turbulence, GPS jitter)
  - Edge cases: high-speed cruise with low angular rates (H8 false-positive risk)
  - Diverse attack profiles: drift, ramp, noise injection, partial spoof
  - Realistic sensor noise profiles per terrain type

Usage:
    python 00_augment_synthetic_v2.py --flights-per-terrain 20 --rows-per-flight 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

WINDOW_LEN = 30
STRIDE = 15
BASE_ISO = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)

EXPECTED_COLS = [
    "time_s", "lat_deg", "lon_deg", "alt_m", "rel_alt_m", "vel_m_s", "hdg_deg",
    "fix_type", "satellites_visible", "eph_m", "epv_m",
    "roll_deg", "pitch_deg", "yaw_deg",
    "rollspeed_radps", "pitchspeed_radps", "yawspeed_radps",
    "vibration_x", "vibration_y", "vibration_z",
    "clipping_0", "clipping_1", "clipping_2",
    "vel_ratio", "pos_horiz_ratio", "pos_vert_ratio",
    "vel_innov", "pos_horiz_innov", "pos_vert_innov",
    "battery_voltage", "battery_remaining_pct",
    "armed", "mode", "failsafe", "connection_ok", "last_update_iso",
    "is_stale_repeat", "dt", "label",
]

TERRAINS: dict[str, dict[str, Any]] = {
    "flat": {
        "lat_range": (47.38, 47.40), "lon_range": (8.53, 8.56),
        "lat_bbox": (47.35, 47.43), "lon_bbox": (8.50, 8.59),
        "alt_range": (450.0, 520.0), "alt_clamp": (440.0, 560.0),
        "rel_range": (5.0, 50.0),
        "vel_mean": 8.5, "vel_range": (5.0, 12.0),
        "roll_max": 8.0, "pitch_max": 6.0,
        "sats_range": (10, 14), "eph_range": (0.3, 1.2), "epv_range": (0.5, 2.0),
        "vib_range": (0.001, 0.05), "drain_rate": 0.003,
    },
    "mountain": {
        "lat_range": (46.50, 46.60), "lon_range": (7.90, 8.10),
        "lat_bbox": (46.47, 46.63), "lon_bbox": (7.87, 8.13),
        "alt_range": (1800.0, 2500.0), "alt_clamp": (1600.0, 2800.0),
        "rel_range": (10.0, 80.0),
        "vel_mean": 5.5, "vel_range": (3.0, 8.0),
        "roll_max": 15.0, "pitch_max": 10.0,
        "sats_range": (7, 11), "eph_range": (0.8, 2.5), "epv_range": (1.0, 4.0),
        "vib_range": (0.01, 0.15), "drain_rate": 0.005,
    },
    "sea": {
        "lat_range": (43.50, 43.70), "lon_range": (7.20, 7.50),
        "lat_bbox": (43.47, 43.73), "lon_bbox": (7.17, 7.53),
        "alt_range": (10.0, 80.0), "alt_clamp": (0.5, 120.0),
        "rel_range": (5.0, 40.0),
        "vel_mean": 13.0, "vel_range": (8.0, 18.0),
        "roll_max": 5.0, "pitch_max": 4.0,
        "sats_range": (12, 15), "eph_range": (0.2, 0.8), "epv_range": (0.3, 1.2),
        "vib_range": (0.001, 0.03), "drain_rate": 0.003,
    },
}


def smooth_trajectory(n_rows, start_lat, start_lon, t, rng):
    hdg = np.cumsum(rng.normal(0, 0.5, n_rows)) % 360
    speed = np.clip(
        np.cumsum(rng.normal(0, 0.3, n_rows)) + t["vel_mean"],
        t["vel_range"][0], t["vel_range"][1],
    )
    dlat = speed * np.cos(np.radians(hdg)) * 0.1 / 111320
    dlon = speed * np.sin(np.radians(hdg)) * 0.1 / (111320 * np.cos(np.radians(start_lat)))
    lat = np.clip(start_lat + np.cumsum(dlat), t["lat_bbox"][0], t["lat_bbox"][1])
    lon = np.clip(start_lon + np.cumsum(dlon), t["lon_bbox"][0], t["lon_bbox"][1])
    return lat, lon, hdg, speed


def smooth_signal(n_rows, mean, sigma, lo, hi, rng):
    raw = mean + np.cumsum(rng.normal(0, sigma, n_rows))
    return np.clip(raw, lo, hi)


# ---------------------------------------------------------------------------
# Attack profiles — more diverse and realistic than v1
# ---------------------------------------------------------------------------
ATTACK_PROFILES = [
    "gradual_drift",       # Slow NE drift, 0.5-2 m/s — hard to detect
    "abrupt_offset",       # Instant 30-80m jump, then hold
    "ramp_capture",        # Gradually increase offset over 30-60s
    "noise_injection",     # Add Gaussian noise to GPS position
    "partial_spoof",       # Spoof only altitude, keep lat/lon correct
    "meaconing_delay",     # Delay GPS updates by 2-5s (simulated retransmission)
    "step_drift",          # Step offset + slow drift (v1 style)
]


def place_spoof_events(n_rows, n_events, rng):
    """Place non-overlapping spoof events with varied durations."""
    margin = int(n_rows * 0.15)
    safe_lo, safe_hi = margin, n_rows - margin
    min_gap = 80

    events = []
    attempts = 0
    while len(events) < n_events and attempts < 1000:
        attempts += 1
        dur = int(rng.integers(100, 400))  # 10-40 seconds at 10Hz
        start = int(rng.integers(safe_lo, safe_hi - dur))
        end = start + dur
        ok = all(end + min_gap <= es or start >= ee + min_gap for es, ee in events)
        if ok:
            events.append((start, end))
    events.sort(key=lambda x: x[0])
    return events


def apply_attack(lat, lon, alt, speed, eph, epv, sats, vel_innov, pos_horiz_innov,
                 vel_ratio, pos_horiz_ratio, failsafe, fix_type, is_stale,
                 ev_start, ev_end, rng, profile):
    """Apply a specific attack profile to the given slice."""
    ev_slice = slice(ev_start, ev_end)
    dur = ev_end - ev_start

    if profile == "gradual_drift":
        # Slow linear drift — hard for heuristics to catch
        drift_rate = rng.uniform(0.5, 2.0)
        angle = rng.uniform(0, 2 * np.pi)
        for i in range(ev_start, ev_end):
            t_offset = (i - ev_start) * 0.1  # seconds since start
            d_north = drift_rate * t_offset * np.cos(angle)
            d_east = drift_rate * t_offset * np.sin(angle)
            lat[i] += d_north / 111320
            lon[i] += d_east / (111320 * np.cos(np.radians(lat[i])))
        speed[ev_slice] += rng.uniform(0.5, 2.0, dur)
        eph[ev_slice] = np.clip(eph[ev_slice] + rng.uniform(0.5, 2.0, dur), 0.1, 50)
        # Keep IMU calm — this is the key spoofing signature
        # But add small noise to make it less obvious
        vel_innov[ev_slice] = np.clip(rng.uniform(0.3, 1.5, dur), 0.0, 5.0)
        pos_horiz_innov[ev_slice] = np.clip(rng.uniform(0.2, 1.2, dur), 0.0, 5.0)

    elif profile == "abrupt_offset":
        # Instant jump, then hold
        lat[ev_slice] += rng.uniform(-0.001, 0.001)
        lon[ev_slice] += rng.uniform(-0.001, 0.001)
        speed[ev_slice] = np.clip(rng.uniform(20.0, 40.0, dur), 15.0, 50.0)
        eph[ev_slice] = np.clip(rng.uniform(5.0, 15.0, dur), 5.0, 50)
        epv[ev_slice] = np.clip(rng.uniform(8.0, 20.0, dur), 8.0, 80)
        sats[ev_slice] = rng.integers(3, 6, dur)
        vel_innov[ev_slice] = np.clip(rng.uniform(1.5, 3.0, dur), 1.0, 5.0)
        pos_horiz_innov[ev_slice] = np.clip(rng.uniform(1.2, 2.5, dur), 1.0, 5.0)
        vel_ratio[ev_slice] = np.clip(rng.uniform(1.0, 2.5, dur), 1.0, 5.0)
        pos_horiz_ratio[ev_slice] = np.clip(rng.uniform(1.0, 2.5, dur), 1.0, 5.0)
        failsafe[ev_slice] = 1
        fix_type[ev_slice] = 1

    elif profile == "ramp_capture":
        # Gradually increase offset
        max_offset = rng.uniform(0.002, 0.008)
        for i in range(ev_start, ev_end):
            progress = (i - ev_start) / dur
            offset = max_offset * progress
            lat[i] += offset * rng.choice([-1, 1])
            lon[i] += offset * rng.choice([-1, 1])
        speed[ev_slice] = np.clip(np.linspace(speed[ev_start], speed[ev_start] + 15, dur), 0, 50)
        eph[ev_slice] = np.clip(np.linspace(eph[ev_start], 8.0, dur), 0.1, 50)
        vel_innov[ev_slice] = np.clip(np.linspace(0.2, 2.0, dur), 0.0, 5.0)
        pos_horiz_innov[ev_slice] = np.clip(np.linspace(0.2, 1.8, dur), 0.0, 5.0)
        failsafe[ev_start:ev_start + dur // 3] = 1  # Only trigger failsafe at start

    elif profile == "noise_injection":
        # Add Gaussian noise to GPS position — subtle
        noise_scale = rng.uniform(0.0005, 0.002)
        lat[ev_slice] += rng.normal(0, noise_scale, dur)
        lon[ev_slice] += rng.normal(0, noise_scale, dur)
        eph[ev_slice] = np.clip(eph[ev_slice] + rng.uniform(1.0, 4.0, dur), 0.1, 50)
        vel_innov[ev_slice] = np.clip(rng.uniform(0.5, 1.8, dur), 0.0, 5.0)
        pos_horiz_innov[ev_slice] = np.clip(rng.uniform(0.4, 1.5, dur), 0.0, 5.0)

    elif profile == "partial_spoof":
        # Spoof only altitude — keep lat/lon correct
        alt_offset = rng.uniform(50, 200) * rng.choice([-1, 1])
        alt[ev_slice] += alt_offset
        eph[ev_slice] = np.clip(eph[ev_slice] + rng.uniform(1.0, 3.0, dur), 0.1, 50)
        vel_innov[ev_slice] = np.clip(rng.uniform(0.3, 1.0, dur), 0.0, 5.0)

    elif profile == "meaconing_delay":
        # Simulate delayed GPS retransmission — shift position backward
        delay_samples = int(rng.uniform(20, 50))  # 2-5 second delay
        for i in range(ev_start + delay_samples, ev_end):
            lat[i] = lat[i - delay_samples]
            lon[i] = lon[i - delay_samples]
        eph[ev_slice] = np.clip(eph[ev_slice] + rng.uniform(2.0, 6.0, dur), 0.1, 50)
        vel_innov[ev_slice] = np.clip(rng.uniform(0.8, 2.0, dur), 0.0, 5.0)

    elif profile == "step_drift":
        # Step offset + slow drift (v1 style, kept for comparison)
        lat_jump = rng.choice([-1, 1]) * rng.uniform(0.005, 0.020)
        lon_jump = rng.choice([-1, 1]) * rng.uniform(0.005, 0.020)
        lat[ev_slice] += lat_jump
        lon[ev_slice] += lon_jump
        speed[ev_slice] = np.clip(rng.uniform(25.0, 45.0, dur), 23.0, 50.0)
        eph[ev_slice] = np.clip(rng.uniform(5.0, 15.0, dur), 5.0, 50)
        epv[ev_slice] = np.clip(rng.uniform(8.0, 20.0, dur), 8.0, 80)
        sats[ev_slice] = rng.integers(3, 6, dur)
        vel_innov[ev_slice] = np.clip(rng.uniform(1.5, 3.0, dur), 1.0, 5.0)
        pos_horiz_innov[ev_slice] = np.clip(rng.uniform(1.2, 2.5, dur), 1.0, 5.0)
        vel_ratio[ev_slice] = np.clip(rng.uniform(1.0, 2.5, dur), 1.0, 5.0)
        pos_horiz_ratio[ev_slice] = np.clip(rng.uniform(1.0, 2.5, dur), 1.0, 5.0)
        failsafe[ev_slice] = 1
        fix_type[ev_slice] = 1

    return (lat, lon, alt, speed, eph, epv, sats, vel_innov, pos_horiz_innov,
            vel_ratio, pos_horiz_ratio, failsafe, fix_type, is_stale)


def generate_flight(terrain_name, flight_idx, n_rows, seed):
    t = TERRAINS[terrain_name]
    rng = np.random.default_rng(seed)

    time_s = np.round(np.arange(1, n_rows + 1) * 0.1, 3)
    start_lat = float(rng.uniform(*t["lat_range"]))
    start_lon = float(rng.uniform(*t["lon_range"]))
    lat, lon, hdg, speed = smooth_trajectory(n_rows, start_lat, start_lon, t, rng)

    # Altitude
    alt_start = float(rng.uniform(*t["alt_range"]))
    alt = np.clip(alt_start + np.cumsum(rng.normal(0, 0.3, n_rows)), t["alt_clamp"][0], t["alt_clamp"][1])
    rel_alt_start = float(rng.uniform(*t["rel_range"]))
    rel_alt = np.clip(rel_alt_start + np.cumsum(rng.normal(0, 0.2, n_rows)), 0.5, 120.0)
    if terrain_name in ("flat", "mountain"):
        rel_alt = np.minimum(rel_alt, alt)

    # Attitude
    roll = np.clip(np.cumsum(rng.normal(0, 0.3, n_rows)), -t["roll_max"], t["roll_max"])
    pitch = np.clip(np.cumsum(rng.normal(0, 0.2, n_rows)), -t["pitch_max"], t["pitch_max"])
    yaw = hdg % 360.0

    rollspeed = np.clip(rng.normal(0, 0.05, n_rows), -2.0, 2.0)
    pitchspeed = np.clip(rng.normal(0, 0.05, n_rows), -2.0, 2.0)
    yawspeed = np.clip(rng.normal(0, 0.05, n_rows), -3.0, 3.0)

    # GPS quality
    sats = np.clip(np.round(smooth_signal(n_rows, np.mean(t["sats_range"]), 0.5, t["sats_range"][0], t["sats_range"][1], rng)).astype(int), t["sats_range"][0], t["sats_range"][1]).astype(int)
    eph = np.clip(smooth_signal(n_rows, np.mean(t["eph_range"]), 0.05, t["eph_range"][0], t["eph_range"][1], rng), 0.1, 50.0)
    epv = np.clip(smooth_signal(n_rows, np.mean(t["epv_range"]), 0.08, t["epv_range"][0], t["epv_range"][1], rng), 0.1, 80.0)

    # Vibration
    vib_x = np.clip(smooth_signal(n_rows, np.mean(t["vib_range"]), 0.002, t["vib_range"][0], t["vib_range"][1], rng), 0.0, 0.3)
    vib_y = np.clip(smooth_signal(n_rows, np.mean(t["vib_range"]), 0.002, t["vib_range"][0], t["vib_range"][1], rng), 0.0, 0.3)
    vib_z = np.clip(smooth_signal(n_rows, np.mean(t["vib_range"]), 0.002, t["vib_range"][0], t["vib_range"][1], rng), 0.0, 0.3)

    # EKF ratios
    vel_ratio = np.clip(smooth_signal(n_rows, 0.3, 0.02, 0.0, 0.9, rng), 0.0, 0.9)
    pos_horiz_ratio = np.clip(smooth_signal(n_rows, 0.3, 0.02, 0.0, 0.9, rng), 0.0, 0.9)
    pos_vert_ratio = np.clip(smooth_signal(n_rows, 0.2, 0.01, 0.0, 0.9, rng), 0.0, 0.9)
    vel_innov = np.clip(smooth_signal(n_rows, 0.2, 0.02, 0.0, 0.9, rng), 0.0, 0.9)
    pos_horiz_innov = np.clip(smooth_signal(n_rows, 0.2, 0.02, 0.0, 0.9, rng), 0.0, 0.9)
    pos_vert_innov = np.clip(smooth_signal(n_rows, 0.15, 0.01, 0.0, 0.9, rng), 0.0, 0.9)

    # Battery
    pct_start = 100.0
    pct_end = max(55.0, pct_start - t["drain_rate"] * n_rows)
    battery_pct = np.linspace(pct_start, pct_end, n_rows)
    battery_voltage = np.clip(10.5 + (battery_pct / 100.0) * 2.1, 10.5, 12.65)

    # Static
    fix_type = np.full(n_rows, 3, dtype=int)
    clipping_0 = np.zeros(n_rows, dtype=int)
    clipping_1 = np.zeros(n_rows, dtype=int)
    clipping_2 = np.zeros(n_rows, dtype=int)
    armed = np.ones(n_rows, dtype=int)
    failsafe = np.zeros(n_rows, dtype=int)
    connection_ok = np.ones(n_rows, dtype=int)
    is_stale = np.zeros(n_rows, dtype=int)

    # --- Inject spoof events with varied attack profiles ---
    n_events = int(rng.integers(1, 5))  # 1-4 events per flight
    events = place_spoof_events(n_rows, n_events, rng)
    segments = []
    prev_end = 0

    for ev_start, ev_end in events:
        # Normal segment before this event
        if ev_start > prev_end:
            segments.append({
                "start_s": round(time_s[prev_end], 3),
                "end_s": round(time_s[ev_start - 1], 3),
                "label": 0, "reason": "normal",
            })

        # Choose attack profile
        profile = rng.choice(ATTACK_PROFILES)

        (lat, lon, alt, speed, eph, epv, sats, vel_innov, pos_horiz_innov,
         vel_ratio, pos_horiz_ratio, failsafe, fix_type, is_stale) = apply_attack(
            lat, lon, alt, speed, eph, epv, sats, vel_innov, pos_horiz_innov,
            vel_ratio, pos_horiz_ratio, failsafe, fix_type, is_stale,
            ev_start, ev_end, rng, profile,
        )

        segments.append({
            "start_s": round(time_s[ev_start], 3),
            "end_s": round(time_s[ev_end - 1], 3),
            "label": 1, "reason": f"gps_spoof_{profile}",
        })
        prev_end = ev_end

    # Final normal segment
    if prev_end < n_rows:
        segments.append({
            "start_s": round(time_s[prev_end], 3),
            "end_s": round(time_s[-1], 3),
            "label": 0, "reason": "normal",
        })

    # Make segments contiguous
    for i in range(1, len(segments)):
        segments[i]["start_s"] = segments[i - 1]["end_s"]

    # Post-spoof clamps
    lat = np.clip(lat, -90.0, 90.0)
    lon = np.clip(lon, -180.0, 180.0)
    speed = np.clip(speed, 0.0, 50.0)
    eph = np.clip(eph, 0.1, 50.0)
    epv = np.clip(epv, 0.1, 80.0)
    sats = np.clip(sats, 0, 20).astype(int)
    vel_innov = np.clip(vel_innov, 0.0, 5.0)
    pos_horiz_innov = np.clip(pos_horiz_innov, 0.0, 5.0)
    pos_vert_innov = np.clip(pos_vert_innov, 0.0, 5.0)
    vel_ratio = np.clip(vel_ratio, 0.0, 5.0)
    pos_horiz_ratio = np.clip(pos_horiz_ratio, 0.0, 5.0)
    pos_vert_ratio = np.clip(pos_vert_ratio, 0.0, 5.0)
    roll = np.clip(roll, -45.0, 45.0)
    pitch = np.clip(pitch, -30.0, 30.0)

    low_sat_mask = sats <= 4
    fix_type[low_sat_mask & (fix_type == 3)] = 2

    last_update_iso = [
        (BASE_ISO + timedelta(seconds=float(ts))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        for ts in time_s
    ]

    df = pd.DataFrame({
        "time_s": time_s, "lat_deg": lat, "lon_deg": lon, "alt_m": alt,
        "rel_alt_m": rel_alt, "vel_m_s": speed, "hdg_deg": hdg,
        "fix_type": fix_type.astype(int), "satellites_visible": sats.astype(int),
        "eph_m": eph, "epv_m": epv,
        "roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw,
        "rollspeed_radps": rollspeed, "pitchspeed_radps": pitchspeed, "yawspeed_radps": yawspeed,
        "vibration_x": vib_x, "vibration_y": vib_y, "vibration_z": vib_z,
        "clipping_0": clipping_0.astype(int), "clipping_1": clipping_1.astype(int), "clipping_2": clipping_2.astype(int),
        "vel_ratio": vel_ratio, "pos_horiz_ratio": pos_horiz_ratio, "pos_vert_ratio": pos_vert_ratio,
        "vel_innov": vel_innov, "pos_horiz_innov": pos_horiz_innov, "pos_vert_innov": pos_vert_innov,
        "battery_voltage": battery_voltage, "battery_remaining_pct": battery_pct,
        "armed": armed.astype(int), "mode": ["AUTO"] * n_rows,
        "failsafe": failsafe.astype(int), "connection_ok": connection_ok.astype(int),
        "last_update_iso": last_update_iso, "is_stale_repeat": is_stale.astype(int),
        "dt": np.full(n_rows, 0.1),
    })

    return df, segments


def count_windows(n_rows, labels):
    n0, n1 = 0, 0
    i = 0
    while i + WINDOW_LEN <= n_rows:
        win = labels[i:i + WINDOW_LEN]
        w_lbl = 1 if win.mean() >= 0.5 else 0
        if w_lbl == 0:
            n0 += 1
        else:
            n1 += 1
        i += STRIDE
    return n0, n1


def row_labels_from_segments(segs, time_s):
    labels = np.zeros(len(time_s), dtype=int)
    for s in segs:
        mask = (time_s >= s["start_s"]) & (time_s <= s["end_s"])
        labels[mask] = s["label"]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Generate realistic synthetic GPS telemetry v2.")
    parser.add_argument("--output-dir", default="Step_5_Data/synthetic_v2")
    parser.add_argument("--flights-per-terrain", type=int, default=20)
    parser.add_argument("--rows-per-flight", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--terrain", choices=["flat", "mountain", "sea"], default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    terrains = [args.terrain] if args.terrain else ["flat", "mountain", "sea"]
    summary = []

    print("Generating realistic synthetic GPS telemetry v2...\n")

    for t_idx, terrain in enumerate(terrains):
        total_windows = 0
        total_rows = 0
        attack_profile_counts = {}

        for f_idx in range(1, args.flights_per_terrain + 1):
            seed = args.seed + t_idx * 100 + f_idx
            csv_name = f"{terrain}_flight_{f_idx:02d}_cleaned.csv"
            jsn_name = f"{terrain}_flight_{f_idx:02d}_auto_segments.json"
            csv_path = os.path.join(args.output_dir, csv_name)
            jsn_path = os.path.join(args.output_dir, jsn_name)

            df, segs = generate_flight(terrain, f_idx, args.rows_per_flight, seed)
            row_lbls = row_labels_from_segments(segs, df["time_s"].values)
            df["label"] = row_lbls

            n0, n1 = count_windows(len(df), row_lbls)
            n_win = n0 + n1
            total_windows += n_win
            total_rows += len(df)

            # Count attack profiles
            for s in segs:
                if s["label"] == 1:
                    reason = s.get("reason", "unknown")
                    attack_profile_counts[reason] = attack_profile_counts.get(reason, 0) + 1

            df.to_csv(csv_path, index=False)
            with open(jsn_path, "w") as f:
                json.dump(segs, f, indent=2)

            print(f"    - {csv_name}: {len(df)} rows, {n_win} windows ({n0} normal, {n1} anomaly)")

        summary.append({
            "terrain": terrain, "flights": args.flights_per_terrain,
            "total_rows": total_rows, "total_windows": total_windows,
            "attack_profiles": attack_profile_counts,
        })

    print("\n" + "-" * 60)
    for s in summary:
        print(f"Terrain {s['terrain']:<10}: {s['flights']} flights, {s['total_windows']} windows")
        print(f"  Attack profiles: {s['attack_profiles']}")
    print("-" * 60)

    # Save summary
    summary_path = os.path.join(args.output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print("✅ Done!")


if __name__ == "__main__":
    sys.exit(main())
