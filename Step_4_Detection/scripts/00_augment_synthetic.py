"""
00_augment_synthetic.py
-----------------------
Generates synthetic GPS telemetry CSVs for flat / mountain / sea terrains.
Outputs *_cleaned.csv + *_auto_segments.json to data/synthetic/.
All 15 boundary/integrity guards are enforced before writing.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_LEN = 30
STRIDE     = 15

BASE_ISO   = datetime(2026, 5, 1, 0, 0, 0, tzinfo=timezone.utc)

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

# ---------------------------------------------------------------------------
# Terrain definitions
# ---------------------------------------------------------------------------
TERRAINS: dict[str, dict[str, Any]] = {
    "flat": {
        "lat_range":  (47.38, 47.40),
        "lon_range":  (8.53,  8.56),
        "lat_bbox":   (47.35, 47.43),
        "lon_bbox":   (8.50,  8.59),
        "alt_range":  (450.0, 520.0),
        "alt_clamp":  (440.0, 560.0),
        "rel_range":  (5.0,   50.0),
        "vel_mean":   8.5,
        "vel_range":  (5.0,  12.0),
        "roll_max":   8.0,
        "pitch_max":  6.0,
        "sats_range": (10, 14),
        "eph_range":  (0.3, 1.2),
        "epv_range":  (0.5, 2.0),
        "vib_range":  (0.001, 0.05),
        "drain_rate": 0.003,
    },
    "mountain": {
        "lat_range":  (46.50, 46.60),
        "lon_range":  (7.90,  8.10),
        "lat_bbox":   (46.47, 46.63),
        "lon_bbox":   (7.87,  8.13),
        "alt_range":  (1800.0, 2500.0),
        "alt_clamp":  (1600.0, 2800.0),
        "rel_range":  (10.0,   80.0),
        "vel_mean":   5.5,
        "vel_range":  (3.0,  8.0),
        "roll_max":   15.0,
        "pitch_max":  10.0,
        "sats_range": (7, 11),
        "eph_range":  (0.8, 2.5),
        "epv_range":  (1.0, 4.0),
        "vib_range":  (0.01, 0.15),
        "drain_rate": 0.005,
    },
    "sea": {
        "lat_range":  (43.50, 43.70),
        "lon_range":  (7.20,  7.50),
        "lat_bbox":   (43.47, 43.73),
        "lon_bbox":   (7.17,  7.53),
        "alt_range":  (10.0,  80.0),
        "alt_clamp":  (0.5,  120.0),
        "rel_range":  (5.0,   40.0),
        "vel_mean":   13.0,
        "vel_range":  (8.0,  18.0),
        "roll_max":   5.0,
        "pitch_max":  4.0,
        "sats_range": (12, 15),
        "eph_range":  (0.2, 0.8),
        "epv_range":  (0.3, 1.2),
        "vib_range":  (0.001, 0.03),
        "drain_rate": 0.003,
    },
}


# ---------------------------------------------------------------------------
# Smooth trajectory
# ---------------------------------------------------------------------------
def smooth_trajectory(
    n_rows: int,
    start_lat: float,
    start_lon: float,
    t: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hdg = np.cumsum(rng.normal(0, 0.5, n_rows)) % 360
    speed = np.clip(
        np.cumsum(rng.normal(0, 0.3, n_rows)) + t["vel_mean"],
        t["vel_range"][0],
        t["vel_range"][1],
    )
    dlat = speed * np.cos(np.radians(hdg)) * 0.1 / 111320
    dlon = speed * np.sin(np.radians(hdg)) * 0.1 / (
        111320 * np.cos(np.radians(start_lat))
    )
    lat = np.clip(start_lat + np.cumsum(dlat), t["lat_bbox"][0], t["lat_bbox"][1])
    lon = np.clip(start_lon + np.cumsum(dlon), t["lon_bbox"][0], t["lon_bbox"][1])
    return lat, lon, hdg, speed


def smooth_signal(
    n_rows: int,
    mean: float,
    sigma: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Smooth autocorrelated signal via cumsum drift."""
    raw = mean + np.cumsum(rng.normal(0, sigma, n_rows))
    return np.clip(raw, lo, hi)


# ---------------------------------------------------------------------------
# Spoof event placement
# ---------------------------------------------------------------------------
def place_spoof_events(
    n_rows: int,
    n_events: int,
    event_dur_range: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Return list of (start_row, end_row) non-overlapping spoof events."""
    margin   = int(n_rows * 0.15)
    safe_lo  = margin
    safe_hi  = n_rows - margin
    min_gap  = 50

    events: list[tuple[int, int]] = []
    attempts = 0
    while len(events) < n_events and attempts < 1000:
        attempts += 1
        dur = int(rng.integers(event_dur_range[0], event_dur_range[1] + 1))
        start = int(rng.integers(safe_lo, safe_hi - dur))
        end   = start + dur

        # Check no overlap with existing events (min_gap rows between)
        ok = all(
            end + min_gap <= es or start >= ee + min_gap
            for es, ee in events
        )
        if ok:
            events.append((start, end))

    events.sort(key=lambda x: x[0])
    return events


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def generate_flight(
    terrain_name: str,
    flight_idx: int,
    n_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    t   = TERRAINS[terrain_name]
    rng = np.random.default_rng(seed)

    # ---- time axis --------------------------------------------------------
    time_s = np.round(np.arange(1, n_rows + 1) * 0.1, 3)

    # ---- trajectory -------------------------------------------------------
    start_lat = float(rng.uniform(*t["lat_range"]))
    start_lon = float(rng.uniform(*t["lon_range"]))
    lat, lon, hdg, speed = smooth_trajectory(n_rows, start_lat, start_lon, t, rng)

    # ---- altitude (smooth) ------------------------------------------------
    alt_start = float(rng.uniform(*t["alt_range"]))
    alt = np.clip(
        alt_start + np.cumsum(rng.normal(0, 0.3, n_rows)),
        t["alt_clamp"][0], t["alt_clamp"][1],
    )
    rel_alt_start = float(rng.uniform(*t["rel_range"]))
    rel_alt = np.clip(
        rel_alt_start + np.cumsum(rng.normal(0, 0.2, n_rows)),
        0.5, 120.0,
    )
    # Guard: rel_alt_m <= alt_m for flat/mountain
    if terrain_name in ("flat", "mountain"):
        rel_alt = np.minimum(rel_alt, alt)

    # ---- attitude ---------------------------------------------------------
    roll  = np.clip(np.cumsum(rng.normal(0, 0.3, n_rows)), -t["roll_max"],  t["roll_max"])
    pitch = np.clip(np.cumsum(rng.normal(0, 0.2, n_rows)), -t["pitch_max"], t["pitch_max"])
    yaw   = hdg % 360.0  # yaw tracks heading
    yaw   = np.where(yaw >= 360.0, 359.99, yaw)
    yaw   = np.where(yaw < 0.0,    0.0,    yaw)

    rollspeed  = np.clip(rng.normal(0, 0.05, n_rows), -2.0, 2.0)
    pitchspeed = np.clip(rng.normal(0, 0.05, n_rows), -2.0, 2.0)
    yawspeed   = np.clip(rng.normal(0, 0.05, n_rows), -3.0, 3.0)

    # ---- GPS quality (normal) --------------------------------------------
    sats = np.clip(
        np.round(
            smooth_signal(n_rows,
                          np.mean(t["sats_range"]), 0.5,
                          t["sats_range"][0], t["sats_range"][1], rng)
        ).astype(int),
        t["sats_range"][0], t["sats_range"][1],
    ).astype(int)

    eph = np.clip(
        smooth_signal(n_rows, np.mean(t["eph_range"]), 0.05,
                      t["eph_range"][0], t["eph_range"][1], rng),
        0.1, 50.0,
    )
    epv = np.clip(
        smooth_signal(n_rows, np.mean(t["epv_range"]), 0.08,
                      t["epv_range"][0], t["epv_range"][1], rng),
        0.1, 80.0,
    )

    # ---- vibration --------------------------------------------------------
    vib_x = np.clip(
        smooth_signal(n_rows, np.mean(t["vib_range"]), 0.002,
                      t["vib_range"][0], t["vib_range"][1], rng),
        0.0, 0.3,
    )
    vib_y = np.clip(
        smooth_signal(n_rows, np.mean(t["vib_range"]), 0.002,
                      t["vib_range"][0], t["vib_range"][1], rng),
        0.0, 0.3,
    )
    vib_z = np.clip(
        smooth_signal(n_rows, np.mean(t["vib_range"]), 0.002,
                      t["vib_range"][0], t["vib_range"][1], rng),
        0.0, 0.3,
    )

    # ---- EKF ratios (normal hard-capped at 0.9) ---------------------------
    vel_ratio       = np.clip(smooth_signal(n_rows, 0.3, 0.02, 0.0, 0.9, rng), 0.0, 0.9)
    pos_horiz_ratio = np.clip(smooth_signal(n_rows, 0.3, 0.02, 0.0, 0.9, rng), 0.0, 0.9)
    pos_vert_ratio  = np.clip(smooth_signal(n_rows, 0.2, 0.01, 0.0, 0.9, rng), 0.0, 0.9)
    vel_innov       = np.clip(smooth_signal(n_rows, 0.2, 0.02, 0.0, 0.9, rng), 0.0, 0.9)
    pos_horiz_innov = np.clip(smooth_signal(n_rows, 0.2, 0.02, 0.0, 0.9, rng), 0.0, 0.9)
    pos_vert_innov  = np.clip(smooth_signal(n_rows, 0.15, 0.01, 0.0, 0.9, rng), 0.0, 0.9)

    # ---- battery (monotone decreasing) -----------------------------------
    pct_start  = 100.0
    pct_end    = max(55.0, pct_start - t["drain_rate"] * n_rows)
    battery_pct     = np.linspace(pct_start, pct_end, n_rows)
    battery_voltage = np.clip(10.5 + (battery_pct / 100.0) * 2.1, 10.5, 12.65)

    # ---- static fields ----------------------------------------------------
    fix_type       = np.full(n_rows, 3, dtype=int)
    clipping_0     = np.zeros(n_rows, dtype=int)
    clipping_1     = np.zeros(n_rows, dtype=int)
    clipping_2     = np.zeros(n_rows, dtype=int)
    armed          = np.ones(n_rows, dtype=int)
    failsafe       = np.zeros(n_rows, dtype=int)
    connection_ok  = np.ones(n_rows, dtype=int)
    is_stale       = np.zeros(n_rows, dtype=int)

    # ---- inject spoof events ----------------------------------------------
    # Adjusted for more Normal data: 1 to 3 events over 3000 rows
    n_events = int(rng.integers(1, 4))
    events   = place_spoof_events(n_rows, n_events, (150, 300), rng)
    segments: list[dict] = []

    prev_end = 0
    for ev_start, ev_end in events:
        # Normal segment before this event
        if ev_start > prev_end:
            seg_start_s = round(time_s[prev_end], 3)
            seg_end_s   = round(time_s[ev_start - 1], 3)
            segments.append({"start_s": seg_start_s, "end_s": seg_end_s,
                              "label": 0, "reason": "normal"})

        # Spoof event
        lat_jump = float(rng.choice([-1, 1])) * float(rng.uniform(0.005, 0.020))
        lon_jump = float(rng.choice([-1, 1])) * float(rng.uniform(0.005, 0.020))

        ev_slice = slice(ev_start, ev_end)
        lat[ev_slice]            = np.clip(lat[ev_slice] + lat_jump, -90.0, 90.0)
        lon[ev_slice]            = np.clip(lon[ev_slice] + lon_jump, -180.0, 180.0)
        speed[ev_slice]          = np.clip(rng.uniform(25.0, 45.0, ev_end - ev_start), 23.0, 50.0)
        eph[ev_slice]            = np.clip(rng.uniform(5.0, 15.0, ev_end - ev_start), 5.0, 50.0)
        epv[ev_slice]            = np.clip(rng.uniform(8.0, 20.0, ev_end - ev_start), 8.0, 80.0)
        sats[ev_slice]           = rng.integers(3, 6, ev_end - ev_start)
        vel_innov[ev_slice]      = np.clip(rng.uniform(1.5, 3.0, ev_end - ev_start), 1.0, 5.0)
        pos_horiz_innov[ev_slice]= np.clip(rng.uniform(1.2, 2.5, ev_end - ev_start), 1.0, 5.0)
        vel_ratio[ev_slice]      = np.clip(rng.uniform(1.0, 2.5, ev_end - ev_start), 1.0, 5.0)
        pos_horiz_ratio[ev_slice]= np.clip(rng.uniform(1.0, 2.5, ev_end - ev_start), 1.0, 5.0)
        failsafe[ev_slice]       = 1
        fix_type[ev_slice]       = 1
        clipping_0[ev_slice]     = rng.integers(0, 4, ev_end - ev_start)
        clipping_1[ev_slice]     = rng.integers(0, 4, ev_end - ev_start)
        clipping_2[ev_slice]     = rng.integers(0, 4, ev_end - ev_start)

        # stale_repeat: max 8 consecutive rows within event
        stale_start = ev_start + int(rng.integers(2, max(3, (ev_end - ev_start) // 3)))
        stale_count = int(rng.integers(3, min(9, ev_end - stale_start)))
        stale_end   = min(stale_start + stale_count, ev_end)
        is_stale[stale_start:stale_end] = 1
        # stale rows: copy GPS from one row before stale block
        for r in range(stale_start, stale_end):
            lat[r] = lat[r - 1]
            lon[r] = lon[r - 1]
            alt[r] = alt[r - 1]
            speed[r] = speed[r - 1]

        # Transition: smooth interpolation back over 5 rows
        trans_end = min(ev_end + 5, n_rows)
        trans_len = trans_end - ev_end
        if trans_len > 0:
            for arr, normal_val in [
                (eph,             float(np.mean(t["eph_range"]))),
                (epv,             float(np.mean(t["epv_range"]))),
                (vel_innov,       0.2),
                (pos_horiz_innov, 0.2),
                (vel_ratio,       0.3),
                (pos_horiz_ratio, 0.3),
            ]:
                arr[ev_end:trans_end] = np.linspace(arr[ev_end - 1], normal_val, trans_len)

            # Clamp transitions
            eph[ev_end:trans_end]  = np.clip(eph[ev_end:trans_end],  0.1, 50.0)
            epv[ev_end:trans_end]  = np.clip(epv[ev_end:trans_end],  0.1, 80.0)

            # Restore categorical fields immediately on first transition row
            failsafe[ev_end]   = 0
            fix_type[ev_end:trans_end] = 3
            sats[ev_end:trans_end] = np.clip(
                rng.integers(t["sats_range"][0], t["sats_range"][1] + 1, trans_len),
                4, 20,
            )
            lat[ev_end:trans_end] = np.linspace(lat[ev_end - 1], lat[min(trans_end, n_rows - 1)], trans_len)
            lon[ev_end:trans_end] = np.linspace(lon[ev_end - 1], lon[min(trans_end, n_rows - 1)], trans_len)

        seg_start_s = round(time_s[ev_start], 3)
        seg_end_s   = round(time_s[ev_end - 1], 3)
        segments.append({"start_s": seg_start_s, "end_s": seg_end_s,
                         "label": 1, "reason": "gps_spoof_synthetic"})
        prev_end = ev_end

    # Final normal segment
    if prev_end < n_rows:
        segments.append({
            "start_s": round(time_s[prev_end], 3),
            "end_s":   round(time_s[-1], 3),
            "label": 0, "reason": "normal",
        })

    # ---- make segments contiguous -----------------------------------------
    for i in range(1, len(segments)):
        segments[i]["start_s"] = segments[i - 1]["end_s"]

    # ---- post-spoof global clamps -----------------------------------------
    lat   = np.clip(lat,   -90.0, 90.0)
    lon   = np.clip(lon,   -180.0, 180.0)
    speed = np.clip(speed,   0.0, 50.0)
    eph   = np.clip(eph,     0.1, 50.0)
    epv   = np.clip(epv,     0.1, 80.0)
    sats  = np.clip(sats,    0,   20).astype(int)
    vel_innov       = np.clip(vel_innov,       0.0, 5.0)
    pos_horiz_innov = np.clip(pos_horiz_innov, 0.0, 5.0)
    pos_vert_innov  = np.clip(pos_vert_innov,  0.0, 5.0)
    vel_ratio       = np.clip(vel_ratio,       0.0, 5.0)
    pos_horiz_ratio = np.clip(pos_horiz_ratio, 0.0, 5.0)
    pos_vert_ratio  = np.clip(pos_vert_ratio,  0.0, 5.0)
    hdg = hdg % 360.0
    hdg = np.where(hdg >= 360.0, 359.99, hdg)
    yaw = yaw % 360.0
    yaw = np.where(yaw >= 360.0, 359.99, yaw)
    roll  = np.clip(roll,  -45.0, 45.0)
    pitch = np.clip(pitch, -30.0, 30.0)

    # Cross-column consistency guard c: sats <= 4 → fix_type <= 2
    low_sat_mask = sats <= 4
    fix_type[low_sat_mask & (fix_type == 3)] = 2

    # ---- ISO timestamps ---------------------------------------------------
    last_update_iso = [
        (BASE_ISO + timedelta(seconds=float(ts))).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        for ts in time_s
    ]

    # ---- assemble DataFrame -----------------------------------------------
    df = pd.DataFrame({
        "time_s":               time_s,
        "lat_deg":              lat,
        "lon_deg":              lon,
        "alt_m":                alt,
        "rel_alt_m":            rel_alt,
        "vel_m_s":              speed,
        "hdg_deg":              hdg,
        "fix_type":             fix_type.astype(int),
        "satellites_visible":   sats.astype(int),
        "eph_m":                eph,
        "epv_m":                epv,
        "roll_deg":             roll,
        "pitch_deg":            pitch,
        "yaw_deg":              yaw,
        "rollspeed_radps":      rollspeed,
        "pitchspeed_radps":     pitchspeed,
        "yawspeed_radps":       yawspeed,
        "vibration_x":          vib_x,
        "vibration_y":          vib_y,
        "vibration_z":          vib_z,
        "clipping_0":           clipping_0.astype(int),
        "clipping_1":           clipping_1.astype(int),
        "clipping_2":           clipping_2.astype(int),
        "vel_ratio":            vel_ratio,
        "pos_horiz_ratio":      pos_horiz_ratio,
        "pos_vert_ratio":       pos_vert_ratio,
        "vel_innov":            vel_innov,
        "pos_horiz_innov":      pos_horiz_innov,
        "pos_vert_innov":       pos_vert_innov,
        "battery_voltage":      battery_voltage,
        "battery_remaining_pct": battery_pct,
        "armed":                armed.astype(int),
        "mode":                 ["AUTO"] * n_rows,
        "failsafe":             failsafe.astype(int),
        "connection_ok":        connection_ok.astype(int),
        "last_update_iso":      last_update_iso,
        "is_stale_repeat":      is_stale.astype(int),
        "dt":                   np.full(n_rows, 0.1),
    })

    return df, segments


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_df(df: pd.DataFrame, filename: str, terrain: str) -> None:
    t = TERRAINS[terrain]

    # 1. No NaN
    nan_counts = df.isna().sum()
    if nan_counts.any():
        bad = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(f"{filename}: NaN in columns: {bad}")

    # 2. No Inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col].values).any():
            raise ValueError(f"{filename}: Inf in column '{col}'")

    # 3. No negatives in magnitude columns
    mag_cols = [
        "vel_m_s", "eph_m", "epv_m", "vibration_x", "vibration_y", "vibration_z",
        "vel_ratio", "pos_horiz_ratio", "pos_vert_ratio",
        "vel_innov", "pos_horiz_innov", "pos_vert_innov",
        "battery_voltage", "battery_remaining_pct",
    ]
    for col in mag_cols:
        if (df[col] < 0).any():
            raise ValueError(f"{filename}: negative values in '{col}'")

    # 4. Integer columns
    int_cols = [
        "fix_type", "satellites_visible", "clipping_0", "clipping_1", "clipping_2",
        "armed", "failsafe", "connection_ok", "is_stale_repeat",
    ]
    for col in int_cols:
        if not np.all(df[col].values == df[col].values.astype(int)):
            raise ValueError(f"{filename}: non-integer values in '{col}'")

    # 5. time_s monotonically increasing
    if not (np.diff(df["time_s"].values) > 0).all():
        raise ValueError(f"{filename}: time_s not strictly increasing")

    # 6. battery monotone decreasing
    if not (np.diff(df["battery_remaining_pct"].values) <= 0).all():
        raise ValueError(f"{filename}: battery_remaining_pct not monotone decreasing")

    # 7. No duplicate time_s
    if df["time_s"].duplicated().any():
        raise ValueError(f"{filename}: duplicate time_s values")

    # 8. lat/lon inside terrain bbox (Relaxed for spoofing research)
    lb, lB = t["lat_bbox"]
    ob, oB = t["lon_bbox"]
    # We just log a warning instead of raising an error
    if (df["lat_deg"] < lb - 0.5).any() or (df["lat_deg"] > lB + 0.5).any():
        print(f"    [WARN] {filename}: lat_deg pushed far outside bbox")
    if (df["lon_deg"] < ob - 0.5).any() or (df["lon_deg"] > oB + 0.5).any():
        print(f"    [WARN] {filename}: lon_deg pushed far outside bbox")

    # 9. alt_m within clamp
    ac_lo, ac_hi = t["alt_clamp"]
    if (df["alt_m"] < ac_lo).any() or (df["alt_m"] > ac_hi).any():
        raise ValueError(f"{filename}: alt_m outside clamp [{ac_lo}, {ac_hi}]")

    # 10. rel_alt <= alt for flat/mountain
    if terrain in ("flat", "mountain"):
        if (df["rel_alt_m"] > df["alt_m"]).any():
            raise ValueError(f"{filename}: rel_alt_m > alt_m found")

    # 11. Column order
    if list(df.columns) != EXPECTED_COLS:
        raise ValueError(f"{filename}: column order mismatch")

    # 12. Cross-column: failsafe=1 must not have fix_type=3
    bad = df[(df["failsafe"] == 1) & (df["fix_type"] == 3)]
    if len(bad) > 0:
        raise ValueError(f"{filename}: {len(bad)} rows have failsafe=1 AND fix_type=3")

    # 13. Cross-column: sats<=4 must not have fix_type=3
    bad2 = df[(df["satellites_visible"] <= 4) & (df["fix_type"] == 3)]
    if len(bad2) > 0:
        raise ValueError(f"{filename}: {len(bad2)} rows have sats<=4 AND fix_type=3")

    # 14. Cross-column: vel > 22 must have vel_innov > 1.0
    bad3 = df[(df["vel_m_s"] > 22) & (df["vel_innov"] < 1.0)]
    if len(bad3) > 0:
        raise ValueError(f"{filename}: {len(bad3)} rows have vel>22 but vel_innov<1.0")

    # 15. mode must always be "AUTO"
    if not (df["mode"] == "AUTO").all():
        raise ValueError(f"{filename}: mode column has non-'AUTO' values")


def validate_segments(segs: list[dict], filename: str) -> None:
    assert len(segs) > 0, f"{filename}: empty segments"
    for i, s in enumerate(segs):
        assert s["end_s"] > s["start_s"], f"{filename} seg {i}: end_s <= start_s"
        assert s["label"] in {0, 1},      f"{filename} seg {i}: invalid label"
        assert s["reason"] in {"normal", "gps_spoof_synthetic"}, \
            f"{filename} seg {i}: invalid reason"
        dur = s["end_s"] - s["start_s"]
        assert dur >= 0.4, f"{filename} seg {i}: duration {dur:.3f}s < 0.5s minimum"

    for i in range(len(segs) - 1):
        gap = round(segs[i + 1]["start_s"] - segs[i]["end_s"], 3)
        assert abs(gap) <= 0.002, f"{filename}: gap/overlap between seg {i} and {i+1}: {gap}s"


# ---------------------------------------------------------------------------
# Window count helper
# ---------------------------------------------------------------------------
def count_windows(n_rows: int, labels: np.ndarray) -> tuple[int, int]:
    n0, n1 = 0, 0
    i = 0
    while i + WINDOW_LEN <= n_rows:
        win = labels[i: i + WINDOW_LEN]
        w_lbl = 1 if win.mean() >= 0.5 else 0
        if w_lbl == 0:
            n0 += 1
        else:
            n1 += 1
        i += STRIDE
    return n0, n1


def row_labels_from_segments(segs: list[dict], time_s: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(time_s), dtype=int)
    for s in segs:
        mask = (time_s >= s["start_s"]) & (time_s <= s["end_s"])
        labels[mask] = s["label"]
    return labels


# ---------------------------------------------------------------------------
# Post-generation file integrity check
# ---------------------------------------------------------------------------
def post_generation_check(output_dir: str) -> None:
    issues: list[str] = []

    csv_files  = sorted(f for f in os.listdir(output_dir) if f.endswith("_cleaned.csv"))
    json_files = sorted(f for f in os.listdir(output_dir) if f.endswith("_auto_segments.json"))

    for csv in csv_files:
        stem = csv.replace("_cleaned.csv", "")
        if f"{stem}_auto_segments.json" not in json_files:
            issues.append(f"MISSING JSON for {csv}")

    for csv in csv_files:
        df = pd.read_csv(os.path.join(output_dir, csv))
        if len(df.columns) != len(EXPECTED_COLS):
            issues.append(f"{csv}: expected {len(EXPECTED_COLS)} cols, got {len(df.columns)}")
        if len(df) < 100:
            issues.append(f"{csv}: too few rows ({len(df)})")
        if list(df.columns) != EXPECTED_COLS:
            issues.append(f"{csv}: column order/names mismatch")

    for jf in json_files:
        with open(os.path.join(output_dir, jf)) as f:
            segs = json.load(f)
        for i in range(len(segs) - 1):
            gap = round(segs[i + 1]["start_s"] - segs[i]["end_s"], 3)
            if abs(gap) > 0.002:
                issues.append(f"{jf}: gap/overlap seg {i}↔{i+1}: {gap}s")

    if issues:
        print("\n❌ POST-GENERATION ISSUES FOUND:")
        for iss in issues:
            print(f"   - {iss}")
        raise RuntimeError(f"{len(issues)} data integrity issue(s) found. Fix before proceeding.")
    else:
        print("\n✅ All files passed post-generation integrity check.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic GPS telemetry CSVs.")
    parser.add_argument("--output-dir",        default="Step_5_Data/synthetic")
    parser.add_argument("--flights-per-terrain", type=int, default=3)
    parser.add_argument("--rows-per-flight",    type=int, default=1500)
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--terrain",            choices=["flat", "mountain", "sea"], default=None,
                        help="Generate only one terrain (omit for all three)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    terrains = [args.terrain] if args.terrain else ["flat", "mountain", "sea"]
    summary: list[dict] = []

    print("Generating synthetic GPS telemetry...\n")

    for t_idx, terrain in enumerate(terrains):
        total_windows = 0
        total_rows    = 0

        for f_idx in range(1, args.flights_per_terrain + 1):
            seed     = args.seed + t_idx * 100 + f_idx
            csv_name = f"{terrain}_flight_{f_idx:02d}_cleaned.csv"
            jsn_name = f"{terrain}_flight_{f_idx:02d}_auto_segments.json"
            csv_path = os.path.join(args.output_dir, csv_name)
            jsn_path = os.path.join(args.output_dir, jsn_name)

            df, segs = generate_flight(terrain, f_idx, args.rows_per_flight, seed)

            # Inject label BEFORE validation (required by EXPECTED_COLS check)
            row_lbls = row_labels_from_segments(segs, df["time_s"].values)
            df["label"] = row_lbls

            # Per-file validate
            validate_df(df, csv_name, terrain)
            validate_segments(segs, jsn_name)

            # Window label counts
            n0, n1   = count_windows(len(df), row_lbls)
            n_win    = n0 + n1
            total_windows += n_win
            total_rows    += len(df)

            # Save
            df.to_csv(csv_path, index=False)
            with open(jsn_path, "w") as f:
                json.dump(segs, f, indent=2)

            print(f"    - {csv_name}: {len(df)} rows, {n_win} windows ({n0} normal, {n1} anomaly)")

        summary.append({
            "terrain":       terrain,
            "flights":       args.flights_per_terrain,
            "total_rows":    total_rows,
            "total_windows": total_windows,
        })

    print("\n" + "-" * 50)
    for s in summary:
        print(f"Terrain {s['terrain']:<10}: {s['flights']} flights, {s['total_windows']} windows total")

    post_generation_check(args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
