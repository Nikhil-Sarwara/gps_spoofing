#!/usr/bin/env python3
"""
Automated GPS Spoofing/Anomaly Detection for Labeling

This script uses heuristic rules to automatically detect and label
anomalies in GPS telemetry data, eliminating manual labeling effort.

Detection Methods:
1. Position jumps - sudden unrealistic movement
2. Speed anomalies - velocity inconsistencies
3. GPS quality degradation - satellite count, accuracy metrics
4. Stale data - repeated identical readings
5. Mode anomalies - failsafe, mode changes
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd


class AutoLabeler:
    def __init__(self, config: Optional[dict] = None):
        defaults = {
            "enabled": True,
            "max_normal_speed_ms": 30.0,
            "max_normal_accel_ms2": 15.0,
            "min_satellites": 6,
            "max_eph_m": 5.0,
            "max_epv_m": 10.0,
            "stale_threshold": 2,
            "max_stale_ratio": 0.1,
            "failsafe_is_anomaly": True,
            "mode_hex_is_anomaly": True,
            "sudden_speed_change_ms": 10.0,
        }
        if config:
            defaults.update(config)
        self.config = defaults
        self.df: Optional[pd.DataFrame] = None
        self.anomaly_reasons: list = []

    def _deep_merge(self, base: dict, update: dict) -> dict:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def load_data(self, csv_path: str | Path) -> pd.DataFrame:
        self.df = pd.read_csv(csv_path)
        self.df = self.df.sort_values("time_s").reset_index(drop=True)
        self._ensure_columns()
        return self.df

    def _ensure_columns(self):
        required = ["time_s", "lat_deg", "lon_deg", "alt_m", "vel_m_s"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        if "satellites_visible" not in self.df.columns:
            self.df["satellites_visible"] = 10
        if "eph_m" not in self.df.columns:
            self.df["eph_m"] = 1.0
        if "epv_m" not in self.df.columns:
            self.df["epv_m"] = 2.0
        if "failsafe" not in self.df.columns:
            self.df["failsafe"] = 0
        if "mode" not in self.df.columns:
            self.df["mode"] = "UNKNOWN"

    def detect_anomalies(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        cfg = self.config["auto_label"]
        self.df["label"] = 0
        self.df["anomaly_reason"] = ""

        self._detect_position_jumps(cfg)
        self._detect_speed_anomalies(cfg)
        self._detect_gps_quality_degradation(cfg)
        self._detect_stale_data(cfg)
        self._detect_mode_anomalies(cfg)

        return self.df

    def _detect_position_jumps(self, cfg: dict):
        """Detect sudden unrealistic position changes."""
        for i in range(1, len(self.df)):
            dt = self.df.loc[i, "time_s"] - self.df.loc[i - 1, "time_s"]
            if dt <= 0:
                continue

            lat1, lon1 = self.df.loc[i - 1, "lat_deg"], self.df.loc[i - 1, "lon_deg"]
            lat2, lon2 = self.df.loc[i, "lat_deg"], self.df.loc[i, "lon_deg"]

            dist = self._haversine_distance(lat1, lon1, lat2, lon2)
            speed = dist / dt

            if speed > cfg["max_normal_speed_ms"]:
                self.df.loc[i, "label"] = 1
                self.df.loc[i, "anomaly_reason"] += f"pos_jump:{speed:.1f}m/s;"

    def _detect_speed_anomalies(self, cfg: dict):
        """Detect velocity inconsistencies."""
        if "vel_m_s" not in self.df.columns:
            return

        for i in range(1, len(self.df)):
            dt = self.df.loc[i, "time_s"] - self.df.loc[i - 1, "time_s"]
            if dt <= 0:
                continue

            vel_change = abs(self.df.loc[i, "vel_m_s"] - self.df.loc[i - 1, "vel_m_s"])

            if vel_change > cfg["sudden_speed_change_ms"]:
                self.df.loc[i, "label"] = 1
                self.df.loc[i, "anomaly_reason"] += f"speed_change:{vel_change:.1f}m/s;"

        if "accel" not in self.df.columns:
            self.df["accel"] = self.df["vel_m_s"].diff() / self.df["time_s"].diff()
            accel_anomaly = self.df["accel"].abs() > cfg["max_normal_accel_ms2"]
            self.df.loc[accel_anomaly.fillna(False), "label"] = 1
            self.df.loc[accel_anomaly.fillna(False), "anomaly_reason"] += "accel_anomaly;"

    def _detect_gps_quality_degradation(self, cfg: dict):
        """Detect GPS quality issues."""
        sats_low = self.df["satellites_visible"] < cfg["min_satellites"]
        eph_high = self.df["eph_m"] > cfg["max_eph_m"]
        epv_high = self.df["epv_m"] > cfg["max_epv_m"]

        quality_mask = sats_low | eph_high | epv_high
        self.df.loc[quality_mask, "label"] = 1
        self.df.loc[sats_low, "anomaly_reason"] += f"low_sats;"
        self.df.loc[eph_high, "anomaly_reason"] += "high_eph;"
        self.df.loc[epv_high, "anomaly_reason"] += "high_epv;"

    def _detect_stale_data(self, cfg: dict):
        """Detect repeated identical readings."""
        cols_to_check = ["lat_deg", "lon_deg", "alt_m", "vel_m_s"]
        stale_mask = pd.Series(True, index=self.df.index)

        for col in cols_to_check:
            if col in self.df.columns:
                stale_mask &= self.df[col] == self.df[col].shift(1)

        stale_mask = stale_mask.fillna(False)
        stale_mask[:cfg["stale_threshold"]] = False

        self.df["is_stale"] = stale_mask.astype(int)
        stale_runs = stale_mask.groupby((stale_mask != stale_mask.shift()).cumsum()).transform("count")
        long_stale = stale_runs >= cfg["stale_threshold"]

        self.df.loc[long_stale, "label"] = 1
        self.df.loc[long_stale, "anomaly_reason"] += "stale_data;"

    def _detect_mode_anomalies(self, cfg: dict):
        """Detect failsafe and mode anomalies."""
        if cfg["failsafe_is_anomaly"] and "failsafe" in self.df.columns:
            failsafe_mask = self.df["failsafe"] == 1
            self.df.loc[failsafe_mask, "label"] = 1
            self.df.loc[failsafe_mask, "anomaly_reason"] += "failsafe;"

        if cfg["mode_hex_is_anomaly"] and "mode" in self.df.columns:
            mode_hex_mask = self.df["mode"].astype(str).str.startswith("Mode(0x")
            self.df.loc[mode_hex_mask, "label"] = 1
            self.df.loc[mode_hex_mask, "anomaly_reason"] += "mode_hex;"

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in meters between two GPS coordinates."""
        R = 6371000
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    def get_segment_summary(self) -> list[dict]:
        """Generate segment summary for windows.py."""
        segments = []
        current_label = None
        start_time = None

        for _, row in self.df.iterrows():
            if row["label"] != current_label:
                if start_time is not None:
                    segments.append({
                        "start_s": float(start_time),
                        "end_s": float(row["time_s"]),
                        "label": int(current_label) if current_label is not None else 0,
                        "reason": self._get_reason_at_time(start_time),
                    })
                current_label = row["label"]
                start_time = row["time_s"]

        if start_time is not None:
            segments.append({
                "start_s": float(start_time),
                "end_s": float(self.df["time_s"].iloc[-1]),
                "label": int(current_label),
                "reason": self._get_reason_at_time(start_time),
            })

        return segments

    def _get_reason_at_time(self, time_s: float) -> str:
        row = self.df[self.df["time_s"] == time_s]
        if len(row) > 0:
            return row.iloc[0]["anomaly_reason"] or "normal"
        return "unknown"

    def save_labels(self, output_path: str | Path, save_segments: bool = True):
        """Save labeled data and optional segment summary."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cols_to_save = ["time_s", "label", "anomaly_reason"]
        for col in cols_to_save:
            if col in self.df.columns:
                continue

        self.df[cols_to_save].to_csv(output_path, index=False)
        print(f"Saved labels: {output_path}")

        if save_segments:
            segments_path = output_path.parent / "auto_segments.json"
            segments = self.get_segment_summary()
            with open(segments_path, "w") as f:
                json.dump(segments, f, indent=2)
            print(f"Saved segments: {segments_path}")

        print(f"\nLabel distribution:\n{self.df['label'].value_counts()}")


def main(
    input_csv: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[dict] = None,
    save_segments: bool = True,
) -> Path:
    """
    Auto-label GPS data for spoofing detection.

    Args:
        input_csv: Path to cleaned CSV file
        output_path: Output CSV path (default: row_labels_auto.csv in same dir)
        config: Optional configuration dict
        save_segments: Whether to save segment summary JSON

    Returns:
        Path to output CSV
    """
    labeler = AutoLabeler(config=config)

    print(f"Loading: {input_csv}")
    labeler.load_data(input_csv)
    print(f"Loaded {len(labeler.df)} rows")

    print("Detecting anomalies...")
    labeler.detect_anomalies()

    output_path = output_path or str(Path(input_csv).parent / "row_labels_auto.csv")
    labeler.save_labels(output_path, save_segments=save_segments)

    anomaly_count = (labeler.df["label"] == 1).sum()
    print(f"\nTotal anomalies detected: {anomaly_count} ({anomaly_count / len(labeler.df) * 100:.1f}%)")

    return Path(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Auto-label GPS data for spoofing detection")
    parser.add_argument("input_csv", help="Path to cleaned CSV file")
    parser.add_argument("-o", "--output", help="Output CSV path (default: row_labels_auto.csv)")
    parser.add_argument("--no-save-segments", action="store_true", help="Don't save segment summary")
    args = parser.parse_args()
    main(args.input_csv, args.output, save_segments=not args.no_save_segments)
