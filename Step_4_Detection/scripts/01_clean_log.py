from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Optional

NUMERIC_COLS = [
    "time_s", "lat_deg", "lon_deg", "alt_m", "rel_alt_m", "vel_m_s", "hdg_deg",
    "fix_type", "satellites_visible", "eph_m", "epv_m",
    "roll_deg", "pitch_deg", "yaw_deg",
    "rollspeed_radps", "pitchspeed_radps", "yawspeed_radps",
    "vibration_x", "vibration_y", "vibration_z",
    "clipping_0", "clipping_1", "clipping_2",
    "vel_ratio", "pos_horiz_ratio", "pos_vert_ratio",
    "vel_innov", "pos_horiz_innov", "pos_vert_innov",
    "battery_voltage", "battery_remaining_pct",
    "armed", "failsafe", "connection_ok"
]

TEXT_COLS = ["mode", "last_update_iso"]


def normalize_mode(x):
    if pd.isna(x):
        return "UNKNOWN"
    x = str(x).strip()
    if x.startswith("Mode(0x"):
        hex_val = x.replace("Mode(0x", "").replace(")", "").strip()
        if hex_val.lower() in ("c0", "0xc0"):
            return "OFFBOARD"
        return f"HEX_{hex_val.upper()}"
    return x.upper()


def main(
    raw_path: str | Path,
    output_dir: str | Path,
    cleaning_config: Optional[dict] = None,
) -> Path:
    """
    Clean raw GPS telemetry data.

    Args:
        raw_path: Path to raw CSV file
        output_dir: Directory for cleaned output
        cleaning_config: Optional cleaning configuration

    Returns:
        Path to cleaned CSV file
    """
    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cleaning_config is None:
        cleaning_config = {}

    min_fix_type = cleaning_config.get("min_fix_type", 3)
    max_dt_gap = cleaning_config.get("max_dt_gap_s", 5.0)

    clean_name = raw_path.stem + "_cleaned.csv"
    clean_path = output_dir / clean_name
    report_path = output_dir / (raw_path.stem + "_cleaning_report.json")

    df = pd.read_csv(raw_path)

    df.columns = [c.strip() for c in df.columns]

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    initial_rows = len(df)

    df = df.dropna(subset=["time_s"])
    df = df.sort_values("time_s").reset_index(drop=True)

    if "mode" in df.columns:
        df["mode"] = df["mode"].apply(normalize_mode)

    if "last_update_iso" in df.columns:
        df["last_update_iso"] = df["last_update_iso"].fillna("UNKNOWN")

    critical_cols = [c for c in ["lat_deg", "lon_deg", "alt_m", "rel_alt_m", "vel_m_s"] if c in df.columns]
    df = df.dropna(subset=critical_cols)

    if "fix_type" in df.columns:
        df = df[df["fix_type"] >= min_fix_type]

    if "connection_ok" in df.columns:
        df = df[df["connection_ok"] == 1]

    df = df.drop_duplicates().reset_index(drop=True)

    df["is_stale_repeat"] = 0

    df["dt"] = df["time_s"].diff()
    df = df[(df["dt"].isna()) | ((df["dt"] > 0) & (df["dt"] < max_dt_gap))].copy()

    df = df.reset_index(drop=True)

    report = {
        "input_file": str(raw_path),
        "output_file": str(clean_path),
        "initial_rows": int(initial_rows),
        "final_rows": int(len(df)),
        "rows_removed": int(initial_rows - len(df)),
        "time_start_s": float(df["time_s"].min()) if len(df) else None,
        "time_end_s": float(df["time_s"].max()) if len(df) else None,
        "duration_s": float(df["time_s"].max() - df["time_s"].min()) if len(df) else None,
        "missing_after_cleaning": df.isna().sum().to_dict(),
        "mode_counts": df["mode"].value_counts(dropna=False).to_dict() if "mode" in df.columns else {},
        "failsafe_counts": df["failsafe"].value_counts(dropna=False).to_dict() if "failsafe" in df.columns else {},
        "armed_counts": df["armed"].value_counts(dropna=False).to_dict() if "armed" in df.columns else {}
    }

    df.to_csv(clean_path, index=False)
    report_path.write_text(json.dumps(report, indent=2))

    print(f"Saved cleaned CSV: {clean_path}")
    print(f"Saved report: {report_path}")
    print(f"Rows: {initial_rows} -> {len(df)}")

    return clean_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", help="Path to raw CSV")
    parser.add_argument("--output-dir", default="Step_5_Data/processed", help="Output directory")
    args = parser.parse_args()
    main(args.raw_path, args.output_dir)