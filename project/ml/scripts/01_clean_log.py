from pathlib import Path
import json
import pandas as pd
import numpy as np

RAW_PATH = Path("gps_logs/raw/gps_log_20260402_184048.csv")
OUT_DIR = Path("gps_logs/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_PATH = OUT_DIR / "gps_log_20260402_184048_cleaned.csv"
REPORT_PATH = OUT_DIR / "gps_log_20260402_184048_cleaning_report.json"

NUMERIC_COLS = [
    "time_s", "lat_deg", "lon_deg", "alt_m", "rel_alt_m", "vel_m_s", "hdg_deg",
    "fix_type", "satellites_visible", "eph_m", "epv_m",
    "ax_mps2", "ay_mps2", "az_mps2",
    "gx_radps", "gy_radps", "gz_radps",
    "battery_voltage", "battery_remaining_pct",
    "armed", "failsafe", "connection_ok"
]

TEXT_COLS = ["mode", "last_update_iso"]

def normalize_mode(x):
    if pd.isna(x):
        return "UNKNOWN"
    x = str(x).strip()
    if x.startswith("Mode(0x"):
        return "MODE_HEX"
    return x.upper()

def main():
    df = pd.read_csv(RAW_PATH)

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
        df = df[df["fix_type"] >= 3]

    if "connection_ok" in df.columns:
        df = df[df["connection_ok"] == 1]

    df = df.drop_duplicates().reset_index(drop=True)

    stale_cols = [c for c in ["lat_deg", "lon_deg", "alt_m", "rel_alt_m", "vel_m_s", "hdg_deg", "last_update_iso"] if c in df.columns]
    stale_mask = df[stale_cols].eq(df[stale_cols].shift(1)).all(axis=1)
    df["is_stale_repeat"] = stale_mask.fillna(False)

    stale_run = df["is_stale_repeat"].groupby((df["is_stale_repeat"] != df["is_stale_repeat"].shift()).cumsum()).transform("sum")
    df = df[~((df["is_stale_repeat"]) & (stale_run >= 5))].copy()

    df["dt"] = df["time_s"].diff()
    df = df[(df["dt"].isna()) | ((df["dt"] > 0) & (df["dt"] < 5))].copy()

    df = df.reset_index(drop=True)

    report = {
        "input_file": str(RAW_PATH),
        "output_file": str(CLEAN_PATH),
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

    df.to_csv(CLEAN_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Saved cleaned CSV: {CLEAN_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Rows: {initial_rows} -> {len(df)}")

if __name__ == "__main__":
    main()
