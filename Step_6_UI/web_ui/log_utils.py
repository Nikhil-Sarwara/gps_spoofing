from __future__ import annotations
import glob
import os
from typing import Optional
import pandas as pd


def list_log_files(log_dir: str) -> list[str]:
    pattern = os.path.join(log_dir, "live_*.csv")
    return sorted(glob.glob(pattern))


def latest_log_file(log_dir: str) -> Optional[str]:
    files = list_log_files(log_dir)
    return files[-1] if files else None


def load_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()
    if df.empty:
        return df
    for col in ["unix_time", "anom_proba"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df
