import json
from pathlib import Path
import pandas as pd
import numpy as np

CLEAN_PATH = Path("gps_logs/processed/gps_log_20260402_184048_cleaned.csv")
SEGMENTS_PATH = Path("gps_logs/processed/segments.json")
LABELS_PATH = Path("gps_logs/processed/row_labels.csv")

def load_cleaned():
    df = pd.read_csv(CLEAN_PATH)
    df["time_s"] = pd.to_numeric(df["time_s"])
    return df.sort_values("time_s").reset_index(drop=True)

def create_segments(df):
    """
    Manual labeling based on inspection:
    - Startup: UNKNOWN, no motion
    - Normal LOITER/Takeoff/Land: stable flight
    - Anomaly: MODE_HEX with failsafe=1, stale repeats, clear disturbance
    """
    segments = []
    
    # Startup (ignore)
    startup_end = 2.0
    segments.append({"start_s": 0.0, "end_s": startup_end, "label": "ignore", "reason": "startup noise"})
    
    # Normal LOITER phases (label 0)
    loiter_ranges = [
        (2.0, 9.0),
        (17.0, 166.0),  # main stable flight
        (170.0, 269.0)  # end stable
    ]
    for start, end in loiter_ranges:
        mask = (df["time_s"] >= start) & (df["time_s"] < end)
        if mask.sum() > 10:
            segments.append({"start_s": start, "end_s": end, "label": 0, "reason": "normal LOITER"})
    
    # Anomaly: MODE_HEX with failsafe=1 (label 1)
    anomaly_ranges = [
        (9.0, 17.0),  # TAKEOFF with possible issues
        (166.0, 170.0),  # stale repeat
        # Add any other disturbance you visually spotted
    ]
    for start, end in anomaly_ranges:
        mask = (df["time_s"] >= start) & (df["time_s"] < end)
        if mask.sum() > 10:
            segments.append({"start_s": start, "end_s": end, "label": 1, "reason": "MODE_HEX/failsafe/stale"})
    
    return segments

def assign_row_labels(df, segments):
    df["label"] = 0  # default normal
    
    for seg in segments:
        if seg["label"] == "ignore":
            continue
        mask = (df["time_s"] >= seg["start_s"]) & (df["time_s"] < seg["end_s"])
        df.loc[mask, "label"] = seg["label"]
    
    return df

def main():
    df = load_cleaned()
    
    segments = create_segments(df)
    print("Segments:")
    for s in segments:
        print(f"  {s['start_s']:.1f}-{s['end_s']:.1f}s: {s['label']} ({s['reason']})")
    
    df_labeled = assign_row_labels(df, segments)
    
    print(f"\nLabel distribution:")
    print(df_labeled["label"].value_counts().to_dict())
    
    # Save segments
    SEGMENTS_PATH.write_text(json.dumps(segments, indent=2))
    
    # Save labeled rows
    df_labeled[["time_s", "label", "mode", "armed", "failsafe", "is_stale_repeat"]].to_csv(LABELS_PATH, index=False)
    
    print(f"\nSaved: {SEGMENTS_PATH}")
    print(f"Saved: {LABELS_PATH}")
    
    print("\nNext: run 03_make_windows.py")

if __name__ == "__main__":
    main()
