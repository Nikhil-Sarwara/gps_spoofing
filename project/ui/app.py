from __future__ import annotations
import os
import sys
import time
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_utils import latest_log_file, load_log

DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
THRESHOLD = 0.5
WINDOW_SIZE = 200


def render_status_banner(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    proba = float(latest["anom_proba"])
    is_anomaly = proba > THRESHOLD
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Status", "🔴 ANOMALY" if is_anomaly else "🟢 Normal")
    c2.metric("p(anomaly)", f"{proba:.3f}")
    c3.metric("Altitude (m)", f"{float(latest.get('alt_m', 0)):.1f}")
    c4.metric("Speed (m/s)", f"{float(latest.get('vel_m_s', 0)):.2f}")
    c5.metric("Total Samples", str(len(df)))


def render_proba_chart(df: pd.DataFrame, window: int) -> None:
    tail = df.tail(window).copy()
    if "timestamp" in tail.columns and tail["timestamp"].notna().any():
        tail = tail.set_index("timestamp")
    st.line_chart(pd.DataFrame({"p(anomaly)": tail["anom_proba"]}), height=280)


def render_gps_row(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latitude", f"{float(latest.get('lat_deg', 0)):.6f}°")
    c2.metric("Longitude", f"{float(latest.get('lon_deg', 0)):.6f}°")
    c3.metric("Altitude (m)", f"{float(latest.get('alt_m', 0)):.2f}")
    c4.metric("Rel Alt (m)", f"{float(latest.get('rel_alt_m', 0)):.2f}")


def page_live() -> None:
    st.title("GPS Anomaly — Live Monitor")

    log_dir = st.sidebar.text_input("Log directory", DEFAULT_LOG_DIR)
    refresh = st.sidebar.slider("Refresh interval (s)", 0.5, 5.0, 1.0)
    window = st.sidebar.slider("Chart window (rows)", 50, 500, WINDOW_SIZE)

    if not os.path.isdir(log_dir):
        st.warning(f"Log directory not found: {log_dir}")
        st.info("Run  →  python ml/live_inference.py  to generate logs.")
        return

    slot_banner = st.empty()
    slot_chart = st.empty()
    slot_gps = st.empty()
    slot_table = st.empty()

    while True:
        path = latest_log_file(log_dir)
        if not path:
            slot_banner.info("Waiting for live_inference.py to produce a log file…")
            time.sleep(refresh)
            continue

        df = load_log(path)
        if df.empty or "anom_proba" not in df.columns:
            slot_banner.info(f"Log exists but has no data yet: {path}")
            time.sleep(refresh)
            continue

        with slot_banner.container():
            render_status_banner(df)

        with slot_chart.container():
            st.subheader("Anomaly Probability Over Time")
            render_proba_chart(df, window)

        with slot_gps.container():
            st.subheader("GPS Telemetry")
            render_gps_row(df)

        with slot_table.container():
            st.subheader("Last 20 Rows")
            st.dataframe(df.tail(20), use_container_width=True)

        time.sleep(refresh)


def page_replay() -> None:
    st.title("GPS Anomaly — Log Replay")

    uploaded = st.file_uploader("Upload a live_*.csv log file", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV produced by live_inference.py to replay it.")
        return

    df = pd.read_csv(uploaded)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["unix_time", "anom_proba", "alt_m", "rel_alt_m", "vel_m_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "anom_proba" not in df.columns:
        st.error("File does not contain an 'anom_proba' column.")
        return

    render_status_banner(df)
    st.divider()

    if "timestamp" in df.columns and df["timestamp"].notna().any():
        idx_df = df.set_index("timestamp")
    else:
        idx_df = df.copy()

    st.subheader("Anomaly Probability")
    st.line_chart(pd.DataFrame({"p(anomaly)": idx_df["anom_proba"]}), height=280)

    col1, col2 = st.columns(2)
    with col1:
        if "alt_m" in idx_df.columns:
            st.subheader("Altitude (m)")
            st.line_chart(pd.DataFrame({"alt_m": idx_df["alt_m"]}), height=220)
    with col2:
        if "vel_m_s" in idx_df.columns:
            st.subheader("Speed (m/s)")
            st.line_chart(pd.DataFrame({"vel_m_s": idx_df["vel_m_s"]}), height=220)

    anomalies = df[df["anom_proba"] > THRESHOLD]
    st.subheader(f"Anomaly Events ({len(anomalies)} detected)")
    if anomalies.empty:
        st.success("No anomalies above threshold in this log.")
    else:
        st.dataframe(anomalies, use_container_width=True)

    st.subheader("Full Log")
    st.dataframe(df, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="GPS Anomaly Monitor",
        page_icon="🛸",
        layout="wide",
    )
    st.sidebar.title("GPS Anomaly Monitor")
    page = st.sidebar.radio("View", ["Live Monitor", "Log Replay"])
    if page == "Live Monitor":
        page_live()
    else:
        page_replay()


if __name__ == "__main__":
    main()
