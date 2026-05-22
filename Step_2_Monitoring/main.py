# gps_monitor/main.py

from __future__ import annotations

import sys
import time
import argparse

from .mavlink_client import MAVLinkClient
from .state_model import TelemetryState
from .console_ui import ConsoleUI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal", action="store_true", help="Output one-line telemetry summaries")
    args = parser.parse_args()

    state = TelemetryState()
    client = MAVLinkClient(state)

    try:
        client.start()
        if args.minimal:
            print("[INFO] Minimal telemetry output started.")
            while True:
                s = client.snapshot()
                gps = s.gps
                health = s.health
                if gps.lat_deg is not None:
                    # Specific tag for the TUI to parse
                    print(f"TELEMETRY: {health.mode} | Lat: {gps.lat_deg:.7f}, Lon: {gps.lon_deg:.7f}, Alt: {gps.alt_m:.1f}m, Vel: {gps.vel_m_s:.1f}m/s")
                time.sleep(1.0)
        else:
            ui = ConsoleUI(refresh_hz=4.0)
            ui.run(
                state_provider=client.snapshot,
                stop_checker=lambda: False,
            )
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        client.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
