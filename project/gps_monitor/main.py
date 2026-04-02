# gps_monitor/main.py

from __future__ import annotations

import sys

from .console_ui import ConsoleUI
from .mavlink_client import MAVLinkClient
from .state_model import TelemetryState


def main() -> int:
    state = TelemetryState()
    client = MAVLinkClient(state)
    ui = ConsoleUI(refresh_hz=4.0)

    try:
        client.start()
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
