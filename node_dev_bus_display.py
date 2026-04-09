from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import os

DISPLAY_LIMIT = 10   # Max collection entries shown per value

mb    = TelemetryBroker()
_perf = PerfMonitor("node_dev_bus_display", broker=mb)

CURSOR_UP_LEFT = "\033[H"  # Jump to top-left (home)
HIDE_CURSOR    = "\033[?25l"
SHOW_CURSOR    = "\033[?25h"


def _truncate(value):
    """If value is a JSON collection with more than DISPLAY_LIMIT entries, truncate it."""
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value

    if isinstance(parsed, dict) and len(parsed) > DISPLAY_LIMIT:
        truncated = dict(list(parsed.items())[:DISPLAY_LIMIT])
        return json.dumps(truncated) + f" … (+{len(parsed) - DISPLAY_LIMIT} more)"
    if isinstance(parsed, list) and len(parsed) > DISPLAY_LIMIT:
        return json.dumps(parsed[:DISPLAY_LIMIT]) + f" … (+{len(parsed) - DISPLAY_LIMIT} more)"
    return value


import argparse, sys
_ap = argparse.ArgumentParser()
_ap.add_argument("--no-output", action="store_true")
if _ap.parse_args().no_output:
    import os; sys.stdout = open(os.devnull, "w")

print(HIDE_CURSOR, end="")

while True:
    try:
        with _perf.measure("loop"):
            data = mb.getall()
            output = ""
            os.system('cls' if os.name == 'nt' else 'clear')
            for key, value in sorted(data.items()):
                output += f"{key} : {_truncate(value)}\n"
            print(f"{CURSOR_UP_LEFT}{output}", end="\r", flush=True)

    except KeyboardInterrupt:
        print(SHOW_CURSOR)
        break

mb.close()