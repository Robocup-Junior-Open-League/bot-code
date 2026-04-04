from robus_core.libs.lib_telemtrybroker import TelemetryBroker
import json
import os
import time
import threading

# ── Configuration ─────────────────────────────────────────────────────────────
REFRESH_HZ = 4   # display refresh rate

PERF_NODES = [
    # ── Production (combined) nodes ───────────────────────────────────────────
    "node_prod_sensor",
    "node_prod_positioning",
    "node_prod_prediction",
    "node_prod_vision",
    "node_prod_communication",
    "node_prod_master",
    # ── Individual (dev) nodes ────────────────────────────────────────────────
    "node_dev_imu",
    "node_dev_lidar",
    "node_dev_pos_walls",
    "node_dev_pos",
    "node_dev_pos_robots",
    "node_dev_predict_robots",
    "node_dev_predict_ball",
    "node_dev_vision",
    "node_dev_twin_vis",
    "node_dev_time",
    "node_dev_bus_display",
]
# ─────────────────────────────────────────────────────────────────────────────

PERF_KEYS = [f"perf_{n}" for n in PERF_NODES]

mb     = TelemetryBroker()
_state = {}   # node_name → {key: {avg_ms, peak_ms, n}}
_lock  = threading.Lock()

# ── ANSI helpers ──────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"

HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
HOME        = "\033[H"

_COL_NODE = 30
_COL_KEY  = 18
_COL_VAL  =  9   # width of the bare number field (e.g. "  123.4ms")
_SEP_W    = _COL_NODE + _COL_KEY + (_COL_VAL + 2) * 2 + 10


def _ms_str(ms: float) -> str:
    """Fixed-width coloured millisecond string."""
    raw = f"{ms:7.1f}ms"
    if ms > 50:
        colour = _RED
    elif ms > 10:
        colour = _YELLOW
    else:
        colour = _GREEN
    return f"{colour}{raw}{_RESET}"


def _render() -> str:
    sep  = _DIM + "─" * _SEP_W + _RESET + "\n"
    title = f"{_BOLD}{_CYAN}Performance Monitor{_RESET}\n"
    header = (
        f"{_BOLD}"
        f"{'Node':<{_COL_NODE}}"
        f"{'Key':<{_COL_KEY}}"
        f"{'Avg':>{_COL_VAL + 2}}"
        f"{'Peak':>{_COL_VAL + 4}}"
        f"{'Samples':>10}"
        f"{_RESET}\n"
    )

    rows = ""
    with _lock:
        for node in PERF_NODES:
            keys = _state.get(node)
            if not keys:
                rows += f"{_DIM}{node:<{_COL_NODE}}{'no data                                                  ':<{_COL_KEY}}{_RESET}\n"
                continue
            for i, (key, stats) in enumerate(sorted(keys.items())):
                node_label = node if i == 0 else ""
                avg_ms     = stats.get("avg_ms",  0.0)
                peak_ms    = stats.get("peak_ms", 0.0)
                n          = stats.get("n",       0)
                rows += (
                    f"{node_label:<{_COL_NODE}}"
                    f"{key:<{_COL_KEY}}"
                    f"  {_ms_str(avg_ms)}"
                    f"  {_ms_str(peak_ms)}"
                    f"{n:>10}\n"
                )

    now = time.strftime("%H:%M:%S")
    footer = f"{_DIM}updated {now}  —  Ctrl+C to exit{_RESET}\n"
    return f"{HOME}{title}{sep}{header}{sep}{rows}{sep}{footer}"


def on_update(key: str, value) -> None:
    if value is None:
        return
    node = key[len("perf_"):]
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return
    with _lock:
        _state[node] = data


if __name__ == "__main__":
    for key in PERF_KEYS:
        try:
            val = mb.get(key)
            if val:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(PERF_KEYS, on_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    os.system("cls" if os.name == "nt" else "clear")
    print(HIDE_CURSOR, end="", flush=True)
    try:
        while True:
            print(_render(), end="", flush=True)
            time.sleep(1.0 / REFRESH_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        print(SHOW_CURSOR, end="")
        mb.close()
