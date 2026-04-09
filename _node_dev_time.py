from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import threading
import time

# ── Configuration ─────────────────────────────────────────────────────────────
TIME_PUBLISH_HZ    = 10   # how often system_time is updated on the broker
POSITION_WINDOW_S  = 3.0  # seconds of position history to retain
POSITION_SAMPLE_S  = 0.5  # minimum interval between saved position samples
BALL_SAMPLE_S      = 0.1  # minimum interval between saved ball position samples
# ─────────────────────────────────────────────────────────────────────────────

mb              = TelemetryBroker()
_perf           = PerfMonitor("node_dev_time", broker=mb)
_start          = time.monotonic()
_pos_lock       = threading.Lock()
_pos_history    = []    # [{"x": float, "y": float, "t": float}, ...]
_pos_last_t     = -999  # monotonic time of the last saved position sample
_robots_lock    = threading.Lock()
_robots_history = []    # [{"t": float, "robots": [{"x": float, "y": float}]}, ...]
_robots_last_t  = -999  # monotonic time of the last saved robots sample
_ball_lock      = threading.Lock()
_ball_history   = []    # [{"x": float, "y": float, "t": float}, ...]
_ball_last_t    = -999  # monotonic time of the last saved ball position sample


def _elapsed():
    return time.monotonic() - _start


def _prune_list(lst, window=POSITION_WINDOW_S):
    """Remove entries older than window seconds. Call with the relevant lock held."""
    cutoff = _elapsed() - window
    while lst and lst[0]["t"] < cutoff:
        lst.pop(0)


def _time_loop():
    interval = 1.0 / TIME_PUBLISH_HZ
    while True:
        mb.set("system_time", str(round(_elapsed(), 3)))
        time.sleep(interval)


def on_update(key, value):
    global _pos_last_t, _robots_last_t, _ball_last_t

    if value is None:
        return

    now = time.monotonic()

    if key == "robot_position":
        if now - _pos_last_t < POSITION_SAMPLE_S:
            return
        with _perf.measure("pos_history"):
            try:
                pos = json.loads(value)
                entry = {"x": float(pos["x"]), "y": float(pos["y"]), "t": round(_elapsed(), 3)}
            except Exception:
                return
            _pos_last_t = now
            with _pos_lock:
                _pos_history.append(entry)
                _prune_list(_pos_history)
                snapshot = list(_pos_history)
            mb.set("position_history", json.dumps(snapshot))

    elif key == "other_robots":
        if now - _robots_last_t < POSITION_SAMPLE_S:
            return
        with _perf.measure("robots_history"):
            try:
                payload = json.loads(value)
                # Support both new {"origin":…,"robots":[…]} and old bare-list formats.
                robot_list = payload.get("robots", payload) if isinstance(payload, dict) else payload
                robots = [{"x": float(r["x"]), "y": float(r["y"]),
                           "id": int(r.get("id", 0))}
                          for r in robot_list]
            except Exception:
                return
            _robots_last_t = now
            entry = {"t": round(_elapsed(), 3), "robots": robots}
            with _robots_lock:
                _robots_history.append(entry)
                _prune_list(_robots_history)
                snapshot = list(_robots_history)
            mb.set("other_robots_history", json.dumps(snapshot))

    elif key == "ball":
        if now - _ball_last_t < BALL_SAMPLE_S:
            return
        with _perf.measure("ball_history"):
            try:
                pos = json.loads(value).get("global_pos")
                if pos is None:
                    return
                entry = {"x": float(pos["x"]), "y": float(pos["y"]), "t": round(_elapsed(), 3)}
            except Exception:
                return
            _ball_last_t = now
            with _ball_lock:
                _ball_history.append(entry)
                _prune_list(_ball_history)
                snapshot = list(_ball_history)
            mb.set("ball_history", json.dumps(snapshot))


if __name__ == "__main__":
    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    threading.Thread(target=_time_loop, daemon=True, name="time-publisher").start()

    mb.setcallback(["robot_position", "other_robots", "ball"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping time node.")
        mb.close()
