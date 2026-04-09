from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import time

# ── Field dimensions ──────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.58   # metres — playing field only
FIELD_HEIGHT = 2.19
ROBOT_RADIUS = 0.09

# ── Dead-reckoning limits ─────────────────────────────────────────────────────
_MAX_PRED_STEPS = 20
_MAX_PRED_DT    = 0.5

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_dev_predict_robots", broker=mb)

# Cache of last known state per tracked robot ID, populated from other_robots_detected.
_robot_last = {}   # id → {"x", "y", "vx", "vy", "t"}


def _predict_with_bounce(x, y, vx, vy, dt):
    dt   = min(dt, _MAX_PRED_DT)
    n    = max(1, min(int(dt / 0.02) + 1, _MAX_PRED_STEPS))
    step = dt / n
    for _ in range(n):
        x += vx * step;  y += vy * step
        if   x < ROBOT_RADIUS:               x = ROBOT_RADIUS;               vx =  abs(vx)
        elif x > FIELD_WIDTH - ROBOT_RADIUS:  x = FIELD_WIDTH - ROBOT_RADIUS; vx = -abs(vx)
        if   y < ROBOT_RADIUS:               y = ROBOT_RADIUS;               vy =  abs(vy)
        elif y > FIELD_HEIGHT - ROBOT_RADIUS: y = FIELD_HEIGHT - ROBOT_RADIUS; vy = -abs(vy)
    return x, y


def on_update(key, value):
    global _robot_last

    if value is None:
        return

    with _perf.measure("predict"):
        try:
            payload  = json.loads(value)
            origin   = payload.get("origin")
            detected = payload.get("robots", [])
            det_t    = payload.get("t", time.monotonic())
        except Exception:
            return

        now = time.monotonic()

        # Update cache with currently detected robots
        detected_ids = set()
        for r in detected:
            rid = r.get("id")
            if rid is not None:
                detected_ids.add(rid)
                _robot_last[rid] = {
                    "x": r["x"], "y": r["y"],
                    "vx": r.get("vx", 0.0), "vy": r.get("vy", 0.0),
                    "t": det_t,
                }

        # Dead-reckon robots that were tracked but not detected this frame
        result = list(detected)
        for rid, last in list(_robot_last.items()):
            if rid in detected_ids:
                continue
            dt = now - last["t"]
            px, py = _predict_with_bounce(last["x"], last["y"],
                                          last["vx"], last["vy"], dt)
            result.append({
                "x": round(px, 3), "y": round(py, 3),
                "pts": 0, "method": "predicted", "confidence": 0.0,
                "id": rid,
                "vx": round(last["vx"], 3), "vy": round(last["vy"], 3),
            })

        mb.set("other_robots", json.dumps({"origin": origin, "robots": result}))


if __name__ == "__main__":
    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    print("[PREDICT_ROBOTS] Starting robot prediction node...")
    mb.setcallback(["other_robots_detected"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\n[PREDICT_ROBOTS] Stopped.")
        mb.close()
