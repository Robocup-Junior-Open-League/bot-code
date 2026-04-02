import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import time
import threading
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
from utils.cooperation_reader import SerialCooperationReader, SimCooperationReader

# ── Reader factory ────────────────────────────────────────────────────────────
# To swap the transport layer, return a different BaseCooperationReader here.
COOP_SIM_REPLACE = True

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_communication", broker=mb, print_every=100)


def _make_reader():
    if COOP_SIM_REPLACE and not os.path.exists(SerialCooperationReader.DEFAULT_PORT):
        return SimCooperationReader(
            get_sim_state=lambda: _sim_state,
            get_ball_sim=lambda: _ball_sim_pos,
        )
    return SerialCooperationReader()


# Broker state — updated by on_update()
_sim_state    = None   # {"robot": [x,y], "obstacles": [[x,y],...]} from sim_state
_ball_sim_pos = None   # {"x": float, "y": float} — true sim ball position


# ── Frame handler ─────────────────────────────────────────────────────────────

def _process_frame(data):
    """
    Publish all ally observations as a single ally_data blob (consumed by the
    positioning node for robot matching) and individual ally_* keys (for
    twin_vis).  Also fuse ball position locally.
    """
    t = time.monotonic()

    def _norm(d):
        if d is None:
            return None
        try:
            return {"x": float(d["x"]), "y": float(d["y"]),
                    "confidence": float(d.get("confidence", 1.0))}
        except (KeyError, TypeError, ValueError):
            return None

    # Individual fields for twin_vis
    for key in ("main_robot_pos", "other_pos_1", "other_pos_2", "other_pos_3",
                "ball_pos", "ball_pred",
                "other_pred_1", "other_pred_2", "other_pred_3"):
        if key in data:
            mb.set(f"ally_{key}", json.dumps(data[key]))

    # Bundled payload — positioning node consumes this for full robot matching
    mb.set("ally_data", json.dumps({
        "t":          round(t, 4),
        "main_pos":   _norm(data.get("main_robot_pos")),
        "other_pos":  [_norm(data.get(f"other_pos_{i}"))  for i in range(1, 4)],
        "other_pred": [_norm(data.get(f"other_pred_{i}")) for i in range(1, 4)],
        "ball_pos":   _norm(data.get("ball_pos")),
        "ball_pred":  _norm(data.get("ball_pred")),
    }))



def on_frame(data):
    with _perf.measure("hw_extract"):
        _process_frame(data)


def on_sim_frame(data):
    with _perf.measure("sim_extract"):
        _process_frame(data)


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _sim_state, _ball_sim_pos

    if value is None:
        return

    if key == "sim_state":
        try:
            _sim_state = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

    elif key == "ball":
        try:
            payload       = json.loads(value)
            _ball_sim_pos = payload.get("sim_pos")
        except (json.JSONDecodeError, TypeError):
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key in ("sim_state", "ball"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["sim_state", "ball"], on_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    reader   = _make_reader()
    frame_cb = on_sim_frame if isinstance(reader, SimCooperationReader) else on_frame
    reader.start(frame_cb)

    _shutdown = threading.Event()
    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[COOP] Stopped.")
        reader.stop()
        mb.close()
