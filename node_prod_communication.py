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

# Our own robot state — sent outward over serial
_own_pos      = None   # {"x": float, "y": float} from robot_position
_other_robots = None   # {"robots": [...]} from other_robots (detections + predictions)
_ball         = None   # {"global_pos": {...}, ...} from ball
_ball_lost    = False  # bool from ball_lost


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


# ── Outgoing serial frame builder ─────────────────────────────────────────────

def _build_outgoing_frame():
    """Build a cooperation frame from our robot's current state."""
    frame = {"type": "communication"}

    if _own_pos is not None:
        frame["main_robot_pos"] = {
            "x": round(float(_own_pos["x"]), 4),
            "y": round(float(_own_pos["y"]), 4),
            "confidence": 0.95,
        }

    if _other_robots is not None:
        robots = _other_robots.get("robots", [])
        det_slot  = 1
        pred_slot = 1
        for r in robots:
            x = r.get("x")
            y = r.get("y")
            if x is None or y is None:
                continue
            conf = float(r.get("confidence", 0.0))
            if r.get("method") == "predicted":
                if pred_slot <= 3:
                    frame[f"other_pred_{pred_slot}"] = {
                        "x": round(float(x), 4),
                        "y": round(float(y), 4),
                        "confidence": conf,
                    }
                    pred_slot += 1
            else:
                if det_slot <= 3:
                    frame[f"other_pos_{det_slot}"] = {
                        "x": round(float(x), 4),
                        "y": round(float(y), 4),
                        "confidence": conf,
                    }
                    det_slot += 1

    if _ball is not None:
        gpos = _ball.get("global_pos")
        if gpos is not None:
            bconf = float(gpos.get("confidence", 0.8))
            entry = {
                "x": round(float(gpos["x"]), 4),
                "y": round(float(gpos["y"]), 4),
                "confidence": bconf,
            }
            if _ball_lost:
                frame["ball_pred"] = entry
            else:
                frame["ball_pos"] = entry

    return frame


def _send_loop(reader_ref):
    """Periodically transmit our robot's state over serial at ~10 Hz."""
    interval = 0.1
    while True:
        frame = _build_outgoing_frame()
        reader_ref[0].send(frame)
        time.sleep(interval)


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _sim_state, _ball_sim_pos, _own_pos, _other_robots, _ball, _ball_lost

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
            _ball         = payload
        except (json.JSONDecodeError, TypeError):
            pass

    elif key == "ball_lost":
        try:
            _ball_lost = bool(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            pass

    elif key == "robot_position":
        try:
            _own_pos = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

    elif key == "other_robots":
        try:
            _other_robots = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key in ("sim_state", "ball", "ball_lost", "robot_position", "other_robots"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["sim_state", "ball", "ball_lost", "robot_position", "other_robots"],
                   on_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    reader    = _make_reader()
    frame_cb  = on_sim_frame if isinstance(reader, SimCooperationReader) else on_frame
    reader.start(frame_cb)

    # Mutable container so _send_loop can access the reader instance
    _reader_ref = [reader]
    threading.Thread(target=_send_loop, args=(_reader_ref,), daemon=True,
                     name="coop-sender").start()

    _shutdown = threading.Event()
    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[COOP] Stopped.")
        reader.stop()
        mb.close()
