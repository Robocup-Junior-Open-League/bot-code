import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import math
import time
import threading
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
from utils.cooperation_reader import SPICooperationReader, SimCooperationReader

# ── Reader factory ────────────────────────────────────────────────────────────
# To swap the transport layer, return a different BaseCooperationReader here.
COOP_SIM_REPLACE = True

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_communication", broker=mb, print_every=100)


def _make_reader():
    spi_path = f"/dev/spidev{SPICooperationReader.DEFAULT_BUS}.{SPICooperationReader.DEFAULT_DEVICE}"
    if COOP_SIM_REPLACE and not os.path.exists(spi_path):
        return SimCooperationReader(
            get_sim_state=lambda: _sim_state,
            get_ball_sim=lambda: _ball_sim_pos,
        )
    return SPICooperationReader()


# ── Motor control constants ───────────────────────────────────────────────────
# Mirrors the C++ motorSpeeds() / setMotorSpeeds() constants.
_KP               = 0.8    # proportional gain
_MIN_TURN_SPEED   = 30     # % — minimum power during a proportional turn
_TOLERANCE_DEG    = 10.0   # degrees — dead-band (full forward / full reverse)
_MIN_STEP_DELAY   = 200    # minimum step delay in µs at 100% speed
_MAX_STEP_DELAY   = 800    # minimum step delay in µs at 100% speed
_ROTATE_OFFSET_DEG = 90.0  # spin offset when ball is unknown — drives constant rotation
# ─────────────────────────────────────────────────────────────────────────────

# Broker state — updated by on_update()
_sim_state       = None   # {"robot": [x,y], "obstacles": [[x,y],...]} from sim_state
_ball_sim_pos    = None   # {"x": float, "y": float} — true sim ball position

# Our own robot state — sent outward over SPI
_own_pos         = None   # {"x": float, "y": float} from robot_position
_other_robots    = None   # {"robots": [...]} from other_robots (detections + predictions)
_ball            = None   # {"global_pos": {...}, ...} from ball
_ball_lost       = False  # bool from ball_lost
_strategy_points = []     # [{"x": float, "y": float}, ...] from robot_strategy_points
_imu_heading     = None   # degrees — robot heading from imu_pitch


# ── Motor control helpers ─────────────────────────────────────────────────────

def _compute_steering_error():
    """
    Return steering error in radians, or None if data is unavailable.
    Equivalent to driveDir() in C++.
    """
    if not _strategy_points or _own_pos is None or _imu_heading is None:
        return None
    try:
        goal_x   = float(_strategy_points[0]["x"])
        goal_y   = float(_strategy_points[0]["y"])
        robot_x  = float(_own_pos["x"])
        robot_y  = float(_own_pos["y"])
    except (KeyError, TypeError, ValueError):
        return None

    target = math.atan2(goal_y - robot_y, goal_x - robot_x)
    error  = target - math.radians(_imu_heading)
    # Normalise to [-π, π]
    while error >  math.pi: error -= 2 * math.pi
    while error < -math.pi: error += 2 * math.pi
    return error


def _motor_speeds(error_rad):
    """
    Convert steering error (radians) to (left, right) speed percentages
    in the range [-100, 100].  Equivalent to motorSpeeds() in C++.
    """
    error_deg = math.degrees(error_rad)

    if abs(error_deg) <= _TOLERANCE_DEG:
        # Within forward dead-band — drive straight ahead
        return 100, 100
    if abs(error_deg) >= (180.0 - _TOLERANCE_DEG):
        # Within backward dead-band — reverse
        return -100, -100

    # Proportional turn
    p_speed = int(abs(error_deg) * _KP)
    p_speed = max(_MIN_TURN_SPEED, min(100, p_speed))
    if error_deg > 0:
        return -p_speed, p_speed   # target left  → spin CCW
    return p_speed, -p_speed       # target right → spin CW


def _compute_spin_error():
    """Return the spin error in degrees (target_dir − imu_heading), normalised
    to [−180, 180], or None if required data is missing.

    Target direction:
      • Ball known → angle from our position toward strategy_points[0]["dir"].
      • Ball unknown → imu_heading + _ROTATE_OFFSET_DEG (constant rotation).
    """
    if _imu_heading is None:
        return None

    ball_known = (
        _ball is not None
        and _ball.get("global_pos") is not None
        and not _ball_lost
    )

    if ball_known and _strategy_points:
        pt = _strategy_points[0]
        d  = pt.get("dir")
        if d is not None and _own_pos is not None:
            try:
                dx = float(d["x"]) - float(_own_pos["x"])
                dy = float(d["y"]) - float(_own_pos["y"])
            except (KeyError, TypeError, ValueError):
                return None
            target_deg = math.degrees(math.atan2(dy, dx))
        else:
            return None
    else:
        target_deg = _imu_heading + _ROTATE_OFFSET_DEG

    error = target_deg - _imu_heading
    while error >  180.0: error -= 360.0
    while error < -180.0: error += 360.0
    return error


def _spin_k_fields(error_deg):
    """Convert a spin error (degrees) to k-motor frame fields."""
    if abs(error_deg) <= _TOLERANCE_DEG:
        return {"s": 0, "d": 0}
    p_speed = int(abs(error_deg) * _KP)
    p_speed = max(_MIN_TURN_SPEED, min(100, p_speed))
    steps   = round(_MIN_STEP_DELAY + (_MAX_STEP_DELAY - _MIN_STEP_DELAY) * p_speed / 100)
    return {"s": steps, "d": 1 if error_deg > 0 else 0}


def _motor_fields(left, right):
    """
    Convert signed speed percentages to the l/r/k/sp frame fields.
    Equivalent to setMotorSpeeds() in C++.
    """
    left  = max(-100, min(100, left))
    right = max(-100, min(100, right))
    steps_l = round(_MIN_STEP_DELAY + (_MAX_STEP_DELAY - _MIN_STEP_DELAY) * abs(left) / 100)
    steps_r = round(_MIN_STEP_DELAY + (_MAX_STEP_DELAY - _MIN_STEP_DELAY) * abs(right) / 100)
    dir_l   = 1 if left  >= 0 else 0
    dir_r   = 1 if right >= 0 else 0

    fields = {
        "l": {"s": steps_l, "d": dir_l},
        "r": {"s": steps_r, "d": dir_r},
        "k": {"s": 0, "d": 0},
    }
    # Include sp only when it deviates from the default (100), matching
    # the C++ drive() convention and keeping frames compact.
    dominant = max(abs(left), abs(right))
    if dominant < 100:
        fields["sp"] = dominant
    return fields


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
    # ── Motor control ──────────────────────────────────────────────────────────
    error = _compute_steering_error()
    if error is not None:
        left, right = _motor_speeds(error)
        frame = _motor_fields(left, right)
    else:
        frame = {"l": {"s": 0, "d": 0}, "r": {"s": 0, "d": 0}, "k": {"s": 0, "d": 0}}

    spin_error = _compute_spin_error()
    if spin_error is not None:
        frame["k"] = _spin_k_fields(spin_error)

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
    """Periodically transmit our robot's state at 20 Hz (every 50 ms)."""
    interval = 0.05
    while True:
        frame = _build_outgoing_frame()
        reader_ref[0].send(frame)
        time.sleep(interval)


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _sim_state, _ball_sim_pos, _own_pos, _other_robots, _ball, _ball_lost
    global _strategy_points, _imu_heading

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

    elif key == "robot_strategy_points":
        try:
            _strategy_points = json.loads(value) or []
        except (json.JSONDecodeError, TypeError):
            pass

    elif key == "imu_pitch":
        try:
            _imu_heading = float(value)
        except (ValueError, TypeError):
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    for key in ("sim_state", "ball", "ball_lost", "robot_position", "other_robots"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["sim_state", "ball", "ball_lost", "robot_position", "other_robots",
                    "robot_strategy_points", "imu_pitch"],
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
