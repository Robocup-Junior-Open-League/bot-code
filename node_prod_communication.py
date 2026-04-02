import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import math
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
_other_robots = None  # {"origin": ..., "robots": [{x, y, id, confidence, ...}, ...]}
_robot_pos    = None  # (x, y) metres — this system's own position
_ball_pos     = None  # {"x": ..., "y": ..., "confidence": ...} or None
_sim_state    = None   # {"robot": [x,y], "obstacles": [[x,y],...]} from sim_state
_ball_sim_pos = None   # {"x": float, "y": float} — true sim ball position
_ally_id      = None   # persistent ally robot ID once identified


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _xy(d):
    """Return (x, y) from a position dict, or None on failure."""
    if d is None:
        return None
    try:
        return float(d["x"]), float(d["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _conf(d, default=1.0):
    """Return the confidence value from a position dict, falling back to *default*."""
    if d is None:
        return default
    try:
        return float(d.get("confidence", default))
    except (TypeError, ValueError):
        return default


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _fuse(pos_a, conf_a, pos_b, conf_b):
    """Confidence-weighted average of two (x, y) positions."""
    total = conf_a + conf_b
    if total < 1e-9:
        return pos_a
    wa, wb = conf_a / total, conf_b / total
    return (
        round(wa * pos_a[0] + wb * pos_b[0], 3),
        round(wa * pos_a[1] + wb * pos_b[1], 3),
    )


# ── Frame handler ─────────────────────────────────────────────────────────────

def _extract_ally_data(data):
    """Extract and publish raw ally fields to the broker under the ally_ prefix."""
    with _perf.measure("extract"):
        for key in ("main_robot_pos", "other_pos_1", "other_pos_2", "other_pos_3",
                    "ball_pos", "other_pred_1", "other_pred_2", "other_pred_3"):
            if key in data:
                mb.set(f"ally_{key}", json.dumps(data[key]))

    return (data.get("main_robot_pos"),
            [data.get(f"other_pos_{i}") for i in range(1, 4)],
            [data.get(f"other_pred_{i}") for i in range(1, 4)],
            data.get("ball_pos"))


def _match_and_fuse_ally_data(ally_main, ally_others, ally_preds, ally_ball):
    """
    Match and fuse ally data with our own robot observations.

    Steps:
      2. Identify the ally in our other_robots list (detected or predicted) by
         known ID first, then proximity to main_robot_pos as fallback.
         - Detected match  → update position to received value, tag as ally.
         - Predicted match → replace prediction with received position, tag as ally.
      3. Among other_pos_1/2/3, discard the entry closest to this system's own
         robot_position (the ally is observing us).
      4. For each remaining ally detection, match to nearest unmatched robot
         (detected or predicted):
         - Detected match  → confidence-weighted position fusion.
         - Predicted match → replace prediction with ally's observation.
      5. For each ally prediction (other_pred_*), match to nearest unmatched robot:
         - Detected match  → discard ally prediction (detection wins, no change).
         - Predicted match → update our prediction to the mean of both predictions.
      6. Publish updated other_robots.
      7. Fuse ball positions (confidence-weighted, or direct if no local value).
    """
    global _ally_id

    with _perf.measure("match"):
        if _other_robots is None:
            return  # not enough broker state to fuse yet

        robots  = [dict(r) for r in _other_robots.get("robots", [])]
        origin  = _other_robots.get("origin")
        matched = set()  # robot indices consumed across all steps

        # ── Step 2: Identify ally robot (detected or predicted) ───────────────
        ally_idx      = None
        ally_main_pos = _xy(ally_main)
        if ally_main_pos is not None and robots:
            # Prefer match by known persistent ID; fall back to proximity.
            if _ally_id is not None:
                ally_idx = next(
                    (i for i, r in enumerate(robots) if r.get("id") == _ally_id),
                    None,
                )
            if ally_idx is None:
                ally_idx = min(
                    range(len(robots)),
                    key=lambda i: _dist((robots[i]["x"], robots[i]["y"]), ally_main_pos),
                )
            _ally_id = robots[ally_idx].get("id", ally_idx)
            matched.add(ally_idx)
            if robots[ally_idx].get("method") == "predicted":
                # Prediction confirmed — replace with ally's self-reported position.
                robots[ally_idx].update({
                    "x": ally_main_pos[0], "y": ally_main_pos[1],
                    "method": "cluster", "ally": True,
                })
            else:
                # Detected — update to ally's self-reported position directly.
                robots[ally_idx]["x"]    = ally_main_pos[0]
                robots[ally_idx]["y"]    = ally_main_pos[1]
                robots[ally_idx]["ally"] = True
            mb.set("ally_id", str(_ally_id))

        # ── Step 3: Discard the ally's observation that matches our position ──
        remaining_ally = list(ally_others)
        if _robot_pos is not None:
            candidates = [
                (i, _xy(p))
                for i, p in enumerate(remaining_ally)
                if _xy(p) is not None
            ]
            if candidates:
                self_idx = min(candidates,
                               key=lambda t: _dist(t[1], _robot_pos))[0]
                remaining_ally[self_idx] = None

        # ── Step 4: Fuse ally detections (detected or predicted match) ────────
        sys_indices    = [i for i in range(len(robots)) if i != ally_idx]
        ally_det_valid = [
            (_xy(p), _conf(p))
            for p in remaining_ally
            if p is not None and _xy(p) is not None
        ]

        for ally_pos, ally_c in ally_det_valid:
            unmatched = [i for i in sys_indices if i not in matched]
            if not unmatched:
                break
            best = min(unmatched,
                       key=lambda i: _dist((robots[i]["x"], robots[i]["y"]), ally_pos))
            matched.add(best)
            if robots[best].get("method") == "predicted":
                # Replace stale prediction with ally's fresher observation.
                robots[best]["x"]      = ally_pos[0]
                robots[best]["y"]      = ally_pos[1]
                robots[best]["method"] = "cluster"
            else:
                # Both detected — confidence-weighted fusion.
                sys_pos = (robots[best]["x"], robots[best]["y"])
                sys_c   = float(robots[best].get("confidence", 1.0))
                fx, fy  = _fuse(sys_pos, sys_c, ally_pos, ally_c)
                robots[best]["x"] = fx
                robots[best]["y"] = fy

        # ── Step 5: Match ally predictions ────────────────────────────────────
        ally_pred_valid = [
            (_xy(p), _conf(p))
            for p in ally_preds
            if p is not None and _xy(p) is not None
        ]

        for ally_pred_pos, _ in ally_pred_valid:
            unmatched = [i for i in sys_indices if i not in matched]
            if not unmatched:
                break
            best = min(unmatched,
                       key=lambda i: _dist((robots[i]["x"], robots[i]["y"]), ally_pred_pos))
            matched.add(best)
            if robots[best].get("method") == "predicted":
                # Both predicted — update ours to the mean.
                robots[best]["x"] = round((robots[best]["x"] + ally_pred_pos[0]) / 2, 3)
                robots[best]["y"] = round((robots[best]["y"] + ally_pred_pos[1]) / 2, 3)
            # else: our detection beats ally's prediction — no change.

        # ── Step 6: Publish updated other_robots ──────────────────────────────
        mb.set("other_robots", json.dumps({"origin": origin, "robots": robots}))

        # ── Step 7: Fuse ball position ─────────────────────────────────────────
        if ally_ball is not None:
            ally_ball_pos  = _xy(ally_ball)
            ally_ball_conf = _conf(ally_ball)
            if ally_ball_pos is not None:
                if _ball_pos is not None:
                    sys_ball_pos  = _xy(_ball_pos)
                    sys_ball_conf = _conf(_ball_pos)
                    if sys_ball_pos is not None:
                        fx, fy = _fuse(sys_ball_pos, sys_ball_conf,
                                       ally_ball_pos, ally_ball_conf)
                        mb.set("ball_pos", json.dumps({
                            "x": fx, "y": fy,
                            "confidence": round(
                                (sys_ball_conf + ally_ball_conf) / 2, 3),
                        }))
                        return
                # No local ball reading — publish the ally's value directly.
                mb.set("ball_pos", json.dumps(ally_ball))


def _process_frame(data):
    """Process one cooperation frame: extract data, then match and fuse."""
    ally_main, ally_others, ally_preds, ally_ball = _extract_ally_data(data)
    _match_and_fuse_ally_data(ally_main, ally_others, ally_preds, ally_ball)


def on_frame(data):
    with _perf.measure("hw_frame"):
        _process_frame(data)


def on_sim_frame(data):
    with _perf.measure("sim_frame"):
        _process_frame(data)


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _other_robots, _robot_pos, _ball_pos, _sim_state, _ball_sim_pos

    if value is None:
        return

    if key == "other_robots":
        try:
            _other_robots = json.loads(value)
            if _ally_id is not None:
                mb.set("ally_id", str(_ally_id))
        except (json.JSONDecodeError, TypeError):
            pass

    elif key == "robot_position":
        try:
            pos        = json.loads(value)
            _robot_pos = (float(pos["x"]), float(pos["y"]))
        except Exception:
            pass

    elif key == "ball_pos":
        try:
            _ball_pos = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

    elif key == "sim_state":
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
    for key in ("other_robots", "robot_position", "ball_pos", "sim_state", "ball"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["other_robots", "robot_position", "ball_pos", "sim_state", "ball"], on_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    reader    = _make_reader()
    frame_cb  = on_sim_frame if isinstance(reader, SimCooperationReader) else on_frame
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
