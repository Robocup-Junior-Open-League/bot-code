from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
from collections import deque
import json
import math
import time
import numpy as np

# ── Field dimensions ──────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.82   # metres
FIELD_HEIGHT = 2.43
ROBOT_RADIUS = 0.09

# ── Robot dead-reckoning ──────────────────────────────────────────────────────
_MAX_PRED_STEPS = 20
_MAX_PRED_DT    = 0.5

# ── Ball prediction ───────────────────────────────────────────────────────────
FOV_DEG              = 62.2
BALL_RADIUS_MM       = 21.0
BALL_VEL_HISTORY_N   = 10
BALL_VEL_HISTORY_MIN = 3
BALL_VEL_MIN_DT      = 0.05   # seconds
MAX_BALL_SPEED       = 3.0    # m/s

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_prediction", broker=mb)

# ── Context from broker ───────────────────────────────────────────────────────
_robot_pos = None
_imu_pitch = None
_sim_state = None

# ── Robot dead-reckoning state ────────────────────────────────────────────────
_robot_last = {}   # id → {"x", "y", "vx", "vy", "t"}

# ── Ball prediction state ─────────────────────────────────────────────────────
_vel_history      = deque(maxlen=BALL_VEL_HISTORY_N)
_vel_history_dirty = False   # True when a new sample was appended since last fit
_vel_last_t       = -999.0
_last_detection_t = -999.0
_last_ball_vx     = 0.0
_last_ball_vy     = 0.0
_hidden_state     = None
_hidden_state_t   = None
_ball_lost        = False


# ── Robot dead-reckoning ──────────────────────────────────────────────────────

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


# ── Ball prediction ───────────────────────────────────────────────────────────

def _fit_ball_velocity(history):
    if len(history) < BALL_VEL_HISTORY_MIN:
        return 0.0, 0.0
    arr = np.array(history, dtype=float)  # (N, 3): t, x, y
    ts  = arr[:, 0] - arr[0, 0]
    if ts[-1] < 1e-9:
        return 0.0, 0.0
    # Closed-form linear regression — avoids np.polyfit's SVD overhead
    n     = len(ts)
    St    = ts.sum()
    Stt   = (ts * ts).sum()
    denom = n * Stt - St * St
    if abs(denom) < 1e-9:
        return 0.0, 0.0
    xy   = arr[:, 1:3]
    Sxy  = (ts[:, None] * xy).sum(axis=0)
    Sval = xy.sum(axis=0)
    vx, vy = (n * Sxy - St * Sval) / denom
    return float(vx), float(vy)


# Robot positions cached and only rebuilt when context keys change
_robot_positions_cache: list = []


def _rebuild_robot_positions_cache():
    global _robot_positions_cache
    if _sim_state is not None:
        robots = []
        r = _sim_state.get("robot")
        if r:
            robots.append((float(r[0]), float(r[1])))
        robots += [(float(p[0]), float(p[1])) for p in _sim_state.get("obstacles", [])]
        _robot_positions_cache = robots
    elif _robot_pos is not None:
        _robot_positions_cache = [_robot_pos]
    else:
        _robot_positions_cache = []


def _in_camera_fov(bx, by):
    if _robot_pos is None or _imu_pitch is None:
        return False
    heading_rad = math.radians(_imu_pitch)
    cam_x = _robot_pos[0] + ROBOT_RADIUS * math.cos(heading_rad)
    cam_y = _robot_pos[1] + ROBOT_RADIUS * math.sin(heading_rad)
    dx, dy  = bx - cam_x, by - cam_y
    local_z =  dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
    local_x = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)
    if local_z <= 0:
        return False
    return math.degrees(math.atan2(abs(local_x), local_z)) <= FOV_DEG / 2.0


def _extrapolate_ball(x, y, vx, vy, dt, robots=None):
    STEP = 0.02
    r    = BALL_RADIUS_MM / 1000.0
    sep  = ROBOT_RADIUS + r
    n    = max(1, round(dt / STEP))
    step = dt / n
    for _ in range(n):
        x += vx * step;  y += vy * step
        if   x < r:                x = r;                vx =  abs(vx)
        elif x > FIELD_WIDTH  - r: x = FIELD_WIDTH  - r; vx = -abs(vx)
        if   y < r:                y = r;                vy =  abs(vy)
        elif y > FIELD_HEIGHT - r: y = FIELD_HEIGHT - r; vy = -abs(vy)
        if robots:
            for rx, ry in robots:
                dx, dy = x - rx, y - ry
                dist   = math.hypot(dx, dy)
                if 0 < dist < sep:
                    nx, ny = dx / dist, dy / dist
                    x, y   = rx + nx * sep, ry + ny * sep
                    dot    = vx * nx + vy * ny
                    if dot < 0:
                        vx -= 2 * dot * nx
                        vy -= 2 * dot * ny
    return x, y, vx, vy


# ── Broker callback ───────────────────────────────────────────────────────────

def on_update(key, value):
    global _robot_pos, _imu_pitch, _sim_state
    global _robot_last
    global _vel_history, _vel_history_dirty, _vel_last_t, _last_detection_t
    global _last_ball_vx, _last_ball_vy
    global _hidden_state, _hidden_state_t, _ball_lost

    if value is None:
        return

    if key == "robot_position":
        try:
            p = json.loads(value)
            _robot_pos = (float(p["x"]), float(p["y"]))
            _rebuild_robot_positions_cache()
        except Exception:
            pass
        return

    if key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except Exception:
            pass
        return

    if key == "sim_state":
        try:
            _sim_state = json.loads(value)
            _rebuild_robot_positions_cache()
        except Exception:
            pass
        return

    # ── Robot dead-reckoning ──────────────────────────────────────────────────
    if key == "other_robots_detected":
        with _perf.measure("robots"):
            try:
                payload  = json.loads(value)
                origin   = payload.get("origin")
                detected = payload.get("robots", [])
                det_t    = payload.get("t", time.monotonic())
            except Exception:
                return

            now = time.monotonic()
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
        return

    # ── Ball prediction ───────────────────────────────────────────────────────
    if key == "ball_raw":
        with _perf.measure("ball"):
            try:
                raw = json.loads(value)
            except Exception:
                return

            now_t  = time.monotonic()
            gpos   = raw.get("global_pos")
            hidden_pos = None
            pub_vx, pub_vy = _last_ball_vx, _last_ball_vy

            if gpos is not None:
                _last_detection_t = now_t
                _ball_lost        = False
                _hidden_state     = [gpos["x"], gpos["y"], _last_ball_vx, _last_ball_vy]
                _hidden_state_t   = None

                if now_t - _vel_last_t >= BALL_VEL_MIN_DT:
                    ok = True
                    if _vel_history:
                        dt_s = now_t - _vel_history[-1][0]
                        if math.hypot(gpos["x"] - _vel_history[-1][1],
                                      gpos["y"] - _vel_history[-1][2]
                                      ) > MAX_BALL_SPEED * dt_s * 1.5:
                            ok = False
                    if ok:
                        _vel_history.append((now_t, gpos["x"], gpos["y"]))  # deque: O(1), auto-truncates
                        _vel_history_dirty = True
                    _vel_last_t = now_t
            else:
                if now_t - _last_detection_t > 1.0:
                    _vel_history.clear()
                    _vel_history_dirty = False
                    _vel_last_t = -999.0

            # Only refit when a new sample was actually added
            if _vel_history_dirty and len(_vel_history) >= BALL_VEL_HISTORY_MIN:
                vx_fit, vy_fit = _fit_ball_velocity(_vel_history)
                spd = math.hypot(vx_fit, vy_fit)
                if spd > MAX_BALL_SPEED:
                    vx_fit *= MAX_BALL_SPEED / spd
                    vy_fit *= MAX_BALL_SPEED / spd
                _last_ball_vx = vx_fit
                _last_ball_vy = vy_fit
                if _hidden_state is not None:
                    _hidden_state[2] = vx_fit
                    _hidden_state[3] = vy_fit
                _vel_history_dirty = False

            if gpos is None and _hidden_state is not None:
                if _ball_lost or _in_camera_fov(_hidden_state[0], _hidden_state[1]):
                    _ball_lost = True
                    pub_vx = pub_vy = 0.0
                else:
                    if _hidden_state_t is None:
                        _hidden_state_t = now_t
                    else:
                        dt_frame = now_t - _hidden_state_t
                        if dt_frame > 0:
                            hx, hy, hvx, hvy = _extrapolate_ball(
                                _hidden_state[0], _hidden_state[1],
                                _hidden_state[2], _hidden_state[3],
                                dt_frame, robots=_robot_positions_cache,
                            )
                            _hidden_state   = [hx, hy, hvx, hvy]
                            _hidden_state_t = now_t
                    pub_vx = _hidden_state[2]
                    pub_vy = _hidden_state[3]
                hidden_pos = {"x": round(_hidden_state[0], 3),
                              "y": round(_hidden_state[1], 3)}

            result = dict(raw)
            result["vx"]         = round(pub_vx, 3)
            result["vy"]         = round(pub_vy, 3)
            result["hidden_pos"] = hidden_pos
            result["ball_lost"]  = _ball_lost

            mb.set("ball",      json.dumps(result))
            mb.set("ball_lost", json.dumps(_ball_lost))


if __name__ == "__main__":
    for key, target, parse in [
        ("robot_position", "_robot_pos", lambda v: (lambda p: (float(p["x"]), float(p["y"])))(json.loads(v))),
        ("imu_pitch",      "_imu_pitch", float),
        ("sim_state",      "_sim_state", json.loads),
    ]:
        try:
            val = mb.get(key)
            if val is not None:
                globals()[target] = parse(val)
        except Exception:
            pass

    _rebuild_robot_positions_cache()
    print("[PREDICTION] Starting combined prediction node...")
    mb.setcallback(
        ["ball_raw", "other_robots_detected", "robot_position", "imu_pitch", "sim_state"],
        on_update,
    )
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\n[PREDICTION] Stopped.")
        mb.close()
