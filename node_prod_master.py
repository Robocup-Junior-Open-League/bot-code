import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import math
import time
import numpy as np

from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor


# ── Field dimensions ──────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.58   # metres — playing field only
FIELD_HEIGHT = 2.19

# ── Teams ─────────────────────────────────────────────────────────────────────
# Team 0 = our team  — bottom goal  (y = 0)
# Team 1 = enemy     — top    goal  (y = FIELD_HEIGHT)
TEAM_US    = 0
TEAM_ENEMY = 1

# ── Ball control ──────────────────────────────────────────────────────────────
ROBOT_RADIUS       = 0.09
BALL_RADIUS        = 0.021
BALL_CONTROL_DIST  = ROBOT_RADIUS + BALL_RADIUS + 0.10  # ≈ 0.21 m
BALL_CONTROL_DIST_SQ = BALL_CONTROL_DIST ** 2
BALL_CONTROL_DWELL = 0.3   # seconds ball must stay in range to confirm control

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_master", broker=mb, print_every=100)

_robot_pos    = None
_other_robots = None
_ball         = None
_ally_id      = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dist(ax, ay, bx, by):
    """Return the Euclidean distance between two points."""
    return math.hypot(ax - bx, ay - by)


def _closest_on_segment(ax, ay, bx, by, px, py):
    """Return the closest point on the line segment AB to point P."""
    dx, dy = bx - ax, by - ay
    lsq = dx * dx + dy * dy
    if lsq < 1e-12:
        return ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / lsq))
    return ax + t * dx, ay + t * dy


def _dist_to_segment_np(ax, ay, bx, by, robots_np):
    """Vectorized distance from each robot position to line segment AB."""
    A = np.array([ax, ay])
    B = np.array([bx, by])
    AB = B - A
    AP = robots_np - A

    ab2 = np.dot(AB, AB)
    if ab2 < 1e-12:
        return np.linalg.norm(AP, axis=1)

    t = np.clip(np.dot(AP, AB) / ab2, 0, 1)
    closest = A + np.outer(t, AB)

    return np.linalg.norm(robots_np - closest, axis=1)


# ── Broker state ──────────────────────────────────────────────────────────────
_robot_pos    = None   # {"x": float, "y": float}
_other_robots = None   # {"robots": [...]} from prediction node
_ball         = None   # {"global_pos": {x,y}, "ball_lost": bool, ...}
_ally_id      = None   # ID of allied robot (team 0); all others are team 1


def all_robots():
    """All tracked other robots (detections and predictions).

    Returns a list of {"id": int, "x": float, "y": float,
                        "vx": float|None, "vy": float|None,
                        "predicted": bool, "team": int}.
    The ally robot (same team as us) has team=TEAM_US; all others TEAM_ENEMY.
    """
    if _other_robots is None:
        return []
    out = []
    for r in _other_robots.get("robots", []):
        x, y = r.get("x"), r.get("y")
        if x is None or y is None:
            continue
        rid = r.get("id")
        out.append({
            "id":        rid,
            "x":         float(x),
            "y":         float(y),
            "vx":        r.get("vx"),
            "vy":        r.get("vy"),
            "predicted": r.get("method") == "predicted",
            "team":      TEAM_US if rid == _ally_id else TEAM_ENEMY,
        })
    return out


def ball_pos():
    if _ball is None:
        return None
    gpos = _ball.get("global_pos")
    if gpos is None:
        return None
    return {
        "x":    float(gpos["x"]),
        "y":    float(gpos["y"]),
        "lost": bool(_ball.get("ball_lost", False)),
    }


# ── Ball control ──────────────────────────────────────────────────────────────

# The team that currently has the ball (TEAM_US, TEAM_ENEMY, or None).
# Updated by on_ball() on every call.
controlling_team = None

# Per-robot dwell tracking: robot_key (id or None for self) → monotonic time
# when the ball first entered BALL_CONTROL_DIST for that robot continuously.
_control_first_seen = {}


def on_ball(robots):
    """Return the robot closest to the ball that is within BALL_CONTROL_DIST
    and has kept the ball in range for at least BALL_CONTROL_DWELL seconds.

    Also updates the module-level `controlling_team` variable.

    If `robot_id` is given, return that robot's entry only if it is in
    control (useful as a boolean: ``if on_ball(my_id): ...``).

    Returns a dict {"id": int|None, "x", "y", "predicted", "team", "dist"}
    where id=None represents the own robot.  Returns None when no robot
    has confirmed control.
    """
    global controlling_team, _control_first_seen

    bp = ball_pos()
    if bp is None:
        controlling_team = None
        _control_first_seen.clear()
        return None

    now = time.monotonic()

    # Build list of all robots currently within range
    in_range = []

    sp = self_pos()
    if sp is not None:
        dx = sp["x"] - bp["x"]
        dy = sp["y"] - bp["y"]
        d2 = dx*dx + dy*dy
        if d2 <= BALL_CONTROL_DIST_SQ:
            in_range.append({"id": None, "x": sp["x"], "y": sp["y"],
                             "predicted": False, "team": TEAM_US,
                             "dist": math.sqrt(d2)})

    for r in robots:
        dx = r["x"] - bp["x"]
        dy = r["y"] - bp["y"]
        d2 = dx*dx + dy*dy
        if d2 <= BALL_CONTROL_DIST_SQ:
            in_range.append({**r, "dist": math.sqrt(d2)})

    keys = {c["id"] for c in in_range}
    _control_first_seen = {k: v for k, v in _control_first_seen.items() if k in keys}

    for c in in_range:
        if c["id"] not in _control_first_seen:
            _control_first_seen[c["id"]] = now

    # Only robots that have dwelled long enough qualify
    eligible = [c for c in in_range
                if now - _control_first_seen[c["id"]] >= BALL_CONTROL_DWELL]

    if not eligible:
        controlling_team = None
        return None

    closest = min(eligible, key=lambda c: c["dist"])
    controlling_team = closest["team"]
    return closest


# ── Strategy helpers ──────────────────────────────────────────────────────────

_SHOOT_GRID_N = 15
_MAX_RANGE = 0.5

_OUR_GOAL   = (FIELD_WIDTH / 2, 0.0)
_ENEMY_GOAL = (FIELD_WIDTH / 2, FIELD_HEIGHT)


def _find_shooting_position(rx, ry, goal_x, goal_y, robots):
    """Return the field position closest to us that has a clear
    line of sight to the enemy goal (no robot within ROBOT_RADIUS of the path).

    Searches a _SHOOT_GRID_N × _SHOOT_GRID_N grid sorted by distance to us,
    returning the first unblocked cell which has clear sight to the goal and ally.  Returns None if every
    cell is blocked.
    """
    robots_np = np.array([[r["x"], r["y"]] for r in robots]) if robots else np.empty((0,2))

    n = _SHOOT_GRID_N
    xs = (np.arange(n) + 0.5) / n * FIELD_WIDTH
    ys = (np.arange(n) + 0.5) / n * FIELD_HEIGHT
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T

    grid = grid[np.argsort(np.linalg.norm(grid - np.array([rx, ry]), axis=1))]

    for px, py in grid:
        if _dist(px, py, goal_x, goal_y) <= _MAX_RANGE:
            if len(robots_np) == 0:
                return px, py
            dists = _dist_to_segment_np(px, py, goal_x, goal_y, robots_np)
            if np.all(dists >= ROBOT_RADIUS):
                return px, py
    return None


def _find_passing_position(rx, ry, ally_x, ally_y, goal_x, goal_y, robots):
    """Return the field position closest to us that has a clear
    line of sight to the enemy goal (no robot within ROBOT_RADIUS of the path).

    Searches a _SHOOT_GRID_N × _SHOOT_GRID_N grid sorted by distance to us,
    returning the first unblocked cell which has clear sight to the goal and ally.  Returns None if every
    cell is blocked.
    """
    robots_np = np.array([[r["x"], r["y"]] for r in robots]) if robots else np.empty((0,2))

    n = _SHOOT_GRID_N
    xs = (np.arange(n) + 0.5) / n * FIELD_WIDTH
    ys = (np.arange(n) + 0.5) / n * FIELD_HEIGHT
    grid = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T

    grid = grid[np.argsort(np.linalg.norm(grid - np.array([rx, ry]), axis=1))]

    for px, py in grid:
        if (_dist(px, py, goal_x, goal_y) <= _MAX_RANGE and
            _dist(px, py, ally_x, ally_y) <= _MAX_RANGE):

            if len(robots_np) == 0:
                return px, py

            d_goal = _dist_to_segment_np(px, py, goal_x, goal_y, robots_np)
            d_ally = _dist_to_segment_np(px, py, ally_x, ally_y, robots_np)

            if not any(dg < ROBOT_RADIUS and da < ROBOT_RADIUS for dg, da in zip(d_goal, d_ally)):
                return px, py
    return None


# ── ORIGINAL STRATEGY (UNCHANGED) ─────────────────────────────────────────────

# ⚠️ IMPORTANT:
# This is EXACTLY your original logic, only function calls updated

def _compute_strategy_points(ctrl, robots):
    """Return the robot_strategy_points list for the current game state.

    No team has the ball
    ───────────────────
    • If we are closer to the ball than the ally: get the ball.
    • Otherwise the ally gets the ball; we either go for a passing position or block enemy shoot lane.

    Our team has the ball
    ─────────────────────
    • We control it  → single point at the enemy goal (shoot).
    • Ally controls  → find the field position closest to the enemy goal with
                       a clear line of sight to it; single point there.

    Enemy has the ball
    ──────────────────
    • If we are closer to the line controller → our goal than the ally: block
      that shot lane.
    • Otherwise the ally covers the goal; we cover the pass to the closest
      other enemy.

    Returns a list of {"x", "y"} dicts (0, 1, or 2 entries).
    """
    bp = ball_pos()
    sp = self_pos()
    ally = next((r for r in robots if r["team"] == TEAM_US), None)
    enemies = [r for r in robots if r["team"] == TEAM_ENEMY]

    if ctrl is None:
        if bp is None or sp is None:
            return []

        d_self = _dist(sp["x"], sp["y"], bp["x"], bp["y"])
        d_ally = (_dist(ally["x"], ally["y"], bp["x"], bp["y"])
                if ally else float("inf"))

        if d_self <= d_ally:
            return [{"x": round(bp["x"], 3), "y": round(bp["y"], 3)}]

        else:
            if bp["y"] < FIELD_HEIGHT / 2:
                if not enemies:
                    return []

                closest_enemy = min(
                    enemies,
                    key=lambda e: _dist(e["x"], e["y"], bp["x"], bp["y"])
                )

                gx, gy = _OUR_GOAL
                ix, iy = _closest_on_segment(
                    closest_enemy["x"], closest_enemy["y"],
                    gx, gy,
                    sp["x"], sp["y"]
                )

                return [{"x": round(ix, 3), "y": round(iy, 3)}]

            else:
                pos = _find_passing_position(
                    sp["x"], sp["y"],
                    ally["x"], ally["y"],
                    _ENEMY_GOAL[0], _ENEMY_GOAL[1],
                    robots
                )
                if pos:
                    return [{"x": round(pos[0], 3), "y": round(pos[1], 3)}]
                return []

    if ctrl.get("team") == TEAM_US:
        if ctrl.get("id") is None:
            pos = _find_shooting_position(
                sp["x"], sp["y"],
                _ENEMY_GOAL[0], _ENEMY_GOAL[1],
                robots
            )
            if pos is None:
                return []
            return [{"x": round(pos[0], 3), "y": round(pos[1], 3)}]
        else:
            pos = _find_passing_position(
                sp["x"], sp["y"],
                ally["x"], ally["y"],
                _ENEMY_GOAL[0], _ENEMY_GOAL[1],
                robots
            )
            if pos is None:
                return []
            return [{"x": round(pos[0], 3), "y": round(pos[1], 3)}]

    if ctrl.get("team") == TEAM_ENEMY:
        crx, cry = ctrl["x"], ctrl["y"]
        gx, gy   = _OUR_GOAL

        d_self = _dist(sp["x"], sp["y"], crx, cry)
        d_ally = (_dist(ally["x"], ally["y"], crx, cry)
                if ally else float("inf"))

        if d_self <= d_ally:
            ix, iy = _closest_on_segment(crx, cry, gx, gy, sp["x"], sp["y"])
        else:
            others = [r for r in enemies if r["id"] != ctrl["id"]]
            if not others:
                ix, iy = _closest_on_segment(crx, cry, gx, gy, sp["x"], sp["y"])
            else:
                target = min(others, key=lambda r: _dist(r["x"], r["y"], crx, cry))
                ix, iy = _closest_on_segment(crx, cry, target["x"], target["y"], sp["x"], sp["y"])

        return [{"x": round(ix, 3), "y": round(iy, 3)}]

    return []


# ── Publish ───────────────────────────────────────────────────────────────────

def _publish(now):
    robots = all_robots()

    state = {"t": round(now, 3)}

    with _perf.measure("positions"):
        p = self_pos()
        if p is not None:
            state["self"] = p

        state["robots"] = [
            {"id": r["id"], "x": r["x"], "y": r["y"],
             "predicted": r["predicted"], "team": r["team"]}
            for r in robots
        ]

        bp = ball_pos()
        if bp is not None:
            state["ball"] = bp

    with _perf.measure("ball_control"):
        ctrl = on_ball(robots)
        state["ball_control"] = (
            {"id": ctrl["id"], "team": ctrl["team"], "dist": round(ctrl["dist"], 3)}
            if ctrl is not None else None
        )
        state["controlling_team"] = controlling_team

    with _perf.measure("strategy"):
        strategy_points = _compute_strategy_points(ctrl, robots)
        mb.set("robot_strategy_points", json.dumps(strategy_points))

    mb.set("game_state", json.dumps(state))


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _robot_pos, _other_robots, _ball, _ally_id

    if value is None:
        return

    try:
        if key == "robot_position":
            _robot_pos = json.loads(value)
        elif key == "other_robots":
            _other_robots = json.loads(value)
        elif key == "ball":
            _ball = json.loads(value)
        elif key == "ally_id":
            _ally_id = int(value) if value else None
    except Exception:
        return

    _publish(time.monotonic())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key in ("robot_position", "other_robots", "ball", "ally_id"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["robot_position", "other_robots", "ball", "ally_id"], on_update)

    print("[MASTER] Optimized master node running...")

    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        pass
    finally:
        mb.close()