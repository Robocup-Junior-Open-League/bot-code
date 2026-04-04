import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import time
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor

# ── Field dimensions ──────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.58   # metres — playing field only
FIELD_HEIGHT = 2.19

# ── Teams ─────────────────────────────────────────────────────────────────────
# Team 0 = our team  — bottom goal  (y = -OUTER_MARGIN)
# Team 1 = enemy     — top    goal  (y = FIELD_HEIGHT + OUTER_MARGIN)
TEAM_US    = 0
TEAM_ENEMY = 1

# ── Subdivision grid ──────────────────────────────────────────────────────────
# The field is split into COLS × ROWS = 4 × 4 = 16 equal subdivisions.
#
#   row 3 │ (0,3) (1,3) (2,3) (3,3) │  ← team 0 attack / team 1 own half
#   row 2 │ (0,2) (1,2) (2,2) (3,2) │  ← team 0 push
#   row 1 │ (0,1) (1,1) (2,1) (3,1) │  ← team 1 push
#   row 0 │ (0,0) (1,0) (2,0) (3,0) │  ← team 1 attack / team 0 own half
#           col 0  col 1  col 2  col 3
#
# col  (0–3) : vertical strip, increases left → right
# row  (0–3) : horizontal strip, increases bottom → top
# rank       : synonym for row  — "which horizontal band"
# file       : synonym for col  — "which file band"
#
# ── Game-state zones ──────────────────────────────────────────────────────────
# Team 1 advances downward toward team 0's goal:
#   row 1 → push    row 0 → attack
# Team 0 advances upward toward team 1's goal:
#   row 2 → push    row 3 → attack
#
# strength: "weak" = 1 robot in zone, "strong" = 2+ robots in zone
# side (ball's horizontal position): col 0 → "left", col 1–2 → "center", col 3 → "right"

COLS = 4
ROWS = 4

_COL_W = FIELD_WIDTH  / COLS   # 0.395 m per column
_ROW_H = FIELD_HEIGHT / ROWS   # 0.5475 m per row

_PUSH_ROW   = {TEAM_US: 2, TEAM_ENEMY: 1}
_ATTACK_ROW = {TEAM_US: 3, TEAM_ENEMY: 0}

# ── Ball control ──────────────────────────────────────────────────────────────
# A robot is considered "on ball control" when its centre is within this
# distance of the ball centre.  Value accounts for robot radius (0.09 m),
# ball radius (0.021 m), and a small detection-noise buffer.
ROBOT_RADIUS      = 0.09
BALL_RADIUS       = 0.021
BALL_CONTROL_DIST = ROBOT_RADIUS + BALL_RADIUS + 0.04   # ≈ 0.15 m

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_master", broker=mb, print_every=100)

# ── Broker state ──────────────────────────────────────────────────────────────
_robot_pos    = None   # {"x": float, "y": float}
_other_robots = None   # {"robots": [...]} from prediction node
_ball         = None   # {"global_pos": {x,y}, "ball_lost": bool, ...}
_ally_id      = None   # ID of allied robot (team 0); all others are team 1


# ── Low-level subdivision math ────────────────────────────────────────────────

def col_of(x):
    """file column index (0 = leftmost) for a global x coordinate."""
    return max(0, min(COLS - 1, int(x / _COL_W)))


def row_of(y):
    """Horizontal row index (0 = bottommost) for a global y coordinate."""
    return max(0, min(ROWS - 1, int(y / _ROW_H)))


def subdiv_of(x, y):
    """(col, row) subdivision indices for a global position."""
    return col_of(x), row_of(y)


# ── Position accessors ────────────────────────────────────────────────────────

def self_pos():
    """Own robot position, or None if unknown.

    Returns {"x": float, "y": float}.
    """
    if _robot_pos is None:
        return None
    return {"x": float(_robot_pos["x"]), "y": float(_robot_pos["y"])}


def all_robots():
    """All tracked other robots (detections and predictions).

    Returns a list of {"id": int, "x": float, "y": float,
                        "predicted": bool, "team": int}.
    The ally robot (same team as us) has team=0; all others have team=1.
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
            "predicted": r.get("method") == "predicted",
            "team":      TEAM_US if rid == _ally_id else TEAM_ENEMY,
        })
    return out


def robot_by_id(robot_id):
    """Position of a specific tracked robot, or None.

    Returns {"id": int, "x": float, "y": float, "predicted": bool, "team": int}.
    """
    for r in all_robots():
        if r["id"] == robot_id:
            return r
    return None


def ball_pos():
    """Ball position, or None if unavailable.

    Returns {"x": float, "y": float, "lost": bool}.
    """
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


# ── Ball control ─────────────────────────────────────────────────────────────

def _dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def on_ball(robot_id=None):
    """
    Return the robot currently closest to the ball and within BALL_CONTROL_DIST,
    or None if no robot is in control.

    If `robot_id` is given, return that robot's entry only if it is in control
    (useful as a boolean check: ``if on_ball(my_id): ...``).

    The returned dict is {"id": int|None, "x": float, "y": float,
                           "predicted": bool, "team": int, "dist": float},
    where id=None and team=TEAM_US represent the own robot.
    """
    bp = ball_pos()
    if bp is None:
        return None

    candidates = []

    # Own robot
    sp = self_pos()
    if sp is not None:
        d = _dist(sp["x"], sp["y"], bp["x"], bp["y"])
        if d <= BALL_CONTROL_DIST:
            candidates.append({"id": None, "x": sp["x"], "y": sp["y"],
                                "predicted": False, "team": TEAM_US, "dist": d})

    # Tracked robots
    for r in all_robots():
        d = _dist(r["x"], r["y"], bp["x"], bp["y"])
        if d <= BALL_CONTROL_DIST:
            candidates.append({**r, "dist": d})

    if not candidates:
        return None

    closest = min(candidates, key=lambda c: c["dist"])

    if robot_id is not None:
        return closest if closest["id"] == robot_id else None

    return closest


def self_on_ball():
    """True if the own robot is the closest robot in ball-control range."""
    r = on_ball()
    return r is not None and r["id"] is None


def ball_controlled():
    """True if any robot (any team) is currently in ball-control range."""
    return on_ball() is not None


# ── Generic location predicates ───────────────────────────────────────────────

def in_subdivision(x, y, col, row):
    """True if (x, y) falls in subdivision (col, row)."""
    return col_of(x) == col and row_of(y) == row


def in_rank(x, y, rank):
    """True if (x, y) is in horizontal row `rank` (0 = bottom, 3 = top)."""
    return row_of(y) == rank


def in_file(x, y, file):
    """True if (x, y) is in file column `file` (0 = left, 3 = right)."""
    return col_of(x) == file


# ── Own-robot location helpers ────────────────────────────────────────────────

def self_in_subdivision(col, row):
    p = self_pos()
    return p is not None and in_subdivision(p["x"], p["y"], col, row)


def self_in_rank(rank):
    p = self_pos()
    return p is not None and in_rank(p["x"], p["y"], rank)


def self_in_file(file):
    p = self_pos()
    return p is not None and in_file(p["x"], p["y"], file)


# ── Ball location helpers ─────────────────────────────────────────────────────

def ball_in_subdivision(col, row):
    p = ball_pos()
    return p is not None and in_subdivision(p["x"], p["y"], col, row)


def ball_in_rank(rank):
    p = ball_pos()
    return p is not None and in_rank(p["x"], p["y"], rank)


def ball_in_file(file):
    p = ball_pos()
    return p is not None and in_file(p["x"], p["y"], file)


# ── Per-robot location helpers ────────────────────────────────────────────────

def robot_in_subdivision(robot_id, col, row):
    r = robot_by_id(robot_id)
    return r is not None and in_subdivision(r["x"], r["y"], col, row)


def robot_in_rank(robot_id, rank):
    r = robot_by_id(robot_id)
    return r is not None and in_rank(r["x"], r["y"], rank)


def robot_in_file(robot_id, file):
    r = robot_by_id(robot_id)
    return r is not None and in_file(r["x"], r["y"], file)


# ── Game state ────────────────────────────────────────────────────────────────

def _ball_side():
    """Horizontal side of the ball: 'left', 'center', or 'right'. None if unknown."""
    bp = ball_pos()
    if bp is None:
        return None
    c = col_of(bp["x"])
    if c == 0:
        return "left"
    if c == 3:
        return "right"
    return "center"


def _team_positions(team):
    """All known positions for `team` (including own robot for team 0)."""
    positions = []
    if team == TEAM_US:
        sp = self_pos()
        if sp:
            positions.append(sp)
    for r in all_robots():
        if r["team"] == team:
            positions.append({"x": r["x"], "y": r["y"]})
    return positions


def _team_game_state(team):
    """
    Compute the offensive game-state for a single team.

    Returns {"state": "push"|"attack", "strength": "weak"|"strong"} or None
    if no robots of this team are in an offensive row.

    attack takes priority over push when robots are in both zones.
    """
    positions = _team_positions(team)
    attack_row = _ATTACK_ROW[team]
    push_row   = _PUSH_ROW[team]

    in_attack = [p for p in positions if row_of(p["y"]) == attack_row]
    in_push   = [p for p in positions if row_of(p["y"]) == push_row]

    if in_attack:
        return {"state": "attack", "strength": "strong" if len(in_attack) >= 2 else "weak"}
    if in_push:
        return {"state": "push",   "strength": "strong" if len(in_push)   >= 2 else "weak"}
    return None


def _state_score(s):
    if s is None:
        return 0
    return (2 if s["state"] == "attack" else 1) * (2 if s["strength"] == "strong" else 1)


def game_state():
    """
    Return the dominant game state across both teams.

    Returns {
        "state":    "push" | "attack" | None,
        "strength": "weak" | "strong" | None,
        "team":     0 | 1 | None,
        "side":     "left" | "center" | "right" | None,
        "team0":    {"state": ..., "strength": ...} | None,
        "team1":    {"state": ..., "strength": ...} | None,
    }

    The top-level state/strength/team reflect whichever team has the more
    dangerous situation (attack > push, strong > weak).  team0 and team1
    contain the full per-team detail.
    """
    s0 = _team_game_state(TEAM_US)
    s1 = _team_game_state(TEAM_ENEMY)
    sc0, sc1 = _state_score(s0), _state_score(s1)

    if sc0 == 0 and sc1 == 0:
        dominant_state    = None
        dominant_strength = None
        dominant_team     = None
    elif sc0 >= sc1:
        dominant_state    = s0["state"]
        dominant_strength = s0["strength"]
        dominant_team     = TEAM_US
    else:
        dominant_state    = s1["state"]
        dominant_strength = s1["strength"]
        dominant_team     = TEAM_ENEMY

    return {
        "state":    dominant_state,
        "strength": dominant_strength,
        "team":     dominant_team,
        "side":     _ball_side(),
        "team0":    s0,
        "team1":    s1,
    }


# ── Game-state convenience predicates ────────────────────────────────────────

def is_push(team=None):
    """True if the dominant state is 'push', optionally filtered by team."""
    gs = game_state()
    return gs["state"] == "push" and (team is None or gs["team"] == team)


def is_attack(team=None):
    """True if the dominant state is 'attack', optionally filtered by team."""
    gs = game_state()
    return gs["state"] == "attack" and (team is None or gs["team"] == team)


def is_strong(team=None):
    """True if the dominant state (optionally for a specific team) is 'strong'."""
    if team is None:
        return game_state()["strength"] == "strong"
    ts = _team_game_state(team)
    return ts is not None and ts["strength"] == "strong"


def game_side():
    """Horizontal side of the ball: 'left', 'center', 'right', or None."""
    return _ball_side()


# ── Broker publish ────────────────────────────────────────────────────────────

def _publish(now):
    state = {"t": round(now, 3)}

    with _perf.measure("sectors"):
        p = self_pos()
        if p is not None:
            c, r = subdiv_of(p["x"], p["y"])
            state["self"] = {"col": c, "row": r}

        robots_out = []
        for rb in all_robots():
            c, r = subdiv_of(rb["x"], rb["y"])
            robots_out.append({"id": rb["id"], "col": c, "row": r,
                                "predicted": rb["predicted"], "team": rb["team"]})
        state["robots"] = robots_out

        bp = ball_pos()
        if bp is not None:
            c, r = subdiv_of(bp["x"], bp["y"])
            state["ball"] = {"col": c, "row": r, "lost": bp["lost"]}

    with _perf.measure("game_state"):
        state["game_state"] = game_state()

    with _perf.measure("ball_control"):
        ctrl = on_ball()
        state["ball_control"] = (
            {"id": ctrl["id"], "team": ctrl["team"], "dist": round(ctrl["dist"], 3)}
            if ctrl is not None else None
        )

    mb.set("field_sectors", json.dumps(state))


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
            _ally_id = int(value) if value and value.strip() else None
        else:
            return
    except (json.JSONDecodeError, TypeError, ValueError):
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
    print("[MASTER] Starting master node...")
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[MASTER] Stopped.")
        mb.close()
