from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from collections import deque
import json
import math

# ── Field & detection configuration ──────────────────────────────────────────
FIELD_WIDTH      = 1.0    # metres, X axis
FIELD_HEIGHT     = 2.0    # metres, Y axis
ROBOT_RADIUS     = 0.1    # metres — assumed radius of all robots

# A lidar hit is classified as a wall hit if it falls within this distance of
# any field boundary.  Slightly smaller than ROBOT_RADIUS so that robots near
# walls can still be detected.
WALL_MARGIN      = 0.08   # metres

# Two interior points belong to the same robot if they are within this distance.
CLUSTER_DIST     = 0.20   # metres

# Clusters smaller than this are discarded as noise.
MIN_CLUSTER_SIZE = 2

# Attempt circle fitting only for clusters at least this large.
MIN_CIRCLE_PTS   = 3

# Iterations of the centre-convergence loop.
CIRCLE_FIT_ITERS = 8

# If the RMS deviation from ROBOT_RADIUS exceeds this, fall back to centroid.
MAX_FIT_ERROR    = 0.06   # metres
# ─────────────────────────────────────────────────────────────────────────────

mb = TelemetryBroker()

_lidar       = {}     # {angle_deg (int): dist_mm (int)}
_robot_pos   = None   # (x, y) metres, in field frame
_field_angle  = None  # degrees — explicitly set; None = not set
_sim_heading  = None  # degrees — fallback from simulation


def _effective_field_angle():
    if _field_angle is not None:
        return _field_angle
    if _sim_heading is not None:
        return _sim_heading
    return 0.0


# ── Coordinate conversion ─────────────────────────────────────────────────────

def _lidar_to_field():
    """Convert all lidar measurements to (x, y) in field coordinates."""
    if not _lidar or _robot_pos is None:
        return []
    rx, ry = _robot_pos
    fa_rad = math.radians(_effective_field_angle())
    pts = []
    for angle_deg, dist_mm in _lidar.items():
        d = dist_mm / 1000.0
        a = math.radians(angle_deg) + fa_rad
        pts.append((rx + d * math.cos(a), ry + d * math.sin(a)))
    return pts


def _is_wall_hit(x, y):
    """True if the point is close enough to a field boundary to be a wall hit."""
    return (
        x <= WALL_MARGIN or x >= FIELD_WIDTH  - WALL_MARGIN or
        y <= WALL_MARGIN or y >= FIELD_HEIGHT - WALL_MARGIN
    )


def _inside_field(x, y):
    return 0 <= x <= FIELD_WIDTH and 0 <= y <= FIELD_HEIGHT


# ── Clustering ────────────────────────────────────────────────────────────────

def _cluster(points):
    """
    Greedy flood-fill: each unvisited point seeds a new cluster; all reachable
    points within CLUSTER_DIST (transitively) join it.  Returns a list of
    point-lists.
    """
    n = len(points)
    assigned = [False] * n
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        queue = deque([i])
        while queue:
            curr = queue.popleft()
            cx, cy = points[curr]
            for j in range(n):
                if not assigned[j]:
                    jx, jy = points[j]
                    if math.hypot(cx - jx, cy - jy) <= CLUSTER_DIST:
                        assigned[j] = True
                        cluster.append(j)
                        queue.append(j)
        clusters.append([points[idx] for idx in cluster])

    return clusters


# ── Circle fitting ────────────────────────────────────────────────────────────

def _fit_center(pts):
    """
    Estimate the centre of the robot given a cluster of lidar hit points.

    For clusters of MIN_CIRCLE_PTS or more:
      Iterative convergence — each point P estimates the centre as
          C_i = P - R * normalise(P - O_prev)
      The new estimate is the mean of all C_i.  Converges to the
      least-squares circle centre for a circle of known radius ROBOT_RADIUS.
      Falls back to centroid if the RMS fit error exceeds MAX_FIT_ERROR.

    For smaller clusters: centroid of the points.

    Returns (x, y, method_str).
    """
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    if n < MIN_CIRCLE_PTS:
        return cx, cy, "centroid"

    # Iterative circle-centre convergence
    ox, oy = cx, cy
    for _ in range(CIRCLE_FIT_ITERS):
        sx, sy = 0.0, 0.0
        for px, py in pts:
            dx, dy = px - ox, py - oy
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                sx += px
                sy += py
            else:
                scale = ROBOT_RADIUS / dist
                sx += px - dx * scale   # = P - R * normalise(P - O)
                sy += py - dy * scale
        ox, oy = sx / n, sy / n

    # Validate: RMS of |P - O| - R should be small
    rms = math.sqrt(
        sum((math.hypot(px - ox, py - oy) - ROBOT_RADIUS) ** 2 for px, py in pts) / n
    )
    if rms > MAX_FIT_ERROR:
        return cx, cy, f"centroid (fit rms={rms:.3f})"

    return ox, oy, f"circle (rms={rms:.3f})"


# ── Main detection ────────────────────────────────────────────────────────────

def _detect_robots():
    if not _lidar or _robot_pos is None:
        return []

    all_pts = _lidar_to_field()

    interior = [
        (x, y) for x, y in all_pts
        if _inside_field(x, y) and not _is_wall_hit(x, y)
    ]

    if not interior:
        return []

    clusters = [c for c in _cluster(interior) if len(c) >= MIN_CLUSTER_SIZE]

    robots = []
    for cluster in clusters:
        ox, oy, method = _fit_center(cluster)
        print(f"  [ROBOTS] {len(cluster):2d} pts → ({ox:.3f}, {oy:.3f})  [{method}]")
        robots.append({"x": round(ox, 3), "y": round(oy, 3), "pts": len(cluster)})

    return robots


# ── Broker interface ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _lidar, _robot_pos, _field_angle, _sim_heading

    if value is None:
        return

    if key == "lidar":
        try:
            raw = json.loads(value)
            _lidar = {int(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError):
            return

    elif key == "robot_position":
        try:
            pos = json.loads(value)
            _robot_pos = (float(pos["x"]), float(pos["y"]))
        except Exception:
            return

    elif key == "field_angle":
        try:
            _field_angle = float(value)
        except (ValueError, TypeError):
            return

    elif key == "sim_heading":
        try:
            _sim_heading = float(value)
        except (ValueError, TypeError):
            return

    if key in ("lidar", "robot_position", "field_angle", "sim_heading"):
        robots = _detect_robots()
        mb.set("other_robots", json.dumps(robots))


if __name__ == "__main__":
    for attr, broker_key in [("_field_angle", "field_angle"),
                              ("_sim_heading", "sim_heading")]:
        try:
            val = mb.get(broker_key)
            if val is not None:
                globals()[attr] = float(val)
        except Exception:
            pass
    try:
        raw = mb.get("robot_position")
        if raw:
            pos = json.loads(raw)
            _robot_pos = (float(pos["x"]), float(pos["y"]))
    except Exception:
        pass

    mb.setcallback(
        ["lidar", "robot_position", "field_angle", "sim_heading"],
        on_update,
    )
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping robot detection.")
        mb.close()
