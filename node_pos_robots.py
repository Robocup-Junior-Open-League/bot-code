from robus_core.libs.lib_telemtrybroker import TelemetryBroker
import json
import math
import time
import numpy as np

# ── Field & detection configuration ──────────────────────────────────────────
FIELD_WIDTH  = 1.82   # metres, X axis
FIELD_HEIGHT = 2.43   # metres, Y axis
ROBOT_RADIUS = 0.09   # metres — assumed radius of all robots
ROBOT_DIAMETER = ROBOT_RADIUS * 2

# Cluster centres within this distance of any field boundary are rejected.
WALL_MARGIN = 0.08   # metres

# Max Cartesian gap between consecutive (angle-sorted) points in a cluster.
CLUSTER_THRESHOLD = 0.08   # metres

# Clusters smaller than this are discarded as noise.
MIN_CLUSTER_POINTS = 3

# Detected diameter may differ from ROBOT_DIAMETER by at most this much.
SIZE_TOLERANCE = 0.05   # metres

# ── Confidence & tracking ─────────────────────────────────────────────────────
MAX_ROBOTS      = 3

# Two detections whose centres are closer than this are considered overlapping.
OVERLAP_DIST    = ROBOT_RADIUS * 2   # metres

VEL_MIN_DT      = 0.05   # seconds — min elapsed time between history samples
VEL_HISTORY_N   = 10     # rolling history length per tracked robot
VEL_HISTORY_MIN = 3      # minimum samples before fitted velocity is trusted
MAX_ROBOT_SPEED = 2.0    # m/s — hard cap after fitting

# ─────────────────────────────────────────────────────────────────────────────

mb = TelemetryBroker()

_lidar     = {}   # {angle_deg (int): dist_mm (int)}
_robot_pos = None # (x, y) metres, in field frame
_imu_pitch = None # degrees — from imu_pitch broker key

# ── Tracking state ────────────────────────────────────────────────────────────
_tracked = {}   # id → {"x","y","vx","vy","t","lost","history"}
_next_id = 1


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


# ── Detection ─────────────────────────────────────────────────────────────────

def _lidar_points(angles, distances, lidar_pos):
    """Convert polar scan to absolute field-frame Cartesian numpy array."""
    x = lidar_pos[0] + distances * np.cos(angles)
    y = lidar_pos[1] + distances * np.sin(angles)
    return np.column_stack((x, y))


def _detect_clusters(points, threshold=CLUSTER_THRESHOLD):
    """
    Group angle-sorted Cartesian points into clusters by consecutive distance.
    Returns a list of (N, 2) numpy arrays.
    """
    if len(points) == 0:
        return []
    clusters = []
    current  = [points[0]]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[i - 1]) < threshold:
            current.append(points[i])
        else:
            clusters.append(np.array(current))
            current = [points[i]]
    clusters.append(np.array(current))
    return clusters


def _is_near_wall(center):
    x, y = center
    return (
        x < WALL_MARGIN or x > FIELD_WIDTH  - WALL_MARGIN or
        y < WALL_MARGIN or y > FIELD_HEIGHT - WALL_MARGIN
    )


# ── Physics-aware position prediction ────────────────────────────────────────

_MAX_PRED_STEPS = 20
_MAX_PRED_DT    = 2.0


def _predict_with_bounce(x, y, vx, vy, dt):
    dt   = min(dt, _MAX_PRED_DT)
    n    = max(1, min(int(dt / 0.02) + 1, _MAX_PRED_STEPS))
    step = dt / n
    for _ in range(n):
        x += vx * step
        y += vy * step
        if x < ROBOT_RADIUS:
            x = ROBOT_RADIUS;  vx = abs(vx)
        elif x > FIELD_WIDTH - ROBOT_RADIUS:
            x = FIELD_WIDTH - ROBOT_RADIUS;  vx = -abs(vx)
        if y < ROBOT_RADIUS:
            y = ROBOT_RADIUS;  vy = abs(vy)
        elif y > FIELD_HEIGHT - ROBOT_RADIUS:
            y = FIELD_HEIGHT - ROBOT_RADIUS;  vy = -abs(vy)
    return x, y


# ── Overlap filtering ─────────────────────────────────────────────────────────

def _filter_overlapping(robots):
    kept = []
    for r in robots:
        if not any(math.hypot(r["x"] - k["x"], r["y"] - k["y"]) < OVERLAP_DIST
                   for k in kept):
            kept.append(r)
    return kept


# ── Velocity fitting ──────────────────────────────────────────────────────────

def _fit_velocity(history):
    n = len(history)
    if n < 2:
        return 0.0, 0.0
    t0     = history[0][0]
    ts     = [h[0] - t0 for h in history]
    xs     = [h[1]       for h in history]
    ys     = [h[2]       for h in history]
    sum_t  = sum(ts)
    sum_t2 = sum(t * t for t in ts)
    denom  = n * sum_t2 - sum_t ** 2
    if abs(denom) < 1e-9:
        return 0.0, 0.0
    sum_tx = sum(t * x for t, x in zip(ts, xs))
    sum_ty = sum(t * y for t, y in zip(ts, ys))
    vx = (n * sum_tx - sum_t * sum(xs)) / denom
    vy = (n * sum_ty - sum_t * sum(ys)) / denom
    return vx, vy


# ── ID tracking ───────────────────────────────────────────────────────────────

def _match_and_track(detections, now):
    global _tracked, _next_id

    predictions = {}
    for tid, tr in _tracked.items():
        dt = now - tr["t"]
        predictions[tid] = _predict_with_bounce(
            tr["x"], tr["y"], tr["vx"], tr["vy"], dt,
        )

    matched_det   = [None] * len(detections)
    matched_track = set()

    pairs = sorted(
        (math.hypot(det["x"] - px, det["y"] - py), di, tid)
        for di, det in enumerate(detections)
        for tid, (px, py) in predictions.items()
    )
    for _, di, tid in pairs:
        if matched_det[di] is None and tid not in matched_track:
            matched_det[di] = tid
            matched_track.add(tid)

    for di, det in enumerate(detections):
        if matched_det[di] is not None:
            continue
        remaining = [(tid, px, py) for tid, (px, py) in predictions.items()
                     if tid not in matched_track]
        if remaining:
            best_tid = min(remaining,
                           key=lambda t: math.hypot(det["x"] - t[1], det["y"] - t[2]))[0]
            matched_det[di] = best_tid
            matched_track.add(best_tid)
        elif len(_tracked) < MAX_ROBOTS:
            pass

    new_tracked = {}

    for di, det in enumerate(detections):
        tid = matched_det[di]

        if tid is not None:
            old     = _tracked[tid]
            dt      = now - old["t"]
            history = old.get("history", [])

            if dt >= VEL_MIN_DT:
                history = history + [(now, det["x"], det["y"])]
                history = history[-VEL_HISTORY_N:]

            if len(history) >= VEL_HISTORY_MIN:
                new_vx, new_vy = _fit_velocity(history)
            else:
                new_vx, new_vy = old["vx"], old["vy"]

            spd = math.hypot(new_vx, new_vy)
            if spd > MAX_ROBOT_SPEED:
                new_vx *= MAX_ROBOT_SPEED / spd
                new_vy *= MAX_ROBOT_SPEED / spd

            new_tracked[tid] = {
                "x": det["x"], "y": det["y"], "t": now,
                "vx": new_vx, "vy": new_vy,
                "lost": 0, "history": history,
            }
        else:
            tid = _next_id
            _next_id += 1
            new_tracked[tid] = {
                "x": det["x"], "y": det["y"], "t": now,
                "vx": 0.0, "vy": 0.0,
                "lost": 0, "history": [(now, det["x"], det["y"])],
            }

        det["id"]  = tid
        det["vx"]  = round(new_tracked[tid]["vx"], 3)
        det["vy"]  = round(new_tracked[tid]["vy"], 3)

    result = list(detections)

    for tid, tr in _tracked.items():
        if tid in matched_track:
            continue
        tr["lost"] += 1
        new_tracked[tid] = tr
        dt = now - tr["t"]
        px, py = _predict_with_bounce(tr["x"], tr["y"], tr["vx"], tr["vy"], dt)
        result.append({
            "x": round(px, 3), "y": round(py, 3),
            "pts": 0, "method": "predicted",
            "confidence": 0.0,
            "id": tid,
            "vx": round(tr["vx"], 3),
            "vy": round(tr["vy"], 3),
        })

    _tracked = new_tracked
    return result


# ── Main detection ────────────────────────────────────────────────────────────

def _detect_robots():
    if not _lidar or _robot_pos is None:
        return [], None

    now    = time.monotonic()
    rx, ry = _robot_pos
    fa_rad = math.radians(_heading())

    # Build angle-sorted absolute field-frame point array
    sorted_angles = sorted(_lidar.keys())
    angles    = np.array([math.radians(a) + fa_rad for a in sorted_angles])
    distances = np.array([_lidar[a] / 1000.0       for a in sorted_angles])
    pts       = _lidar_points(angles, distances, (rx, ry))

    clusters = _detect_clusters(pts)

    robots = []
    for cluster in clusters:
        if len(cluster) < MIN_CLUSTER_POINTS:
            continue

        center = np.mean(cluster, axis=0)

        # Arc centroid sits on the near surface; push outward by one radius
        # along the vector from the observing robot to get the true centre.
        direction = center - np.array([rx, ry])
        d = np.linalg.norm(direction)
        if d > 1e-9:
            center = center + (direction / d) * ROBOT_RADIUS

        if _is_near_wall(center):
            continue

        dists = np.linalg.norm(cluster - center, axis=1)
        
        # Standard deviation of distances should be small for a perfect circle, big for line-like features
        if np.std(dists) > 0.03:
            continue

        cx, cy = float(center[0]), float(center[1])
        robots.append({
            "x": round(cx, 3), "y": round(cy, 3),
            "pts": len(cluster), "method": "cluster",
            "confidence": float(len(cluster)),
        })

    robots.sort(key=lambda r: r["confidence"], reverse=True)
    robots = _filter_overlapping(robots)[:MAX_ROBOTS]
    robots = _match_and_track(robots, now)

    for r in robots:
        print(f"  [ROBOTS] id={r['id']}  {r['pts']:2d} pts"
              f"  ({r['x']:.3f}, {r['y']:.3f})"
              f"  conf={r['confidence']:.2f}  [{r['method']}]"
              f"  v=({r['vx']:.2f}, {r['vy']:.2f})")

    origin = {"x": round(rx, 4), "y": round(ry, 4), "heading": round(math.degrees(fa_rad), 3)}
    return robots, origin


# ── Broker interface ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _lidar, _robot_pos, _imu_pitch

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

    elif key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except (ValueError, TypeError):
            return

    if key == "lidar":
        robots, origin = _detect_robots()
        mb.set("other_robots", json.dumps({"origin": origin, "robots": robots}))


if __name__ == "__main__":
    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass
    try:
        raw = mb.get("robot_position")
        if raw:
            pos = json.loads(raw)
            _robot_pos = (float(pos["x"]), float(pos["y"]))
    except Exception:
        pass

    mb.setcallback(["lidar", "robot_position", "imu_pitch"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping robot detection.")
        mb.close()
