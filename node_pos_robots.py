from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
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

# ── Confidence & detection limits ─────────────────────────────────────────────
MAX_ROBOTS   = 3
OVERLAP_DIST = ROBOT_RADIUS * 2   # metres

# ── Tracking configuration ────────────────────────────────────────────────────
VEL_MIN_DT      = 0.05   # seconds — min elapsed time between history samples
VEL_HISTORY_N   = 10     # rolling history length per tracked robot
VEL_HISTORY_MIN = 3      # minimum samples before fitted velocity is trusted
MAX_ROBOT_SPEED = 2.0    # m/s — hard cap after fitting

_MAX_PRED_DT    = 0.5    # used internally for matching only

DEBUG = False   # set True to print per-scan detection results

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_pos_robots", broker=mb)

_lidar     = {}   # {angle_deg (int): dist_mm (int)}
_robot_pos = None # (x, y) metres, in field frame
_imu_pitch = None # degrees — from imu_pitch broker key

# ── Tracking state ────────────────────────────────────────────────────────────
_tracked  = {}   # id → {"x","y","vx","vy","t","history"}
_next_id  = 1


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
    diffs  = np.diff(points, axis=0)
    dists  = np.hypot(diffs[:, 0], diffs[:, 1])
    splits = np.where(dists >= threshold)[0] + 1
    return np.split(points, splits)


def _is_near_wall(center):
    x, y = center
    return (
        x < WALL_MARGIN or x > FIELD_WIDTH  - WALL_MARGIN or
        y < WALL_MARGIN or y > FIELD_HEIGHT - WALL_MARGIN
    )


# ── Overlap filtering ─────────────────────────────────────────────────────────

def _filter_overlapping(robots):
    kept = []
    for r in robots:
        if not any(math.hypot(r["x"] - k["x"], r["y"] - k["y"]) < OVERLAP_DIST
                   for k in kept):
            kept.append(r)
    return kept


# ── Tracking ──────────────────────────────────────────────────────────────────

def _predict_pos(x, y, vx, vy, dt):
    """Advance position with wall bouncing — used internally for matching."""
    dt   = min(dt, _MAX_PRED_DT)
    n    = max(1, int(dt / 0.02) + 1)
    step = dt / n
    for _ in range(n):
        x += vx * step;  y += vy * step
        if   x < ROBOT_RADIUS:               x = ROBOT_RADIUS;               vx =  abs(vx)
        elif x > FIELD_WIDTH - ROBOT_RADIUS:  x = FIELD_WIDTH - ROBOT_RADIUS; vx = -abs(vx)
        if   y < ROBOT_RADIUS:               y = ROBOT_RADIUS;               vy =  abs(vy)
        elif y > FIELD_HEIGHT - ROBOT_RADIUS: y = FIELD_HEIGHT - ROBOT_RADIUS; vy = -abs(vy)
    return x, y


def _fit_velocity(history):
    if len(history) < 2:
        return 0.0, 0.0
    arr = np.array(history, dtype=float)
    ts  = arr[:, 0] - arr[0, 0]
    if ts[-1] < 1e-9:
        return 0.0, 0.0
    coeffs = np.polyfit(ts, arr[:, 1:3], 1)
    return float(coeffs[0, 0]), float(coeffs[0, 1])


def _match_and_track(detections, now):
    """Match detections to tracked robots, assign IDs and fit velocity.
    Returns only currently-detected robots (with id/vx/vy); dead-reckoning
    of missing robots is handled by node_pos_predict."""
    global _tracked, _next_id

    predictions = {
        tid: _predict_pos(tr["x"], tr["y"], tr["vx"], tr["vy"], now - tr["t"])
        for tid, tr in _tracked.items()
    }

    matched_det   = [None] * len(detections)
    matched_track = set()

    if detections and predictions:
        pred_ids = list(predictions.keys())
        det_xy   = np.array([[d["x"], d["y"]] for d in detections])
        pred_xy  = np.array([predictions[tid] for tid in pred_ids])
        dist_mat = np.hypot(det_xy[:, 0:1] - pred_xy[:, 0],
                            det_xy[:, 1:2] - pred_xy[:, 1])
        for _, di, tid in sorted(
            (dist_mat[di, ti], di, pred_ids[ti])
            for di in range(len(detections))
            for ti in range(len(pred_ids))
        ):
            if matched_det[di] is None and tid not in matched_track:
                matched_det[di] = tid
                matched_track.add(tid)

    new_tracked = {}

    for di, det in enumerate(detections):
        tid = matched_det[di]
        if tid is not None:
            old     = _tracked[tid]
            history = old.get("history", [])
            if now - old["t"] >= VEL_MIN_DT:
                history = (history + [(now, det["x"], det["y"])])[-VEL_HISTORY_N:]
            new_vx, new_vy = _fit_velocity(history) if len(history) >= VEL_HISTORY_MIN \
                             else (old["vx"], old["vy"])
            spd = math.hypot(new_vx, new_vy)
            if spd > MAX_ROBOT_SPEED:
                new_vx *= MAX_ROBOT_SPEED / spd
                new_vy *= MAX_ROBOT_SPEED / spd
            new_tracked[tid] = {"x": det["x"], "y": det["y"], "t": now,
                                 "vx": new_vx, "vy": new_vy, "history": history}
        else:
            if len(new_tracked) >= MAX_ROBOTS:
                continue
            tid = _next_id;  _next_id += 1
            new_tracked[tid] = {"x": det["x"], "y": det["y"], "t": now,
                                 "vx": 0.0, "vy": 0.0,
                                 "history": [(now, det["x"], det["y"])]}
        det["id"]  = tid
        det["vx"]  = round(new_tracked[tid]["vx"], 3)
        det["vy"]  = round(new_tracked[tid]["vy"], 3)

    # Preserve unmatched tracked robots in state (without appending to output)
    for tid, tr in _tracked.items():
        if tid not in matched_track:
            new_tracked[tid] = tr   # keep last known state for dead-reckoning

    _tracked = new_tracked
    return list(detections)   # detected only


# ── Main detection ────────────────────────────────────────────────────────────

def _detect_robots():
    if not _lidar or _robot_pos is None:
        return [], None

    rx, ry = _robot_pos
    fa_rad = math.radians(_heading())

    sorted_items  = sorted(_lidar.items())
    angles        = np.radians([a for a, _ in sorted_items]) + fa_rad
    distances     = np.array([d for _, d in sorted_items]) / 1000.0
    pts           = _lidar_points(angles, distances, (rx, ry))

    robot_pos_arr = np.array([rx, ry])
    clusters      = _detect_clusters(pts)

    robots = []
    for cluster in clusters:
        if len(cluster) < MIN_CLUSTER_POINTS:
            continue

        center = np.mean(cluster, axis=0)
        direction = center - robot_pos_arr
        d = np.hypot(direction[0], direction[1])
        if d > 1e-9:
            center = center + (direction / d) * ROBOT_RADIUS

        if _is_near_wall(center):
            continue

        dists = np.linalg.norm(cluster - center, axis=1)
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

    if DEBUG:
        for r in robots:
            print(f"  [ROBOTS] {r['pts']:2d} pts  ({r['x']:.3f}, {r['y']:.3f})"
                  f"  conf={r['confidence']:.2f}")

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
        with _perf.measure("lidar"):
            now = time.monotonic()
            robots, origin = _detect_robots()
            robots = _match_and_track(robots, now)
            mb.set("other_robots_detected", json.dumps({"origin": origin, "robots": robots,
                                                         "t": now}))


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
