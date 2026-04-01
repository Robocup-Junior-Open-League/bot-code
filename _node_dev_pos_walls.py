from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import numpy as np

# Distance from the observed extreme within which a point is a wall candidate.
WALL_TOL = 0.05       # metres
# Minimum number of wall-candidate points required to report a wall.
MIN_WALL_POINTS = 8

DEBUG = False   # set True to print per-scan wall detection results

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_pos_walls", broker=mb)

_lidar     = {}    # angle_deg (int) → dist_mm (int)
_imu_pitch = None  # degrees — from imu_pitch broker key; None = not yet received


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


def _lidar_to_cartesian():
    """
    Convert the polar lidar scan to robot-centred field-aligned Cartesian
    coordinates.  Returns an (N, 2) numpy array, or None if there is no data.
    """
    if not _lidar:
        return None
    n         = len(_lidar)
    keys, vals = zip(*_lidar.items())
    angles    = np.radians(np.array(keys, dtype=float)) + np.radians(_heading())
    distances = np.array(vals, dtype=float) / 1000.0
    return np.column_stack((distances * np.cos(angles), distances * np.sin(angles)))


def _detect_walls(pts):
    """
    Detect walls from robot-centred field-aligned lidar points.

    For each axis, points within WALL_TOL of the observed minimum are
    candidates for one wall, and points within WALL_TOL of the observed
    maximum are candidates for the opposing wall.  The median of each
    qualifying group is used as the wall offset.

    Returns wall dicts: {"gradient": 0, "offset": y} for horizontal walls or
                        {"gradient": null, "offset": x} for vertical walls,
    all in robot-centred metres.
    """
    walls = []

    specs = [
        (0, None, "Left",   "Right"),
        (1,    0, "Bottom", "Top"),
    ]
    for axis, gradient, label_lo, label_hi in specs:
        vals  = pts[:, axis]
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))

        lo_group = vals[vals <= v_min + WALL_TOL]
        if len(lo_group) >= MIN_WALL_POINTS:
            offset = round(float(np.median(lo_group)), 3)
            if DEBUG:
                print(f"  [WALLS] {label_lo:6s}  offset={offset:+.3f} m  pts={len(lo_group)}")
            walls.append({"gradient": gradient, "offset": offset})

        hi_group = vals[vals >= v_max - WALL_TOL]
        if len(hi_group) >= MIN_WALL_POINTS:
            offset = round(float(np.median(hi_group)), 3)
            if DEBUG:
                print(f"  [WALLS] {label_hi:6s}  offset={offset:+.3f} m  pts={len(hi_group)}")
            walls.append({"gradient": gradient, "offset": offset})

    return walls


def on_update(key, value):
    global _lidar, _imu_pitch

    if value is None:
        return

    if key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except (ValueError, TypeError):
            pass
        return

    if key == "lidar":
        try:
            raw = json.loads(value)
            _lidar = {int(k): int(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            return

        with _perf.measure("lidar"):
            pts = _lidar_to_cartesian()
            if pts is None:
                return

            walls = _detect_walls(pts)
            if DEBUG:
                print(f"[WALLS] heading={_heading():.1f}°  points={len(_lidar)}"
                      f"  → {len(walls)} wall(s)")
            mb.set("lidar_walls", json.dumps(walls))


if __name__ == "__main__":
    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass

    mb.setcallback(["lidar", "imu_pitch"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping wall detection node.")
        mb.close()
