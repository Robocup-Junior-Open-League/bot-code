from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import numpy as np

# ── Field configuration ────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.58   # metres, X axis — playing field only
FIELD_HEIGHT = 2.19   # metres, Y axis
ROBOT_RADIUS = 0.09   # metres

# Tolerance for the valid-position bounds check.
_MARGIN            = 0.05   # metres
# Candidates within this distance count as mutual support.
_OUTLIER_THRESHOLD = 0.15   # metres
# How far outside the field boundary a lidar point may land.
_LIDAR_FIELD_TOL   = 0.05   # metres

DEBUG = False   # set True to print per-update positioning details
# ──────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_dev_pos", broker=mb)

_imu_pitch   = None   # degrees — from imu_pitch broker key
_lidar       = {}     # {angle_deg: dist_mm}
_lidar_walls = []     # [{"gradient": 0|null, "offset": float}, ...]


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


def _compute_position():
    """
    Derive robot position from detected wall offsets.

    Each wall offset can correspond to either of the two parallel field
    boundaries.  Both candidates are generated, filtered to positions that
    keep the robot inside the field (with margin), and optionally tested
    against the lidar bounding box.  The most-supported candidate along
    each axis is returned.

    Vertical walls   (gradient=null)  constrain x.
    Horizontal walls (gradient=0)     constrain y.
    """
    if not _lidar_walls:
        return None

    eff_angle = _heading()
    fa_rad    = math.radians(eff_angle)

    # ── Lidar bounding box (field-aligned, robot-centred) ─────────────────────
    lidar_min_x = lidar_max_x = lidar_min_y = lidar_max_y = 0.0
    if _lidar:
        n     = len(_lidar)
        keys, vals = zip(*_lidar.items())
        a_arr = np.radians(np.array(keys, dtype=float)) + fa_rad
        d_arr = np.array(vals, dtype=float) / 1000.0
        ox = d_arr * np.cos(a_arr)
        oy = d_arr * np.sin(a_arr)
        lidar_min_x, lidar_max_x = float(ox.min()), float(ox.max())
        lidar_min_y, lidar_max_y = float(oy.min()), float(oy.max())

    # ── Generate axis candidates ───────────────────────────────────────────────
    x_candidates = []
    y_candidates = []

    for wall in _lidar_walls:
        gradient = wall.get("gradient")
        offset   = float(wall.get("offset", 0.0))

        _OM = 0.12   # outer margin
        if gradient is None:  # vertical wall → constrains x
            for rx in (-_OM - offset, FIELD_WIDTH + _OM - offset):
                if not (ROBOT_RADIUS - _MARGIN <= rx <= FIELD_WIDTH - ROBOT_RADIUS + _MARGIN):
                    continue
                if _lidar and not (
                    rx + lidar_min_x >= -_OM - _LIDAR_FIELD_TOL and
                    rx + lidar_max_x <= FIELD_WIDTH + _OM + _LIDAR_FIELD_TOL
                ):
                    continue
                x_candidates.append(rx)

        else:  # horizontal wall → constrains y
            for ry in (-_OM - offset, FIELD_HEIGHT + _OM - offset):
                if not (ROBOT_RADIUS - _MARGIN <= ry <= FIELD_HEIGHT - ROBOT_RADIUS + _MARGIN):
                    continue
                if _lidar and not (
                    ry + lidar_min_y >= -_OM - _LIDAR_FIELD_TOL and
                    ry + lidar_max_y <= FIELD_HEIGHT + _OM + _LIDAR_FIELD_TOL
                ):
                    continue
                y_candidates.append(ry)

    if not x_candidates or not y_candidates:
        if DEBUG:
            print(f"[POS] heading={eff_angle:.1f}°  walls={len(_lidar_walls)}  → no valid candidates")
        return None

    def _best(candidates):
        return max(candidates, key=lambda c: sum(
            1 for o in candidates if abs(c - o) <= _OUTLIER_THRESHOLD
        ))

    rx = _best(x_candidates)
    ry = _best(y_candidates)
    if DEBUG:
        print(f"[POS] heading={eff_angle:.1f}°  walls={len(_lidar_walls)}"
              f"  → pos=({rx:.3f}, {ry:.3f}) m")
    return round(rx, 3), round(ry, 3)


def on_update(key, value):
    global _imu_pitch, _lidar, _lidar_walls

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
            pass
        return  # lidar alone doesn't trigger repositioning

    if key == "lidar_walls":
        try:
            _lidar_walls = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return

        with _perf.measure("lidar_walls"):
            pos = _compute_position()
            if pos is not None:
                mb.set("robot_position", json.dumps({"x": pos[0], "y": pos[1]}))


if __name__ == "__main__":
    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass

    mb.setcallback(["lidar", "imu_pitch", "lidar_walls"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping positioning node.")
        mb.close()
