import json
import math
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from robus_core.libs.lib_telemtrybroker import TelemetryBroker

# ── Field configuration ───────────────────────────────────────────────────────
FIELD_WIDTH  = 1.82   # metres
FIELD_HEIGHT = 2.43   # metres
ROBOT_RADIUS = 0.09   # metres
# ─────────────────────────────────────────────────────────────────────────────

mb = TelemetryBroker()

# ── Broker state ──────────────────────────────────────────────────────────────
_lidar            = {}    # {angle_deg (int): dist_mm (int)}  — sensor frame
_detection_origin  = None  # (rx, ry) position snapshot from the last detection cycle
_detection_heading = None  # heading (degrees) snapshot from the last detection cycle
_imu_pitch        = None  # degrees — from imu_pitch broker key; None = not yet received
_robot_pos        = None  # (x, y) metres, field frame
_other_robots     = []    # [[x, y, method, id], ...]  field frame
_corners      = []    # [[angle_deg, dist_mm], ...]  sensor frame
_wall_corners       = []   # [[x, y], ...]  robot-centred field-aligned metres (primary)
_wall_corners_hist  = []   # [[x, y], ...]  robot-centred field-aligned metres (histogram)
_walls              = []   # [{"gradient": 0|None, "offset": float}]  robot-centred
_walls_hist         = []   # same format, from histogram detection
_positioning_corner  = None # {"dx": float, "dy": float}  robot-centred field-aligned
_position_history    = []   # [{"x", "y", "t"}, ...]
_other_robots_history = []  # [{"t", "robots": [{"x","y"}]}, ...]

_state_lock    = threading.Lock()
_needs_redraw  = threading.Event()
# ─────────────────────────────────────────────────────────────────────────────


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


# ── Matplotlib setup ──────────────────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(6, 9))
plt.tight_layout(pad=1.5)
plt.show(block=False)


def _lidar_to_field(angle_deg, dist_mm, fa_rad, origin=(0.0, 0.0)):
    """Convert a single sensor-frame reading to field-frame (x, y)."""
    d = dist_mm / 1000.0
    a = math.radians(angle_deg) + fa_rad
    return origin[0] + d * math.cos(a), origin[1] + d * math.sin(a)


def _redraw():
    ax.cla()

    fa          = _heading()
    fa_rad      = math.radians(fa)
    known_pos   = _robot_pos is not None
    origin      = _robot_pos if known_pos else (0.0, 0.0)

    # ── Lidar points ──────────────────────────────────────────────────────────
    # Use the detection-origin snapshot bundled in the last other_robots message
    # so that the scatter and the detected circles share the exact coordinate
    # frame, eliminating the inter-process race that otherwise shifts them apart.
    pts = None
    if _lidar:
        # Heading (fa_rad) comes from sim_heading published per-scan — always
        # current and consistent across processes, so use the vis's own value.
        # Position origin uses the detection snapshot to match the circles.
        lidar_origin  = (_detection_origin[0], _detection_origin[1]) \
                        if _detection_origin  is not None else origin
        lidar_fa_rad  = math.radians(_detection_heading) \
                        if _detection_heading is not None else fa_rad
        pts = np.array([
            _lidar_to_field(a, d, lidar_fa_rad, lidar_origin)
            for a, d in _lidar.items()
        ])
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=5, c='#222222', zorder=10, label='Lidar')

    # Centre point for the heading arrow: robot pos if known, else lidar centroid
    if known_pos:
        arrow_origin = origin
    elif pts is not None and len(pts):
        arrow_origin = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    else:
        arrow_origin = None

    # ── Detected walls (primary — clustering) ─────────────────────────────────
    # Wall offsets are robot-centred; shift into field coordinates.
    for wall in _walls:
        off = wall["offset"]
        if wall["gradient"] == 0:
            ax.axhline(origin[1] + off, color='steelblue', lw=1.5, ls='--', zorder=4)
        else:
            ax.axvline(origin[0] + off, color='steelblue', lw=1.5, ls='--', zorder=4)

    # ── Detected walls (secondary — histogram) ────────────────────────────────
    for wall in _walls_hist:
        off = wall["offset"]
        if wall["gradient"] == 0:
            ax.axhline(origin[1] + off, color='mediumpurple', lw=1.5, ls=':', zorder=4)
        else:
            ax.axvline(origin[0] + off, color='mediumpurple', lw=1.5, ls=':', zorder=4)

    if known_pos:
        # ── Field boundary ────────────────────────────────────────────────────
        ax.add_patch(patches.Rectangle(
            (0, 0), FIELD_WIDTH, FIELD_HEIGHT,
            linewidth=2, edgecolor='#888888', facecolor='#f8f8f8', zorder=1,
        ))

        # ── Own robot ─────────────────────────────────────────────────────────
        rx, ry = origin
        ax.add_patch(patches.Circle(
            (rx, ry), ROBOT_RADIUS,
            linewidth=1.5, edgecolor='#555555', facecolor='#cccccc', zorder=7,
        ))

        # ── Other robots ──────────────────────────────────────────────────────
        for r in _other_robots:
            ox, oy    = r[0], r[1]
            method    = str(r[2]) if len(r) > 2 else ""
            rid       = int(r[3]) if len(r) > 3 else 0
            predicted = method == "predicted"
            centroid  = method.startswith("centroid")

            # Unique colour per ID via tab10 (cycles every 10 IDs)
            c_solid = plt.cm.tab10((rid - 1) % 10)[:3]
            c_face  = (*c_solid, 0.15 if predicted else 0.3)

            if predicted:
                # Dashed circle at extrapolated position
                ax.add_patch(patches.Circle(
                    (ox, oy), ROBOT_RADIUS,
                    linewidth=1.5, edgecolor=c_solid, facecolor=c_face,
                    linestyle='--', zorder=6,
                ))
            elif centroid:
                ax.scatter(ox, oy, s=200, marker='X',
                           color=c_solid, linewidths=1.5, zorder=7)
            else:
                ax.add_patch(patches.Circle(
                    (ox, oy), ROBOT_RADIUS,
                    linewidth=1.5, edgecolor=c_solid, facecolor=c_face, zorder=7,
                ))
            ax.text(ox, oy + ROBOT_RADIUS + 0.03, f"#{rid}",
                    ha='center', va='bottom', fontsize=8,
                    color=c_solid, fontweight='bold',
                    alpha=0.5 if predicted else 1.0, zorder=8)

    # ── Heading arrow (always drawn if an origin is available) ───────────────
    if arrow_origin is not None:
        ax_x, ax_y = arrow_origin
        arrow_len  = ROBOT_RADIUS * 1.8
        ax.annotate(
            "", xy=(ax_x + arrow_len * math.cos(fa_rad),
                    ax_y + arrow_len * math.sin(fa_rad)),
            xytext=(ax_x, ax_y),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=2.0),
            zorder=8,
        )

    # ── Depth corners (sensor frame → field frame) ────────────────────────────
    if _corners:
        cpts = np.array([
            _lidar_to_field(a, d, fa_rad, origin)
            for a, d in _corners
        ])
        ax.scatter(cpts[:, 0], cpts[:, 1],
                   s=140, marker='X', c='red', edgecolors='black',
                   linewidths=0.8, zorder=6, label='Depth corners')

    # ── Wall corners — primary (robot-centred field-aligned → field frame) ─────
    if _wall_corners:
        wcpts = np.array([[origin[0] + x, origin[1] + y] for x, y in _wall_corners])
        ax.scatter(wcpts[:, 0], wcpts[:, 1],
                   s=140, marker='X', c='steelblue', edgecolors='black',
                   linewidths=0.8, zorder=6, label='Wall corners')

    # ── Wall corners — histogram ───────────────────────────────────────────────
    if _wall_corners_hist:
        whpts = np.array([[origin[0] + x, origin[1] + y] for x, y in _wall_corners_hist])
        ax.scatter(whpts[:, 0], whpts[:, 1],
                   s=140, marker='X', c='mediumpurple', edgecolors='black',
                   linewidths=0.8, zorder=6, label='Wall corners (hist)')

    # ── Position history — fading green trail ────────────────────────────────
    if known_pos and len(_position_history) > 1:
        t0  = _position_history[0]["t"]
        t1  = _position_history[-1]["t"]
        rng = max(t1 - t0, 1e-9)
        xs  = [p["x"] for p in _position_history]
        ys  = [p["y"] for p in _position_history]
        colors = [(0.4, 0.4, 0.4, 0.05 + 0.7 * (p["t"] - t0) / rng)
                  for p in _position_history]
        ax.scatter(xs, ys, s=18, color=colors, zorder=5)

    # ── Other robots history — fading per-ID coloured trail ──────────────────
    if known_pos and _other_robots_history:
        t0  = _other_robots_history[0]["t"]
        t1  = _other_robots_history[-1]["t"]
        rng = max(t1 - t0, 1e-9)
        for snap in _other_robots_history:
            alpha = 0.05 + 0.6 * (snap["t"] - t0) / rng
            for r in snap["robots"]:
                rid     = int(r.get("id", 0))
                c_solid = plt.cm.tab10((rid - 1) % 10)[:3]
                ax.scatter(r["x"], r["y"], s=18,
                           color=(*c_solid, alpha), zorder=5)

    # ── Winning positioning corner (green, topmost) ────────────────────────────
    if _positioning_corner is not None:
        pcx = origin[0] + _positioning_corner["dx"]
        pcy = origin[1] + _positioning_corner["dy"]
        ax.scatter(pcx, pcy, s=200, marker='X', c='limegreen', edgecolors='black',
                   linewidths=1.0, zorder=11, label='Positioning corner')

    # ── Axes ──────────────────────────────────────────────────────────────────
    if known_pos:
        margin = 0.25
        ax.set_xlim(-margin, FIELD_WIDTH  + margin)
        ax.set_ylim(-margin, FIELD_HEIGHT + margin)
    elif arrow_origin is not None:
        cx, cy  = arrow_origin
        spread  = max(float(pts[:, 0].max() - pts[:, 0].min()),
                      float(pts[:, 1].max() - pts[:, 1].min()), 0.5) / 2 + 0.3
        ax.set_xlim(cx - spread, cx + spread)
        ax.set_ylim(cy - spread, cy + spread)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    pos_str = f"({origin[0]:.2f}, {origin[1]:.2f})" if known_pos else "unknown"
    ax.set_title(
        f"Twin Visualisation\n"
        f"pos={pos_str}  heading={fa:.1f}°"
        f"  walls={len(_walls)}/hist={len(_walls_hist)}  corners={len(_corners)}  bots={len(_other_robots)}"
    )
    ax.grid(True, alpha=0.25, zorder=0)

    fig.canvas.draw()
    fig.canvas.flush_events()


# ── Broker callbacks ──────────────────────────────────────────────────────────
def on_update(key, value):
    global _lidar, _detection_origin, _detection_heading, _imu_pitch
    global _robot_pos, _other_robots, _corners, _wall_corners, _wall_corners_hist
    global _walls, _walls_hist, _positioning_corner
    global _position_history, _other_robots_history

    if value is None:
        return

    try:
        with _state_lock:
            if key == "lidar":
                raw = json.loads(value)
                _lidar = {int(k): int(v) for k, v in raw.items()}

            elif key == "imu_pitch":
                _imu_pitch = float(value)

            elif key == "robot_position":
                p = json.loads(value)
                _robot_pos = (float(p["x"]), float(p["y"]))

            elif key == "other_robots":
                payload = json.loads(value)
                # New format: {"origin": {x, y}, "robots": [...]}
                # Fallback: bare list (old format / other producers)
                if isinstance(payload, dict):
                    orig = payload.get("origin")
                    if orig:
                        _detection_origin  = (float(orig["x"]), float(orig["y"]))
                        if "heading" in orig:
                            _detection_heading = float(orig["heading"])
                    robot_list = payload.get("robots", [])
                else:
                    robot_list = payload
                _other_robots = [[float(r["x"]), float(r["y"]),
                                   r.get("method", ""), int(r.get("id", 0))]
                                  for r in robot_list]

            elif key == "depth_corners":
                _corners = json.loads(value)

            elif key == "wall_corners":
                _wall_corners = json.loads(value)

            elif key == "wall_corners_hist":
                _wall_corners_hist = json.loads(value)

            elif key == "positioning_corner":
                _positioning_corner = json.loads(value)

            elif key == "lidar_walls":
                _walls = json.loads(value)

            elif key == "lidar_walls_hist":
                _walls_hist = json.loads(value)

            elif key == "position_history":
                _position_history = json.loads(value)

            elif key == "other_robots_history":
                _other_robots_history = json.loads(value)

    except Exception as e:
        print(f"[VIS] parse error on {key!r}: {e}")
        return

    _needs_redraw.set()


if __name__ == "__main__":
    # Seed existing broker values so the display is populated immediately
    _SEEDS = {
        "imu_pitch":      lambda v: float(v),
        "lidar":          lambda v: {int(k): int(x) for k, x in json.loads(v).items()},
        "robot_position": lambda v: (float(json.loads(v)["x"]), float(json.loads(v)["y"])),
        "other_robots":   lambda v: [[float(r["x"]), float(r["y"]), r.get("method", ""), int(r.get("id", 0))]
                                      for r in (lambda p: p.get("robots", p) if isinstance(p, dict) else p)(json.loads(v))],
        "depth_corners":  lambda v: json.loads(v),
        "wall_corners":       lambda v: json.loads(v),
        "wall_corners_hist":  lambda v: json.loads(v),
        "positioning_corner": lambda v: json.loads(v),
        "lidar_walls":           lambda v: json.loads(v),
        "lidar_walls_hist":      lambda v: json.loads(v),
        "position_history":      lambda v: json.loads(v),
        "other_robots_history":  lambda v: json.loads(v),
    }
    _TARGETS = {
        "imu_pitch":      "_imu_pitch",
        "lidar":          "_lidar",
        "robot_position": "_robot_pos",
        "other_robots":   "_other_robots",
        "depth_corners":  "_corners",
        "wall_corners":       "_wall_corners",
        "wall_corners_hist":  "_wall_corners_hist",
        "positioning_corner": "_positioning_corner",
        "lidar_walls":           "_walls",
        "lidar_walls_hist":      "_walls_hist",
        "position_history":      "_position_history",
        "other_robots_history":  "_other_robots_history",
    }
    for key, parse in _SEEDS.items():
        try:
            val = mb.get(key)
            if val is not None:
                globals()[_TARGETS[key]] = parse(val)
        except Exception:
            pass

    _redraw()

    mb.setcallback(list(_SEEDS.keys()), on_update)
    threading.Thread(target=mb.receiver_loop, daemon=True, name="broker-receiver").start()

    try:
        while plt.fignum_exists(fig.number):
            if _needs_redraw.is_set():
                _needs_redraw.clear()
                with _state_lock:
                    _redraw()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping twin visualisation.")
        plt.close(fig)
        mb.close()
