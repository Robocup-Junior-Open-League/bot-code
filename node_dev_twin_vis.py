import json
import math
import time
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
import numpy as np
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor

# ── Field configuration ───────────────────────────────────────────────────────
FIELD_WIDTH  = 1.58   # metres — playing field only (white line to white line)
FIELD_HEIGHT = 2.19   # metres
ROBOT_RADIUS = 0.09   # metres
OUTER_MARGIN = 0.12   # outer area beyond white line on all sides
GOAL_WIDTH   = 0.60   # goal inner width (centred on each short side)
_MARGIN         = 0.10   # extra plot margin beyond the outer walls
_MAX_VIS_ROBOTS = 3
_MAX_WALLS      = 4   # pre-allocated wall line slots per detection source
# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_dev_twin_vis", broker=mb, print_every=50)

# ── Broker state ──────────────────────────────────────────────────────────────
_lidar             = {}
_detection_origin  = None
_detection_heading = None  # radians
_imu_pitch         = None
_robot_pos         = None
_other_robots      = []
_walls                = []
_position_history     = []
_other_robots_history = []
_ball_pos             = None  # {"x": float, "y": float} or None — fused detection (global_pos from prediction)
_ball_hidden_pos      = None  # {"x": float, "y": float} or None — extrapolated while hidden
_ball_lost            = False # True when prediction is in FOV but ball not detected
_ball_vx              = None  # m/s — fitted horizontal velocity
_ball_vy              = None  # m/s — fitted vertical velocity
_ball_history         = []    # [{"x", "y", "t"}, ...] from ball_history key
_sim_ball_pos         = None  # {"x": float, "y": float} or None — true sim position
_sim_state            = None  # {"robot": [x,y], "obstacles": [[x,y],...]} from sim_state key
_ally_id              = None  # persistent ally robot ID from cooperation node
_ally_pos_raw         = {}    # "ally_main_robot_pos" / "ally_other_pos_N" / "ally_ball_pos" → {"x","y"}
_raw_robots           = None  # list of raw robot positions from positioning (without ally)
_ball_raw             = None  # raw ball position from detection (without ally)
_game_state        = None  # latest game_state payload from master node
_strategy_points      = []   # list of {"x", "y"} from robot_strategy_points

_state_lock   = threading.Lock()
_needs_redraw = threading.Event()
# ─────────────────────────────────────────────────────────────────────────────


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


# ── Matplotlib setup ──────────────────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(6, 9))
plt.tight_layout(pad=1.5)

# Fixed axes — set once, never changed (required for stable blitting)
_OW = OUTER_MARGIN   # shorthand
ax.set_xlim(-_OW - _MARGIN, FIELD_WIDTH  + _OW + _MARGIN)
ax.set_ylim(-_OW - _MARGIN, FIELD_HEIGHT + _OW + _MARGIN)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.grid(True, alpha=0.25, zorder=0)

# ── Static background (baked into cached background, not animated) ────────────
_gx_min = (FIELD_WIDTH - GOAL_WIDTH) / 2
_gx_max = (FIELD_WIDTH + GOAL_WIDTH) / 2

# Outer area (outer wall rectangle, matte black/dark)
ax.add_patch(patches.Rectangle(
    (-_OW, -_OW), FIELD_WIDTH + 2 * _OW, FIELD_HEIGHT + 2 * _OW,
    linewidth=2, edgecolor='#222222', facecolor='#cccccc', zorder=1,
))

# Playing field (inner green)
ax.add_patch(patches.Rectangle(
    (0, 0), FIELD_WIDTH, FIELD_HEIGHT,
    linewidth=1, edgecolor='white', facecolor='#c8e6c9', zorder=2,
))

# Bottom goal — yellow interior
ax.add_patch(patches.Rectangle(
    (_gx_min, -_OW), GOAL_WIDTH, _OW,
    linewidth=0, facecolor='#ffee44', zorder=3,
))
# Top goal — blue interior
ax.add_patch(patches.Rectangle(
    (_gx_min, FIELD_HEIGHT), GOAL_WIDTH, _OW,
    linewidth=0, facecolor='#4488ff', zorder=3,
))

# Goal side walls (solid black lines in the outer margin)
for _gx in (_gx_min, _gx_max):
    ax.add_line(plt.Line2D([_gx, _gx], [-_OW, 0.0],
                           color='#222222', lw=2, zorder=4))
    ax.add_line(plt.Line2D([_gx, _gx], [FIELD_HEIGHT, FIELD_HEIGHT + _OW],
                           color='#222222', lw=2, zorder=4))

# ── Pre-allocated animated artists ───────────────────────────────────────────

# Lidar scatter
# Pre-cached tab10 palette — avoids a colormap call every robot every frame
_TAB10 = [tuple(plt.cm.tab10(i)[:3]) for i in range(10)]

# Lidar scatter
_art_lidar = ax.scatter([], [], s=5, c='#222222', zorder=10, animated=True)

# Own robot
_art_self = patches.Circle((0, 0), ROBOT_RADIUS,
    lw=1.5, edgecolor='#2a7a2a', facecolor='#aaddaa',
    zorder=7, animated=True, visible=False)
ax.add_patch(_art_self)

# Other robots — pre-allocated slots
_art_bots  = []
_art_blbls = []
for _ in range(_MAX_VIS_ROBOTS):
    c = patches.Circle((0, 0), ROBOT_RADIUS,
        lw=1.5, edgecolor='gray', facecolor='lightgray',
        zorder=7, animated=True, visible=False)
    ax.add_patch(c)
    t = ax.text(0, 0, '', ha='center', va='bottom', fontsize=8,
        fontweight='bold', animated=True, visible=False, zorder=8)
    _art_bots.append(c)
    _art_blbls.append(t)

# Heading arrow
_art_arrow = FancyArrowPatch((0, 0), (0.1, 0),
    arrowstyle='->', color='#2a7a2a', lw=2.0, mutation_scale=15,
    zorder=8, animated=True, visible=False)
ax.add_patch(_art_arrow)

# Wall lines — 4 primary + 4 hist
_WSPAN_X = (-_OW - _MARGIN, FIELD_WIDTH  + _OW + _MARGIN)
_WSPAN_Y = (-_OW - _MARGIN, FIELD_HEIGHT + _OW + _MARGIN)

def _wall_line(color, ls):
    (ln,) = ax.plot([], [], color=color, lw=1.5, ls=ls,
        zorder=4, animated=True, visible=False)
    return ln

_art_walls = [_wall_line('steelblue', '--') for _ in range(_MAX_WALLS)]

# Position history scatter
_art_pos_hist = ax.scatter([], [], s=18, zorder=5, animated=True,
                            edgecolors='none')

# Robot history scatter
_art_bot_hist = ax.scatter([], [], s=18, zorder=5, animated=True,
                            edgecolors='none')

# Ball — filled circle for detected position, cross for true sim position
_BALL_RADIUS = 0.021   # metres (21 mm physical radius)
_art_ball = patches.Circle((0, 0), _BALL_RADIUS,
    lw=1.5, edgecolor='darkorange', facecolor='orange',
    zorder=9, animated=True, visible=False)
ax.add_patch(_art_ball)

# Ball history scatter
_art_ball_hist = ax.scatter([], [], s=14, zorder=5, animated=True, edgecolors='none')

# Ball velocity arrow
_art_ball_arrow = FancyArrowPatch((0, 0), (0.1, 0),
    arrowstyle='->', color='darkorange', lw=1.5, mutation_scale=10,
    zorder=9, animated=True, visible=False)
ax.add_patch(_art_ball_arrow)

# Ball hidden (extrapolated) position — dashed ghost circle shown when occluded
_art_ball_hidden = patches.Circle((0, 0), _BALL_RADIUS,
    lw=1.5, edgecolor='darkorange', facecolor=(1.0, 0.55, 0.0, 0.25), ls='--',
    zorder=8, animated=True, visible=False)
ax.add_patch(_art_ball_hidden)

# Sim ground-truth crosses (shown alongside the detected circles).
# Use Line2D (plot) rather than scatter — simpler blitting semantics,
# no risk of scatter's PathCollection invalidating the axes bounding box.
def _sim_cross(color):
    (art,) = ax.plot([], [], marker='+', markersize=12, markeredgewidth=2,
                     color=color, ls='none', zorder=8, animated=True)
    return art

_art_sim_ball = _sim_cross('darkorange')
_art_sim_self = _sim_cross('#2a7a2a')
_art_sim_obs = [_sim_cross((0.90, 0.22, 0.18)) for _ in range(3)]

# Ally detected positions — raw reports from the cooperation node
_ALLY_BLUE = (0.13, 0.53, 0.90)
_MAX_ALLY_OTHERS = 3
def _ally_marker(marker):
    (art,) = ax.plot([], [], marker=marker, markersize=10, markeredgewidth=2,
                     color=_ALLY_BLUE, ls='none', zorder=9, animated=True)
    return art

_art_ally_main = _ally_marker('D')
_art_ally_det  = [_ally_marker('x') for _ in range(_MAX_ALLY_OTHERS)]
_art_ally_ball = _ally_marker('*')   # ally's detected ball position

# Strategy points — green circles + connecting polyline + index labels
_MAX_STRATEGY_PTS = 5
_STRATEGY_GREEN   = '#22aa22'

_art_strategy_pts    = []
_art_strategy_lbls   = []
_art_strategy_arrows = []
for _i in range(_MAX_STRATEGY_PTS):
    _c = patches.Circle((0, 0), 0.025,
        lw=1.5, edgecolor=_STRATEGY_GREEN, facecolor='#88dd88',
        zorder=6, animated=True, visible=False)
    ax.add_patch(_c)
    _art_strategy_pts.append(_c)
    _t = ax.text(0, 0, str(_i), ha='center', va='center', fontsize=8,
        color='black', fontweight='bold', animated=True, visible=False, zorder=7)
    _art_strategy_lbls.append(_t)
    _a = FancyArrowPatch(
        posA=(0, 0), posB=(0.1, 0.1),
        arrowstyle='->', color=_STRATEGY_GREEN,
        mutation_scale=10, lw=1.5,
        animated=True, visible=False, zorder=6)
    ax.add_patch(_a)
    _art_strategy_arrows.append(_a)

(_art_strategy_line,) = ax.plot([], [], color=_STRATEGY_GREEN, lw=1.5,
                                 zorder=5, animated=True)

# Status text (inside axes, top-left corner)
_art_status = ax.text(0.01, 0.99, '', transform=ax.transAxes,
    ha='left', va='top', fontsize=8, animated=True, zorder=15,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              alpha=0.75, edgecolor='none'))

# Game-state text (outside axes, below the legend on the right)
_art_game_state = ax.text(1.02, 0.055, '', transform=ax.transAxes,
    ha='left', va='top', fontsize=12, animated=True, zorder=15, clip_on=False,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              alpha=0.75, edgecolor='gray'))

# ── Legend ────────────────────────────────────────────────────────────────────
_legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaddaa',
           markeredgecolor='#2a7a2a', markersize=8, label='Own robot'),
    Line2D([0], [0], color='#2a7a2a', lw=2, label='Heading'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaddaa',
           markeredgecolor=None, markersize=3, label='Own history'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=_ALLY_BLUE,
           markeredgecolor=_ALLY_BLUE, markersize=8, label='Ally robot'),
    Line2D([0], [0], marker='d', color='w', markerfacecolor=_ALLY_BLUE,
           markeredgecolor=_ALLY_BLUE, markersize=4, label='Ally position'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=_ALLY_BLUE,
           markeredgecolor=None, markersize=3, label='Ally history'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor='darkred', markersize=8, label='Enemy robot'),
    Line2D([0], [0], marker='+', color='red', linestyle='None',
           markersize=4, label='Own detections'),
    Line2D([0], [0], marker='x', color=_ALLY_BLUE, linestyle='None',
           markersize=4, label='Ally detections'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor=None, markersize=3, label='Enemy history'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
           markeredgecolor='darkorange', markersize=4, label='Ball'),
    Line2D([0], [0], color='orange', lw=1, label='Ball velocity'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=(1.0, 0.55, 0.0, 0.25),
           markeredgecolor='darkorange', linestyle='--', markersize=4,
           label='Ball (extrapolated prediction)'),
    Line2D([0], [0], marker='+', color='darkorange', linestyle='None',
           markersize=4, label='Own ball detection)'),
    Line2D([0], [0], marker='*', color=_ALLY_BLUE, linestyle='None',
           markersize=4, label='Ally ball detection)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkorange',
           markeredgecolor=None, markersize=3, label='Ball history'),
    Line2D([0], [0], color='steelblue', lw=1.5, linestyle='--',
           label='Walls'),
    Line2D([0], [0], marker='.', color='black', linestyle='None',
           markersize=4, label='Lidar'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#88dd88',
           markeredgecolor=_STRATEGY_GREEN, markersize=8, label='Strategy point'),
    Line2D([0], [0], color=_STRATEGY_GREEN, lw=1.5, label='Strategy path'),
]

ax.legend(handles=_legend_handles,
          loc='upper left',
          bbox_to_anchor=(1.02, 1.0),
          borderaxespad=0.)

# ── Background cache ──────────────────────────────────────────────────────────
_bg = None

def _cache_bg(event=None):
    global _bg
    _bg = fig.canvas.copy_from_bbox(fig.bbox)

fig.canvas.mpl_connect('draw_event', _cache_bg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _update_wall_lines(walls, art_list, origin):
    for i, line in enumerate(art_list):
        if i < len(walls):
            off = walls[i]["offset"]
            if walls[i]["gradient"] == 0:
                y = origin[1] + off
                line.set_xdata([*_WSPAN_X])
                line.set_ydata([y, y])
            else:
                x = origin[0] + off
                line.set_xdata([x, x])
                line.set_ydata([*_WSPAN_Y])
            line.set_visible(True)
        else:
            line.set_visible(False)


# ── Blit redraw ───────────────────────────────────────────────────────────────

def _redraw():
    if _bg is None:
        return

    fa        = _heading()
    fa_rad    = math.radians(fa)
    known_pos = _robot_pos is not None
    origin    = _robot_pos if known_pos else (0.0, 0.0)

    # ── Lidar ─────────────────────────────────────────────────────────────────
    if _lidar:
        lo    = _detection_origin if _detection_origin is not None else origin
        lh    = _detection_heading if _detection_heading is not None else fa_rad
        angs  = np.radians(list(_lidar.keys())) + lh
        dists = np.array(list(_lidar.values())) / 1000.0
        _art_lidar.set_offsets(np.column_stack((
            lo[0] + dists * np.cos(angs),
            lo[1] + dists * np.sin(angs),
        )))
    else:
        _art_lidar.set_offsets(np.empty((0, 2)))

    # ── Own robot ─────────────────────────────────────────────────────────────
    _art_self.set_visible(known_pos)
    if known_pos:
        _art_self.set_center(origin)

    # ── Other robots ──────────────────────────────────────────────────────────
    for i in range(_MAX_VIS_ROBOTS):
        circ = _art_bots[i]
        lbl  = _art_blbls[i]
        if known_pos and i < len(_other_robots):
            r         = _other_robots[i]
            ox, oy    = r[0], r[1]
            method    = str(r[2]) if len(r) > 2 else ""
            rid       = int(r[3]) if len(r) > 3 else 0
            is_ally   = _ally_id is not None and rid == _ally_id
            predicted = method == "predicted"
            c_solid   = _ALLY_BLUE if is_ally else (0.90, 0.22, 0.18)
            c_face    = c_solid + (0.15 if predicted else 0.3,)
            circ.set_center((ox, oy))
            circ.set_edgecolor(c_solid)
            circ.set_facecolor(c_face)
            circ.set_linestyle('--' if predicted else '-')
            circ.set_visible(True)
            lbl.set_position((ox, oy + ROBOT_RADIUS + 0.03))
            lbl.set_text(f"#{rid}")
            lbl.set_color(c_solid)
            lbl.set_alpha(0.5 if predicted else 1.0)
            lbl.set_visible(True)
        else:
            circ.set_visible(False)
            lbl.set_visible(False)

    # ── Heading arrow ─────────────────────────────────────────────────────────
    arrow_len = ROBOT_RADIUS * 1.8
    _art_arrow.set_positions(
        origin,
        (origin[0] + arrow_len * math.cos(fa_rad),
         origin[1] + arrow_len * math.sin(fa_rad)),
    )
    _art_arrow.set_visible(True)

    # ── Ball ──────────────────────────────────────────────────────────────────
    if _ball_pos is not None:
        _art_ball.set_center((_ball_pos["x"], _ball_pos["y"]))
        _art_ball.set_visible(True)
        _art_ball_hidden.set_visible(False)
    elif _ball_hidden_pos is not None:
        _art_ball.set_visible(False)
        _art_ball_hidden.set_center((_ball_hidden_pos["x"], _ball_hidden_pos["y"]))
        _art_ball_hidden.set_edgecolor('red' if _ball_lost else 'darkorange')
        _art_ball_hidden.set_facecolor((1.0, 0.0, 0.0, 0.25) if _ball_lost
                                       else (1.0, 0.55, 0.0, 0.25))
        _art_ball_hidden.set_visible(True)
    else:
        _art_ball.set_visible(False)
        _art_ball_hidden.set_visible(False)

    # ── Ball history trail ────────────────────────────────────────────────────
    if len(_ball_history) > 1:
        arr   = np.array([(p["x"], p["y"], p["t"]) for p in _ball_history])
        t0    = arr[0, 2];  rng = max(arr[-1, 2] - t0, 1e-9)
        alpha = 0.05 + 0.7 * (arr[:, 2] - t0) / rng
        rgba  = np.column_stack([np.full((len(arr), 3), [1.0, 0.55, 0.0]), alpha])
        _art_ball_hist.set_offsets(arr[:, :2])
        _art_ball_hist.set_facecolors(rgba)
    else:
        _art_ball_hist.set_offsets(np.empty((0, 2)))

    # ── Ball velocity arrow ───────────────────────────────────────────────────
    _arrow_origin = _ball_pos or _ball_hidden_pos
    if (_arrow_origin is not None and _ball_vx is not None and _ball_vy is not None
            and math.hypot(_ball_vx, _ball_vy) > 0.1):
        bx, by = _arrow_origin["x"], _arrow_origin["y"]
        _art_ball_arrow.set_positions(
            (bx, by),
            (bx + _ball_vx * 0.5, by + _ball_vy * 0.5),
        )
        _art_ball_arrow.set_visible(True)
    else:
        _art_ball_arrow.set_visible(False)

    # ── Raw detection crosses ─────────────────────────────────────────────────
    if _ball_raw is not None and _ball_raw["global_pos"] is not None:
        _art_sim_ball.set_data([_ball_raw["global_pos"]["x"]], [_ball_raw["global_pos"]["y"]])
    else:
        _art_sim_ball.set_data([], [])

    if _raw_robots is not None:
        for i, art in enumerate(_art_sim_obs):
            if i < len(_raw_robots):
                robot = _raw_robots[i]
                art.set_data([robot["x"]], [robot["y"]])
            else:
                art.set_data([], [])
    else:
        for i, art in enumerate(_art_sim_obs):
            art.set_data([], [])

    # ── Ally detected positions ───────────────────────────────────────────────
    p = _ally_pos_raw.get("ally_main_robot_pos")
    _art_ally_main.set_data([p["x"]], [p["y"]]) if p else _art_ally_main.set_data([], [])
    for i, art in enumerate(_art_ally_det):
        p = _ally_pos_raw.get(f"ally_other_pos_{i + 1}")
        art.set_data([p["x"]], [p["y"]]) if p else art.set_data([], [])
    p = _ally_pos_raw.get("ally_ball_pos")
    _art_ally_ball.set_data([p["x"]], [p["y"]]) if p else _art_ally_ball.set_data([], [])

    # ── Walls ─────────────────────────────────────────────────────────────────
    _update_wall_lines(_walls, _art_walls, origin)

    # ── Position history ──────────────────────────────────────────────────────
    if known_pos and len(_position_history) > 1:
        arr   = np.array([(p["x"], p["y"], p["t"]) for p in _position_history])
        t0    = arr[0, 2];  rng = max(arr[-1, 2] - t0, 1e-9)
        alpha = 0.05 + 0.7 * (arr[:, 2] - t0) / rng
        rgba  = np.column_stack([np.full((len(arr), 3), [0.17, 0.48, 0.17]), alpha])
        _art_pos_hist.set_offsets(arr[:, :2])
        _art_pos_hist.set_facecolors(rgba)
    else:
        _art_pos_hist.set_offsets(np.empty((0, 2)))

    # ── Robot history ─────────────────────────────────────────────────────────
    if known_pos and _other_robots_history:
        t0   = _other_robots_history[0]["t"]
        rng  = max(_other_robots_history[-1]["t"] - t0, 1e-9)
        pts, rgba = [], []
        for snap in _other_robots_history:
            alpha = 0.05 + 0.6 * (snap["t"] - t0) / rng
            for r in snap["robots"]:
                rid = int(r.get("id", 0))
                c   = _ALLY_BLUE if (_ally_id is not None and rid == _ally_id) else (0.90, 0.22, 0.18)  # history lacks ally flag → use id
                pts.append((r["x"], r["y"]))
                rgba.append(c + (alpha,))
        if pts:
            _art_bot_hist.set_offsets(np.array(pts))
            _art_bot_hist.set_facecolors(rgba)
        else:
            _art_bot_hist.set_offsets(np.empty((0, 2)))
    else:
        _art_bot_hist.set_offsets(np.empty((0, 2)))

    # ── Strategy points ───────────────────────────────────────────────────────
    n = len(_strategy_points)
    for i, (c, lbl, arrow) in enumerate(
            zip(_art_strategy_pts, _art_strategy_lbls, _art_strategy_arrows)):
        if i < n:
            pt = _strategy_points[i]
            px, py = pt["x"], pt["y"]
            c.set_center((px, py))
            c.set_visible(True)
            lbl.set_position((px, py))
            lbl.set_visible(True)
            d = pt.get("dir")
            if d is not None:
                _adx, _ady = d["x"] - px, d["y"] - py
                _alen = math.hypot(_adx, _ady)
                if _alen > 1e-6:
                    _s = ROBOT_RADIUS * 2.0 / _alen
                    arrow.set_positions((px, py), (px + _adx * _s, py + _ady * _s))
                    arrow.set_visible(True)
                else:
                    arrow.set_visible(False)
            else:
                arrow.set_visible(False)
        else:
            c.set_visible(False)
            lbl.set_visible(False)
            arrow.set_visible(False)
    if n >= 1:
        pos = _robot_pos
        _art_strategy_line.set_data(
            [pos[0]] + [p["x"] for p in _strategy_points],
            [pos[1]] + [p["y"] for p in _strategy_points],
        )
    else:
        _art_strategy_line.set_data([], [])

    # ── Status text ───────────────────────────────────────────────────────────
    pos_str = f"({origin[0]:.2f}, {origin[1]:.2f})" if known_pos else "unknown"
    _art_status.set_text(
        f"pos={pos_str}  heading={fa:.1f}°\n"
        f"walls={len(_walls)}  bots={len(_other_robots)}"
    )

    # ── Ball control text ─────────────────────────────────────────────────────
    if _game_state is not None:
        ctrl      = _game_state.get("ball_control")
        ctrl_team = _game_state.get("controlling_team")
        bp        = _game_state.get("ball")

        ball_str = f"({bp['x']:.2f}, {bp['y']:.2f})" if bp else "—"
        team_str = f"T{ctrl_team}" if ctrl_team is not None else "—"

        ctrl_str = "none"
        if ctrl is not None:
            cid   = ctrl.get("id")
            cteam = ctrl.get("team", "?")
            ctrl_str = f"self (T{cteam})" if cid is None else f"#{cid} (T{cteam})"

        _art_game_state.set_text(
            f"ball: {ball_str}\n"
            f"ctrl: {ctrl_str}  [{team_str}]"
        )
    else:
        _art_game_state.set_text("ball: —")

    # ── Blit ──────────────────────────────────────────────────────────────────
    fig.canvas.restore_region(_bg)
    for artist in [
        _art_lidar,
        _art_self, *_art_bots, *_art_blbls,
        _art_strategy_line, *_art_strategy_pts, *_art_strategy_lbls, *_art_strategy_arrows,
        _art_arrow,
        _art_ball, _art_ball_hidden, _art_ball_hist, _art_ball_arrow,
        _art_sim_ball, _art_sim_self, *_art_sim_obs,
        _art_ally_main, *_art_ally_det, _art_ally_ball,
        *_art_walls,
        _art_pos_hist, _art_bot_hist,
        _art_status,
    ]:
        ax.draw_artist(artist)
    fig.draw_artist(_art_game_state)
    fig.canvas.blit(fig.bbox)


# ── Broker callbacks ──────────────────────────────────────────────────────────
def on_update(key, value):
    global _lidar, _detection_origin, _detection_heading, _imu_pitch
    global _robot_pos, _other_robots, _walls
    global _position_history, _other_robots_history
    global _ball_pos, _ball_hidden_pos, _ball_lost, _ball_vx, _ball_vy, _ball_history
    global _sim_ball_pos, _sim_state, _ally_id, _ally_pos_raw
    global _raw_robots, _ball_raw, _game_state, _strategy_points

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
                if isinstance(payload, dict):
                    orig = payload.get("origin")
                    if orig:
                        _detection_origin  = (float(orig["x"]), float(orig["y"]))
                        if "heading" in orig:
                            _detection_heading = math.radians(float(orig["heading"]))
                    robot_list = payload.get("robots", [])
                else:
                    robot_list = payload
                _other_robots = [[float(r["x"]), float(r["y"]),
                                   r.get("method", ""), int(r.get("id", 0)),
                                   bool(r.get("ally", False))]
                                  for r in robot_list]

            elif key == "lidar_walls":
                _walls = json.loads(value)

            elif key == "position_history":
                _position_history = json.loads(value)

            elif key == "other_robots_history":
                _other_robots_history = json.loads(value)

            elif key == "ball":
                payload          = json.loads(value)
                _ball_pos        = payload.get("global_pos")
                _ball_hidden_pos = payload.get("hidden_pos")
                _ball_lost       = bool(payload.get("ball_lost", False))
                _ball_vx         = payload.get("vx")
                _ball_vy         = payload.get("vy")
                _sim_ball_pos    = payload.get("sim_pos")

            elif key == "ball_history":
                _ball_history = json.loads(value)

            elif key == "sim_state":
                _sim_state = json.loads(value)
            
            elif key == "raw_robots":
                _raw_robots = json.loads(value)
            
            elif key == "ball_raw":
                _ball_raw = json.loads(value)

            elif key == "game_state":
                _game_state = json.loads(value)

            elif key == "robot_strategy_points":
                _strategy_points = json.loads(value)

            elif key == "ally_id":
                try:
                    _ally_id = int(value) if value else None
                except (ValueError, TypeError):
                    _ally_id = None

            elif key in ("ally_main_robot_pos", "ally_other_pos_1",
                         "ally_other_pos_2", "ally_other_pos_3", "ally_ball_pos"):
                try:
                    p = json.loads(value)
                    _ally_pos_raw[key] = {"x": float(p["x"]), "y": float(p["y"])}
                except Exception:
                    _ally_pos_raw.pop(key, None)

    except Exception as e:
        print(f"[VIS] parse error on {key!r}: {e}")
        return

    _needs_redraw.set()


if __name__ == "__main__":
    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    _SEEDS = {
        "imu_pitch":            lambda v: float(v),
        "lidar":                lambda v: {int(k): int(x) for k, x in json.loads(v).items()},
        "robot_position":       lambda v: (float(json.loads(v)["x"]), float(json.loads(v)["y"])),
        "other_robots":         lambda v: [[float(r["x"]), float(r["y"]), r.get("method", ""), int(r.get("id", 0)), bool(r.get("ally", False))]
                                            for r in (lambda p: p.get("robots", p) if isinstance(p, dict) else p)(json.loads(v))],
        "lidar_walls":          lambda v: json.loads(v),
        "position_history":     lambda v: json.loads(v),
        "other_robots_history": lambda v: json.loads(v),
        "ball":                 lambda v: json.loads(v).get("global_pos"),
        "ball_history":         lambda v: json.loads(v),
        "sim_state":            lambda v: json.loads(v),
        "ally_id":              lambda v: int(v) if v and v.strip() else None,
        "raw_robots":           lambda v: json.loads(v),
        "ball_raw":             lambda v: json.loads(v),
        "game_state":           lambda v: json.loads(v),
        "robot_strategy_points":   lambda v: json.loads(v),
    }
    _TARGETS = {
        "imu_pitch":            "_imu_pitch",
        "lidar":                "_lidar",
        "robot_position":       "_robot_pos",
        "other_robots":         "_other_robots",
        "lidar_walls":          "_walls",
        "position_history":     "_position_history",
        "other_robots_history": "_other_robots_history",
        "ball":                 "_ball_pos",
        "ball_history":         "_ball_history",
        "sim_state":            "_sim_state",
        "ally_id":              "_ally_id",
        "raw_robots":           "_raw_robots",
        "ball_raw":             "_ball_raw",
        "game_state":           "_game_state",
        "robot_strategy_points":   "_strategy_points",
    }
    for key, parse in _SEEDS.items():
        try:
            val = mb.get(key)
            if val is not None:
                globals()[_TARGETS[key]] = parse(val)
        except Exception:
            pass

    plt.show(block=False)
    fig.canvas.draw()   # draws static background + fires draw_event → caches _bg
    _redraw()           # first blit pass

    _ally_keys = ["ally_main_robot_pos", "ally_other_pos_1",
                  "ally_other_pos_2", "ally_other_pos_3", "ally_ball_pos"]
    mb.setcallback(list(_SEEDS.keys()) + _ally_keys + ["robot_strategy_points"], on_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    try:
        while plt.fignum_exists(fig.number):
            if _needs_redraw.is_set():
                _needs_redraw.clear()
                with _state_lock:
                    with _perf.measure("redraw"):
                        _redraw()
            fig.canvas.flush_events()
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping twin visualisation.")
        plt.close(fig)
        mb.close()
