"""
node_dev_web_vis — Browser-based live field visualisation.

Renders exactly the same view as node_dev_twin_vis using matplotlib's Agg
(non-interactive) backend and serves it over HTTP so any browser on the
local network can follow the live state without needing a display.

Default URL:  http://localhost:5050/
Frame rate:   RENDER_HZ (default 10 fps)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import io
import json
import math
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np

from robus_core.libs.lib_telemtrybroker import TelemetryBroker

# ── Config ────────────────────────────────────────────────────────────────────
HOST      = "0.0.0.0"
PORT      = 5050
RENDER_HZ = 10

# ── Field configuration (must match other nodes) ──────────────────────────────
FIELD_WIDTH  = 1.58
FIELD_HEIGHT = 2.19
ROBOT_RADIUS = 0.09
OUTER_MARGIN = 0.12
GOAL_WIDTH   = 0.60
_MARGIN         = 0.10
_MAX_VIS_ROBOTS = 3
_MAX_WALLS      = 4

# ─────────────────────────────────────────────────────────────────────────────

mb = TelemetryBroker()

# ── Broker state ──────────────────────────────────────────────────────────────
_state_lock           = threading.Lock()
_lidar                = {}
_detection_origin     = None
_detection_heading    = None
_imu_pitch            = None
_robot_pos            = None
_other_robots         = []
_walls                = []
_position_history     = []
_other_robots_history = []
_ball_pos             = None
_ball_hidden_pos      = None
_ball_lost            = False
_ball_vx              = None
_ball_vy              = None
_ball_history         = []
_sim_ball_pos         = None
_sim_state            = None
_ally_id              = None
_ally_pos_raw         = {}
_raw_robots           = None
_ball_raw             = None
_field_sectors        = None


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


# ── Matplotlib setup ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 9))
plt.tight_layout(pad=1.5)

_OW = OUTER_MARGIN
ax.set_xlim(-_OW - _MARGIN, FIELD_WIDTH  + _OW + _MARGIN)
ax.set_ylim(-_OW - _MARGIN, FIELD_HEIGHT + _OW + _MARGIN)
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.grid(True, alpha=0.25, zorder=0)

# ── Static background ─────────────────────────────────────────────────────────
_gx_min = (FIELD_WIDTH - GOAL_WIDTH) / 2
_gx_max = (FIELD_WIDTH + GOAL_WIDTH) / 2

ax.add_patch(patches.Rectangle(
    (-_OW, -_OW), FIELD_WIDTH + 2 * _OW, FIELD_HEIGHT + 2 * _OW,
    linewidth=2, edgecolor='#222222', facecolor='#cccccc', zorder=1))

ax.add_patch(patches.Rectangle(
    (0, 0), FIELD_WIDTH, FIELD_HEIGHT,
    linewidth=1, edgecolor='white', facecolor='#c8e6c9', zorder=2))

ax.add_patch(patches.Rectangle(
    (_gx_min, -_OW), GOAL_WIDTH, _OW,
    linewidth=0, facecolor='#ffee44', zorder=3))

ax.add_patch(patches.Rectangle(
    (_gx_min, FIELD_HEIGHT), GOAL_WIDTH, _OW,
    linewidth=0, facecolor='#4488ff', zorder=3))

for _gx in (_gx_min, _gx_max):
    ax.add_line(plt.Line2D([_gx, _gx], [-_OW, 0.0],
                           color='#222222', lw=2, zorder=4))
    ax.add_line(plt.Line2D([_gx, _gx], [FIELD_HEIGHT, FIELD_HEIGHT + _OW],
                           color='#222222', lw=2, zorder=4))

# ── Pre-allocated dynamic artists ─────────────────────────────────────────────
# (No animated=True — Agg renders everything in fig.canvas.draw())

_TAB10 = [tuple(plt.cm.tab10(i)[:3]) for i in range(10)]

_art_lidar = ax.scatter([], [], s=5, c='#222222', zorder=10)

_art_self = patches.Circle((0, 0), ROBOT_RADIUS,
    lw=1.5, edgecolor='#2a7a2a', facecolor='#aaddaa',
    zorder=7, visible=False)
ax.add_patch(_art_self)

_art_bots  = []
_art_blbls = []
for _ in range(_MAX_VIS_ROBOTS):
    c = patches.Circle((0, 0), ROBOT_RADIUS,
        lw=1.5, edgecolor='gray', facecolor='lightgray',
        zorder=7, visible=False)
    ax.add_patch(c)
    t = ax.text(0, 0, '', ha='center', va='bottom', fontsize=8,
        fontweight='bold', visible=False, zorder=8)
    _art_bots.append(c)
    _art_blbls.append(t)

_art_arrow = FancyArrowPatch((0, 0), (0.1, 0),
    arrowstyle='->', color='#2a7a2a', lw=2.0, mutation_scale=15,
    zorder=8, visible=False)
ax.add_patch(_art_arrow)

_WSPAN_X = (-_OW - _MARGIN, FIELD_WIDTH  + _OW + _MARGIN)
_WSPAN_Y = (-_OW - _MARGIN, FIELD_HEIGHT + _OW + _MARGIN)

def _wall_line(color, ls):
    (ln,) = ax.plot([], [], color=color, lw=1.5, ls=ls,
        zorder=4, visible=False)
    return ln

_art_walls = [_wall_line('steelblue', '--') for _ in range(_MAX_WALLS)]

_art_pos_hist = ax.scatter([], [], s=18, zorder=5, edgecolors='none')
_art_bot_hist = ax.scatter([], [], s=18, zorder=5, edgecolors='none')

_BALL_RADIUS = 0.021
_art_ball = patches.Circle((0, 0), _BALL_RADIUS,
    lw=1.5, edgecolor='darkorange', facecolor='orange',
    zorder=9, visible=False)
ax.add_patch(_art_ball)

_art_ball_hist = ax.scatter([], [], s=14, zorder=5, edgecolors='none')

_art_ball_arrow = FancyArrowPatch((0, 0), (0.1, 0),
    arrowstyle='->', color='darkorange', lw=1.5, mutation_scale=10,
    zorder=9, visible=False)
ax.add_patch(_art_ball_arrow)

_art_ball_hidden = patches.Circle((0, 0), _BALL_RADIUS,
    lw=1.5, edgecolor='darkorange', facecolor=(1.0, 0.55, 0.0, 0.25), ls='--',
    zorder=8, visible=False)
ax.add_patch(_art_ball_hidden)

def _sim_cross(color):
    (art,) = ax.plot([], [], marker='+', markersize=12, markeredgewidth=2,
                     color=color, ls='none', zorder=8)
    return art

_art_sim_ball = _sim_cross('darkorange')
_art_sim_self = _sim_cross('#2a7a2a')
_art_sim_obs  = [_sim_cross((0.90, 0.22, 0.18)) for _ in range(3)]

_ALLY_BLUE = (0.13, 0.53, 0.90)
_MAX_ALLY_OTHERS = 3

def _ally_marker(marker):
    (art,) = ax.plot([], [], marker=marker, markersize=10, markeredgewidth=2,
                     color=_ALLY_BLUE, ls='none', zorder=9)
    return art

_art_ally_main = _ally_marker('D')
_art_ally_det  = [_ally_marker('x') for _ in range(_MAX_ALLY_OTHERS)]
_art_ally_ball = _ally_marker('*')

_art_status = ax.text(0.01, 0.99, '', transform=ax.transAxes,
    ha='left', va='top', fontsize=8, zorder=15,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
              alpha=0.75, edgecolor='none'))

_art_game_state = ax.text(1.02, 0.055, '', transform=ax.transAxes,
    ha='left', va='top', fontsize=12, zorder=15, clip_on=False,
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
]

ax.legend(handles=_legend_handles,
          loc='upper left',
          bbox_to_anchor=(1.02, 1.0),
          borderaxespad=0.)

# ── Frame buffer ──────────────────────────────────────────────────────────────
_frame_lock  = threading.Lock()
_frame_bytes = b""
_render_lock = threading.Lock()   # serialises fig.canvas.draw() calls


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


# ── Redraw (mirrors node_dev_twin_vis._redraw exactly) ────────────────────────

def _redraw():
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
        for art in _art_sim_obs:
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
                c   = _ALLY_BLUE if (_ally_id is not None and rid == _ally_id) else (0.90, 0.22, 0.18)
                pts.append((r["x"], r["y"]))
                rgba.append(c + (alpha,))
        if pts:
            _art_bot_hist.set_offsets(np.array(pts))
            _art_bot_hist.set_facecolors(rgba)
        else:
            _art_bot_hist.set_offsets(np.empty((0, 2)))
    else:
        _art_bot_hist.set_offsets(np.empty((0, 2)))

    # ── Status text ───────────────────────────────────────────────────────────
    pos_str = f"({origin[0]:.2f}, {origin[1]:.2f})" if known_pos else "unknown"
    _art_status.set_text(
        f"pos={pos_str}  heading={fa:.1f}°\n"
        f"walls={len(_walls)}  bots={len(_other_robots)}"
    )

    # ── Game state text ───────────────────────────────────────────────────────
    if _field_sectors is not None:
        gs   = _field_sectors.get("game_state") or {}
        ctrl = _field_sectors.get("ball_control")

        state_str    = gs.get("state")    or "—"
        strength_str = gs.get("strength") or ""
        team_val     = gs.get("team")
        side_str     = gs.get("side")     or "—"
        substate_str = gs.get("substate") or "—"
        team_str     = f"T{team_val}" if team_val is not None else "—"

        ctrl_str = "none"
        if ctrl is not None:
            cid   = ctrl.get("id")
            cteam = ctrl.get("team", "?")
            ctrl_str = f"self (T{cteam})" if cid is None else f"#{cid} (T{cteam})"

        _art_game_state.set_text(
            f"{strength_str} {state_str}  {team_str}  ·  {side_str}  ·  {substate_str}\n"
            f"ctrl: {ctrl_str}"
        )
    else:
        _art_game_state.set_text("game state: —")

    # ── Render to PNG ─────────────────────────────────────────────────────────
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()


# ── Render loop ───────────────────────────────────────────────────────────────

def _render_loop():
    global _frame_bytes
    interval = 1.0 / RENDER_HZ
    while True:
        t0 = time.monotonic()
        try:
            with _render_lock:
                with _state_lock:
                    png = _redraw()
            with _frame_lock:
                _frame_bytes = png
        except Exception as e:
            print(f"[WEB-VIS] render error: {e}")
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))


# ── HTTP server ───────────────────────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Field Visualisation</title>
  <style>
    body { margin: 0; background: #111; display: flex;
           flex-direction: column; align-items: center; }
    img  { max-width: 100vw; max-height: 100vh; object-fit: contain; }
    #fps { color: #888; font: 11px monospace; margin: 4px; }
  </style>
</head>
<body>
  <div id="fps">connecting…</div>
  <img id="frame" src="/frame.png" alt="field">
  <script>
    const img    = document.getElementById('frame');
    const fps_el = document.getElementById('fps');
    let last = performance.now(), frames = 0;

    function refresh() {
      const next = new Image();
      next.onload = () => {
        img.src = next.src;
        frames++;
        const now = performance.now();
        if (now - last >= 1000) {
          fps_el.textContent = frames + ' fps';
          frames = 0;
          last = now;
        }
        setTimeout(refresh, INTERVAL_MS);
      };
      next.onerror = () => setTimeout(refresh, 500);
      next.src = '/frame.png?t=' + Date.now();
    }
    refresh();
  </script>
</body>
</html>
""".replace("INTERVAL_MS", str(int(1000 / RENDER_HZ)))


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request console noise

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/frame.png":
            with _frame_lock:
                body = _frame_bytes
            if not body:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _lidar, _detection_origin, _detection_heading, _imu_pitch
    global _robot_pos, _other_robots, _walls
    global _position_history, _other_robots_history
    global _ball_pos, _ball_hidden_pos, _ball_lost, _ball_vx, _ball_vy, _ball_history
    global _sim_ball_pos, _sim_state, _ally_id, _ally_pos_raw
    global _raw_robots, _ball_raw, _field_sectors

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

            elif key == "field_sectors":
                _field_sectors = json.loads(value)

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
        print(f"[WEB-VIS] parse error on {key!r}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _SEEDS = {
        "imu_pitch":            lambda v: float(v),
        "lidar":                lambda v: {int(k): int(x) for k, x in json.loads(v).items()},
        "robot_position":       lambda v: (float(json.loads(v)["x"]), float(json.loads(v)["y"])),
        "other_robots":         lambda v: [[float(r["x"]), float(r["y"]), r.get("method", ""), int(r.get("id", 0)), bool(r.get("ally", False))]
                                            for r in (lambda p: p.get("robots", p) if isinstance(p, dict) else p)(json.loads(v))],
        "lidar_walls":          lambda v: json.loads(v),
        "position_history":     lambda v: json.loads(v),
        "other_robots_history": lambda v: json.loads(v),
        "ball":                 lambda v: json.loads(v),
        "ball_history":         lambda v: json.loads(v),
        "sim_state":            lambda v: json.loads(v),
        "ally_id":              lambda v: int(v) if v and v.strip() else None,
        "raw_robots":           lambda v: json.loads(v),
        "ball_raw":             lambda v: json.loads(v),
        "field_sectors":        lambda v: json.loads(v),
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
        "field_sectors":        "_field_sectors",
    }
    for key, parse in _SEEDS.items():
        try:
            val = mb.get(key)
            if val is not None:
                globals()[_TARGETS[key]] = parse(val)
        except Exception:
            pass

    _ally_keys = ["ally_main_robot_pos", "ally_other_pos_1",
                  "ally_other_pos_2", "ally_other_pos_3", "ally_ball_pos"]
    mb.setcallback(list(_SEEDS.keys()) + _ally_keys, on_update)

    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()
    threading.Thread(target=_render_loop,     daemon=True,
                     name="render").start()

    print(f"[WEB-VIS] Serving at http://{HOST}:{PORT}/  ({RENDER_HZ} fps)")
    server = HTTPServer((HOST, PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[WEB-VIS] Stopped.")
        mb.close()
