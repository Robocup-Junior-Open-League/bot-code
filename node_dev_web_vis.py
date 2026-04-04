"""
node_dev_web_vis — Browser-based live field visualisation.

Renders the field view using matplotlib's Agg (non-interactive) backend and
serves it over HTTP so any browser on the local network can see the live
state without needing a display.

Default URL:  http://localhost:5050/
Frame rate:   RENDER_HZ (default 10 fps)

Reads the same broker keys as node_dev_twin_vis:
  robot_position, other_robots, ball, field_sectors, ally_id, sim_state
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
from matplotlib.patches import FancyArrowPatch

from robus_core.libs.lib_telemtrybroker import TelemetryBroker

# ── Config ────────────────────────────────────────────────────────────────────
HOST       = "0.0.0.0"
PORT       = 5050
RENDER_HZ  = 10

# ── Field geometry (must match other nodes) ───────────────────────────────────
FIELD_WIDTH  = 1.58
FIELD_HEIGHT = 2.19
OUTER_MARGIN = 0.12
GOAL_WIDTH   = 0.60
ROBOT_RADIUS = 0.09
_MARGIN      = 0.10   # extra whitespace beyond outer walls in the plot

# ─────────────────────────────────────────────────────────────────────────────

mb = TelemetryBroker()

# ── Broker state (protected by _state_lock) ───────────────────────────────────
_state_lock    = threading.Lock()
_robot_pos     = None   # (x, y) — own robot
_other_robots  = []     # [[x, y, method, id, ally], ...]
_ball_pos      = None   # {"x": float, "y": float}
_ball_vx       = None
_ball_vy       = None
_ball_lost     = False
_field_sectors = None   # parsed field_sectors JSON from master node
_sim_state     = None   # {"robot": [x,y], "obstacles": [[x,y],...]}
_ally_id       = None

# ── Frame buffer (protected by _frame_lock) ───────────────────────────────────
_frame_lock  = threading.Lock()
_frame_bytes = b""

# ── Figure setup ─────────────────────────────────────────────────────────────
_OW  = OUTER_MARGIN
_GW2 = GOAL_WIDTH / 2
_GCX = FIELD_WIDTH / 2

fig, ax = plt.subplots(figsize=(6, 8.5), dpi=100)
fig.patch.set_facecolor("#1a1a1a")
ax.set_facecolor("#1a1a1a")
ax.set_aspect("equal")
ax.set_xlim(-_OW - _MARGIN, FIELD_WIDTH + _OW + _MARGIN)
ax.set_ylim(-_OW - _MARGIN, FIELD_HEIGHT + _OW + _MARGIN)
ax.tick_params(colors="#888888")
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")

# Reserve space below the axes for the game-state text
fig.subplots_adjust(bottom=0.14)

# ── Static background patches (drawn once, not cleared each frame) ────────────
# Outer wall area
ax.add_patch(patches.Rectangle(
    (-_OW, -_OW), FIELD_WIDTH + 2 * _OW, FIELD_HEIGHT + 2 * _OW,
    linewidth=1.5, edgecolor="#888888", facecolor="#2d2d2d", zorder=0))
# Inner playing field
ax.add_patch(patches.Rectangle(
    (0, 0), FIELD_WIDTH, FIELD_HEIGHT,
    linewidth=1, edgecolor="white", facecolor="#2d5a27", zorder=1))
# Goal — team 0 (bottom, yellow)
ax.add_patch(patches.Rectangle(
    (_GCX - _GW2, -_OW), GOAL_WIDTH, _OW,
    linewidth=0, facecolor="#ccaa00", zorder=2))
# Goal — team 1 (top, blue)
ax.add_patch(patches.Rectangle(
    (_GCX - _GW2, FIELD_HEIGHT), GOAL_WIDTH, _OW,
    linewidth=0, facecolor="#1155aa", zorder=2))

# ── Game-state text (below the axes in figure space) ─────────────────────────
_art_game_state = fig.text(
    0.5, 0.06, "", ha="center", va="top",
    fontsize=9, color="white",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333", edgecolor="#555555"))


# ── Render helpers ────────────────────────────────────────────────────────────

def _draw_robot(x, y, heading_rad=None, color="#aaddaa", edge="#2a7a2a",
                label=None, alpha=1.0):
    circle = plt.Circle((x, y), ROBOT_RADIUS, color=color, ec=edge,
                         lw=1.5, zorder=5, alpha=alpha)
    ax.add_patch(circle)
    if heading_rad is not None:
        dx = math.cos(heading_rad) * ROBOT_RADIUS
        dy = math.sin(heading_rad) * ROBOT_RADIUS
        ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.5),
                    zorder=6)
    if label:
        ax.text(x, y + ROBOT_RADIUS + 0.03, label, ha="center", va="bottom",
                fontsize=7, color="white", zorder=7)


def _render_frame():
    """Clear dynamic artists, redraw from current state, return PNG bytes."""
    # Remove all dynamic artists added in previous frame.
    # Static patches (field, goals) survive because they were added before
    # the first clear — we only remove artists whose zorder >= 4.
    for artist in list(ax.lines + ax.texts):
        artist.remove()
    for artist in list(ax.patches):
        if getattr(artist, "zorder", 0) >= 4:
            artist.remove()

    with _state_lock:
        robot_pos    = _robot_pos
        other_robots = list(_other_robots)
        ball_pos     = dict(_ball_pos) if _ball_pos else None
        ball_vx      = _ball_vx
        ball_vy      = _ball_vy
        ball_lost    = _ball_lost
        field_sectors = _field_sectors
        sim_state    = _sim_state
        ally_id      = _ally_id

    # ── Sim true position (faint white cross) ────────────────────────────────
    if sim_state is not None:
        srxy = sim_state.get("robot")
        if srxy:
            ax.plot(srxy[0], srxy[1], "+", color="white", ms=10,
                    mew=1.5, alpha=0.4, zorder=4)
        for obs in sim_state.get("obstacles", []):
            ax.plot(obs[0], obs[1], "x", color="#ff8888", ms=8,
                    mew=1.5, alpha=0.4, zorder=4)

    # ── Other robots ─────────────────────────────────────────────────────────
    for r in other_robots:
        rx, ry, method, rid, is_ally = r
        predicted = method == "predicted"
        is_our_team = (rid == ally_id)
        color = "#aaddaa" if is_our_team else "#ff8888"
        edge  = "#2a7a2a" if is_our_team else "#aa2222"
        alpha = 0.5 if predicted else 1.0
        label = f"#{rid}"
        _draw_robot(rx, ry, color=color, edge=edge, label=label, alpha=alpha)

    # ── Own robot ─────────────────────────────────────────────────────────────
    if robot_pos is not None:
        _draw_robot(robot_pos[0], robot_pos[1],
                    color="#aaddaa", edge="#2a7a2a", label="self")

    # ── Ball ─────────────────────────────────────────────────────────────────
    if ball_pos is not None:
        bx, by = ball_pos["x"], ball_pos["y"]
        alpha  = 0.45 if ball_lost else 1.0
        ball_c = plt.Circle((bx, by), 0.025, color="orange",
                             ec="darkorange", lw=1, zorder=6, alpha=alpha)
        ax.add_patch(ball_c)
        # Velocity arrow
        if ball_vx is not None and ball_vy is not None:
            speed = math.hypot(ball_vx, ball_vy)
            if speed > 0.05:
                scale = min(speed * 0.15, 0.25)
                ax.annotate("",
                    xy=(bx + ball_vx / speed * scale,
                        by + ball_vy / speed * scale),
                    xytext=(bx, by),
                    arrowprops=dict(arrowstyle="-|>", color="orange", lw=1.5),
                    zorder=7)

    # ── Game-state text ───────────────────────────────────────────────────────
    if field_sectors is not None:
        gs      = field_sectors.get("game_state") or {}
        ctrl    = field_sectors.get("ball_control")

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
            f"{strength_str} {state_str}  {team_str}  ·  {side_str}  ·  {substate_str}"
            f"    ctrl: {ctrl_str}"
        )
    else:
        _art_game_state.set_text("game state: —")

    # ── Encode to PNG ─────────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()


# ── Render loop ───────────────────────────────────────────────────────────────

def _render_loop():
    global _frame_bytes
    interval = 1.0 / RENDER_HZ
    while True:
        t0 = time.monotonic()
        try:
            png = _render_frame()
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
    body  { margin: 0; background: #1a1a1a; display: flex;
            flex-direction: column; align-items: center; }
    img   { max-width: 100vw; max-height: 100vh; object-fit: contain; }
    #fps  { color: #888; font: 11px monospace; margin: 4px; }
  </style>
</head>
<body>
  <div id="fps">connecting…</div>
  <img id="frame" src="/frame.png" alt="field">
  <script>
    const img   = document.getElementById('frame');
    const fps_el = document.getElementById('fps');
    let last = performance.now(), frames = 0;

    function refresh() {
      const url = '/frame.png?t=' + Date.now();
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
        setTimeout(refresh, {interval_ms});
      };
      next.onerror = () => setTimeout(refresh, 500);
      next.src = url;
    }
    refresh();
  </script>
</body>
</html>
""".replace("{interval_ms}", str(int(1000 / RENDER_HZ)))


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
    global _robot_pos, _other_robots, _ball_pos, _ball_vx, _ball_vy
    global _ball_lost, _field_sectors, _sim_state, _ally_id

    if value is None:
        return
    try:
        with _state_lock:
            if key == "robot_position":
                p = json.loads(value)
                _robot_pos = (float(p["x"]), float(p["y"]))

            elif key == "other_robots":
                payload = json.loads(value)
                robot_list = payload.get("robots", []) if isinstance(payload, dict) else payload
                _other_robots = [
                    [float(r["x"]), float(r["y"]),
                     r.get("method", ""), int(r.get("id", 0)),
                     bool(r.get("ally", False))]
                    for r in robot_list
                ]

            elif key == "ball":
                payload    = json.loads(value)
                _ball_pos  = payload.get("global_pos")
                _ball_lost = bool(payload.get("ball_lost", False))
                _ball_vx   = payload.get("vx")
                _ball_vy   = payload.get("vy")

            elif key == "field_sectors":
                _field_sectors = json.loads(value)

            elif key == "sim_state":
                _sim_state = json.loads(value)

            elif key == "ally_id":
                _ally_id = int(value) if value and value.strip() else None

    except Exception as e:
        print(f"[WEB-VIS] parse error on {key!r}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Seed initial state from broker
    for key in ("robot_position", "other_robots", "ball",
                "field_sectors", "sim_state", "ally_id"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(
        ["robot_position", "other_robots", "ball",
         "field_sectors", "sim_state", "ally_id"],
        on_update)

    # Start render loop in background
    t = threading.Thread(target=_render_loop, daemon=True)
    t.start()

    # Start broker listener in background
    broker_thread = threading.Thread(target=mb.receiver_loop, daemon=True)
    broker_thread.start()

    print(f"[WEB-VIS] Serving at http://{HOST}:{PORT}/  ({RENDER_HZ} fps)")
    server = HTTPServer((HOST, PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[WEB-VIS] Stopped.")
        mb.close()
