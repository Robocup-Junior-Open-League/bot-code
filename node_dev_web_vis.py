"""
node_dev_web_vis — Browser-based live field visualisation.

State is pushed to the browser via Server-Sent Events (SSE) as compact JSON.
The browser renders everything with HTML5 Canvas — no matplotlib, no PNG
encoding, no image transfer.  Frame rate is limited only by the browser's
requestAnimationFrame and the rate at which the broker emits updates.

Default URL:  http://localhost:5050/
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import math
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor

# ── Config ────────────────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 5050

# ─────────────────────────────────────────────────────────────────────────────
mb = TelemetryBroker()
_perf = PerfMonitor("node_dev_web_vis", broker=mb, print_every=100)

# ── Broker state (same variables as node_dev_twin_vis) ────────────────────────
_state_lock           = threading.Lock()
_lidar                = {}
_detection_origin     = None
_detection_heading    = None   # radians
_imu_pitch            = None   # degrees
_robot_pos            = None   # (x, y)
_other_robots         = []     # [[x,y,method,id,ally], ...]
_walls                = []
_position_history     = []
_other_robots_history = []
_ball_pos             = None
_ball_hidden_pos      = None
_ball_lost            = False
_ball_vx              = None
_ball_vy              = None
_ball_history         = []
_ally_id              = None
_ally_pos_raw         = {}
_raw_robots           = None
_ball_raw             = None
_game_state        = None
_strategy_points      = []

# ── SSE push ──────────────────────────────────────────────────────────────────
_push_cond = threading.Condition()
_push_seq  = 0

def _notify():
    global _push_seq
    with _push_cond:
        _push_seq += 1
        _push_cond.notify_all()

def _build_state():
    """Snapshot current state as a JSON string (called under _state_lock)."""
    with _perf.measure("serialize"):
        return json.dumps({
            "robot_pos":            list(_robot_pos) if _robot_pos else None,
            "heading":              _imu_pitch,
            "detection_origin":     list(_detection_origin) if _detection_origin else None,
            "detection_heading":    _detection_heading,
            "other_robots":         _other_robots,
            "ally_id":              _ally_id,
            "lidar":                _lidar,
            "walls":                _walls,
            "ball_pos":             _ball_pos,
            "ball_hidden_pos":      _ball_hidden_pos,
            "ball_lost":            _ball_lost,
            "ball_vx":              _ball_vx,
            "ball_vy":              _ball_vy,
            "ball_history":         _ball_history,
            "position_history":     _position_history,
            "other_robots_history": _other_robots_history,
            "raw_robots":           _raw_robots,
            "ball_raw":             _ball_raw,
            "game_state":        _game_state,
            "ally_pos_raw":         _ally_pos_raw,
            "strategy_points":      _strategy_points,
        })

# ── HTML page (served once; all rendering is client-side) ─────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Field Vis</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; display: flex; align-items: flex-start;
       gap: 10px; padding: 8px; font-family: monospace; overflow: hidden; }
#canvas-col { display: flex; flex-direction: column; gap: 4px; flex-shrink: 0; }
canvas { display: block; background: white; }
#fps { color: #666; font-size: 11px; }
#side { display: flex; flex-direction: column; gap: 8px; min-width: 175px; max-width: 200px; }
#gs { background: #f5f5f5; border: 1px solid #aaa; border-radius: 4px;
      padding: 6px 8px; color: #222; font-size: 12px; white-space: pre-line; line-height: 1.5; }
#legend { background: #f5f5f5; border: 1px solid #aaa; border-radius: 4px;
          padding: 6px 8px; color: #222; font-size: 11px; }
#legend h4 { font-size: 11px; color: #555; margin-bottom: 4px; }
.li { display: flex; align-items: center; gap: 5px; margin: 2px 0; line-height: 1.3; }
.sw { flex-shrink: 0; }
</style>
</head>
<body>
<div id="canvas-col">
  <canvas id="c"></canvas>
  <div id="fps">connecting…</div>
</div>
<div id="side">
  <div id="gs">game state: —</div>
  <div id="legend">
    <h4>Legend</h4>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="6" fill="#aaddaa" stroke="#2a7a2a" stroke-width="1.5"/></svg>Own robot</div>
    <div class="li"><svg class="sw" width="22" height="14"><line x1="2" y1="7" x2="18" y2="7" stroke="#2a7a2a" stroke-width="2"/><polygon points="18,4 22,7 18,10" fill="#2a7a2a"/></svg>Heading</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="3" fill="#aaddaa"/></svg>Own history</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="6" fill="rgba(33,135,230,0.3)" stroke="rgb(33,135,230)" stroke-width="1.5"/></svg>Ally robot</div>
    <div class="li"><svg class="sw" width="22" height="14"><polygon points="11,2 18,7 11,12 4,7" fill="none" stroke="rgb(33,135,230)" stroke-width="2"/></svg>Ally position</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="3" fill="rgb(33,135,230)"/></svg>Ally history</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="6" fill="rgba(230,56,46,0.3)" stroke="rgb(230,56,46)" stroke-width="1.5"/></svg>Enemy robot</div>
    <div class="li"><svg class="sw" width="22" height="14"><line x1="11" y1="2" x2="11" y2="12" stroke="red" stroke-width="2"/><line x1="6" y1="7" x2="16" y2="7" stroke="red" stroke-width="2"/></svg>Own detections</div>
    <div class="li"><svg class="sw" width="22" height="14"><line x1="4" y1="4" x2="18" y2="10" stroke="rgb(33,135,230)" stroke-width="2"/><line x1="18" y1="4" x2="4" y2="10" stroke="rgb(33,135,230)" stroke-width="2"/></svg>Ally detections</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="3" fill="rgb(230,56,46)"/></svg>Enemy history</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="5" fill="orange" stroke="darkorange" stroke-width="1.5"/></svg>Ball</div>
    <div class="li"><svg class="sw" width="22" height="14"><line x1="2" y1="7" x2="17" y2="7" stroke="darkorange" stroke-width="1.5"/><polygon points="17,4 22,7 17,10" fill="darkorange"/></svg>Ball velocity</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="5" fill="rgba(255,140,0,0.25)" stroke="darkorange" stroke-width="1.5" stroke-dasharray="3,2"/></svg>Ball (extrapolated)</div>
    <div class="li"><svg class="sw" width="22" height="14"><line x1="11" y1="2" x2="11" y2="12" stroke="darkorange" stroke-width="2"/><line x1="6" y1="7" x2="16" y2="7" stroke="darkorange" stroke-width="2"/></svg>Own ball det.</div>
    <div class="li"><svg class="sw" width="22" height="14"><text x="11" y="11" text-anchor="middle" fill="rgb(33,135,230)" font-size="14">★</text></svg>Ally ball det.</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="3" fill="darkorange"/></svg>Ball history</div>
    <div class="li"><svg class="sw" width="22" height="14"><line x1="2" y1="7" x2="20" y2="7" stroke="steelblue" stroke-width="1.5" stroke-dasharray="4,3"/></svg>Walls</div>
    <div class="li"><svg class="sw" width="22" height="14"><circle cx="11" cy="7" r="2.5" fill="#222"/></svg>Lidar</div>
  </div>
</div>

<script>
(function () {
'use strict';

// ── Field constants (must match Python nodes) ─────────────────────────────────
const FW = 1.58, FH = 2.19, OW = 0.12, MAR = 0.10;
const GW = 0.60, RR = 0.09, BR = 0.021;
const GX1 = (FW - GW) / 2, GX2 = (FW + GW) / 2;
const XMIN = -OW - MAR, XMAX = FW + OW + MAR;
const YMIN = -OW - MAR, YMAX = FH + OW + MAR;
const VWID = XMAX - XMIN, VHGT = YMAX - YMIN;
const ALLY_RGB  = '33,135,230';
const ENEMY_RGB = '230,56,46';
const OWN_EDGE  = '#2a7a2a';
const OWN_FACE  = '#aaddaa';

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
let W = 1, H = 1, SF = 1;

function resize() {
    const maxH = window.innerHeight - 32;
    H = maxH;
    W = Math.round(H * VWID / VHGT);
    canvas.width  = W;
    canvas.height = H;
    SF = W / VWID;
}
window.addEventListener('resize', resize);
resize();

// ── Coordinate helpers ────────────────────────────────────────────────────────
function cx(x)    { return (x - XMIN) * SF; }
function cy(y)    { return H - (y - YMIN) * SF; }
function sc(m)    { return m * SF; }

// Rectangle from field bottom-left to top-right
function frect(x1, y1, x2, y2) {
    return [cx(x1), cy(y2), sc(x2 - x1), sc(y2 - y1)];
}

// ── Draw helpers ──────────────────────────────────────────────────────────────
function arrow(x1, y1, x2, y2, color, lw) {
    const [ax1, ay1] = [cx(x1), cy(y1)];
    const [ax2, ay2] = [cx(x2), cy(y2)];
    const dx = ax2 - ax1, dy = ay2 - ay1;
    const len = Math.hypot(dx, dy);
    if (len < 1) return;
    const nx = dx / len, ny = dy / len;
    const hs = Math.max(sc(0.028), 5);
    ctx.beginPath();
    ctx.moveTo(ax1, ay1);
    ctx.lineTo(ax2 - nx * hs * 0.7, ay2 - ny * hs * 0.7);
    ctx.strokeStyle = color; ctx.lineWidth = lw || 2; ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ax2, ay2);
    ctx.lineTo(ax2 - nx*hs - ny*hs*0.4, ay2 - ny*hs + nx*hs*0.4);
    ctx.lineTo(ax2 - nx*hs + ny*hs*0.4, ay2 - ny*hs - nx*hs*0.4);
    ctx.closePath();
    ctx.fillStyle = color; ctx.fill();
}

function cross(x, y, size, color, lw) {
    const [pcx, pcy] = [cx(x), cy(y)];
    const s = sc(size);
    ctx.strokeStyle = color; ctx.lineWidth = lw || 2;
    ctx.beginPath();
    ctx.moveTo(pcx - s, pcy); ctx.lineTo(pcx + s, pcy);
    ctx.moveTo(pcx, pcy - s); ctx.lineTo(pcx, pcy + s);
    ctx.stroke();
}

function xcross(x, y, size, color, lw) {
    const [pcx, pcy] = [cx(x), cy(y)];
    const s = sc(size);
    ctx.strokeStyle = color; ctx.lineWidth = lw || 2;
    ctx.beginPath();
    ctx.moveTo(pcx - s, pcy - s); ctx.lineTo(pcx + s, pcy + s);
    ctx.moveTo(pcx + s, pcy - s); ctx.lineTo(pcx - s, pcy + s);
    ctx.stroke();
}

function diamond(x, y, size, color, lw) {
    const [pcx, pcy] = [cx(x), cy(y)];
    const s = sc(size);
    ctx.strokeStyle = color; ctx.lineWidth = lw || 2;
    ctx.beginPath();
    ctx.moveTo(pcx, pcy - s);
    ctx.lineTo(pcx + s * 0.65, pcy);
    ctx.lineTo(pcx, pcy + s);
    ctx.lineTo(pcx - s * 0.65, pcy);
    ctx.closePath();
    ctx.stroke();
}

function star(x, y, size, color) {
    const [pcx, pcy] = [cx(x), cy(y)];
    ctx.font = `${sc(size * 2.5)}px sans-serif`;
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('★', pcx, pcy);
}

function dot(x, y, r, fill) {
    ctx.beginPath();
    ctx.arc(cx(x), cy(y), r, 0, Math.PI * 2);
    ctx.fillStyle = fill; ctx.fill();
}

// ── Static field ──────────────────────────────────────────────────────────────
function drawField() {
    // Grid
    ctx.strokeStyle = 'rgba(0,0,0,0.25)'; ctx.lineWidth = 0.5; ctx.setLineDash([]);
    for (let gx = -0.5; gx <= FW + OW + MAR; gx += 0.5) {
        ctx.beginPath(); ctx.moveTo(cx(gx), 0); ctx.lineTo(cx(gx), H); ctx.stroke();
    }
    for (let gy = -0.5; gy <= FH + OW + MAR; gy += 0.5) {
        ctx.beginPath(); ctx.moveTo(0, cy(gy)); ctx.lineTo(W, cy(gy)); ctx.stroke();
    }

    // Outer wall area
    ctx.fillStyle = '#cccccc'; ctx.strokeStyle = '#222'; ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.fillRect(...frect(-OW, -OW, FW + OW, FH + OW));
    ctx.strokeRect(...frect(-OW, -OW, FW + OW, FH + OW));

    // Playing field
    ctx.fillStyle = '#c8e6c9'; ctx.strokeStyle = 'white'; ctx.lineWidth = 1;
    ctx.fillRect(...frect(0, 0, FW, FH));
    ctx.strokeRect(...frect(0, 0, FW, FH));

    // Bottom goal (yellow)
    ctx.fillStyle = '#ffee44'; ctx.fillRect(...frect(GX1, -OW, GX2, 0));
    // Top goal (blue)
    ctx.fillStyle = '#4488ff'; ctx.fillRect(...frect(GX1, FH, GX2, FH + OW));

    // Goal side walls
    ctx.strokeStyle = '#222'; ctx.lineWidth = 2;
    for (const gx of [GX1, GX2]) {
        ctx.beginPath(); ctx.moveTo(cx(gx), cy(-OW));  ctx.lineTo(cx(gx), cy(0));       ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx(gx), cy(FH));   ctx.lineTo(cx(gx), cy(FH + OW)); ctx.stroke();
    }
}

// ── Walls ─────────────────────────────────────────────────────────────────────
function drawWalls(s) {
    if (!s.walls || !s.walls.length || !s.robot_pos) return;
    const [ox, oy] = s.robot_pos;
    ctx.strokeStyle = 'steelblue'; ctx.lineWidth = 1.5; ctx.setLineDash([8, 4]);
    for (const w of s.walls) {
        if (w.gradient === 0) {
            const wy = oy + w.offset;
            ctx.beginPath(); ctx.moveTo(0, cy(wy)); ctx.lineTo(W, cy(wy)); ctx.stroke();
        } else {
            const wx = ox + w.offset;
            ctx.beginPath(); ctx.moveTo(cx(wx), 0); ctx.lineTo(cx(wx), H); ctx.stroke();
        }
    }
    ctx.setLineDash([]);
}

// ── Lidar ─────────────────────────────────────────────────────────────────────
function drawLidar(s) {
    if (!s.lidar) return;
    const origin = s.detection_origin || s.robot_pos;
    if (!origin) return;
    const lh = s.detection_heading != null ? s.detection_heading
             : ((s.heading || 0) * Math.PI / 180);
    ctx.fillStyle = '#222222';
    ctx.beginPath();
    for (const [angStr, distMm] of Object.entries(s.lidar)) {
        const a = parseFloat(angStr) * Math.PI / 180 + lh;
        const d = distMm / 1000;
        const pcx = cx(origin[0] + d * Math.cos(a));
        const pcy = cy(origin[1] + d * Math.sin(a));
        ctx.moveTo(pcx + 2.5, pcy);
        ctx.arc(pcx, pcy, 2.5, 0, Math.PI * 2);
    }
    ctx.fill();
}

// ── History trails ────────────────────────────────────────────────────────────
function drawTrail(pts, r, rgbStr, alphaScale) {
    if (!pts || pts.length < 2) return;
    const t0 = pts[0].t, tMax = pts[pts.length - 1].t;
    const rng = Math.max(tMax - t0, 1e-9);
    for (const p of pts) {
        const a = 0.05 + (alphaScale || 0.7) * (p.t - t0) / rng;
        dot(p.x, p.y, r, `rgba(${rgbStr},${a})`);
    }
}

function drawRobotHistory(s) {
    if (!s.other_robots_history || !s.other_robots_history.length) return;
    const snaps = s.other_robots_history;
    const t0 = snaps[0].t, tMax = snaps[snaps.length - 1].t;
    const rng = Math.max(tMax - t0, 1e-9);
    for (const snap of snaps) {
        const a = 0.05 + 0.6 * (snap.t - t0) / rng;
        for (const r of snap.robots) {
            const isAlly = s.ally_id != null && parseInt(r.id) === s.ally_id;
            dot(r.x, r.y, sc(0.018), `rgba(${isAlly ? ALLY_RGB : ENEMY_RGB},${a})`);
        }
    }
}

// ── Robots ────────────────────────────────────────────────────────────────────
function drawRobot(x, y, face, edge, dashed, alpha) {
    const [pcx, pcy] = [cx(x), cy(y)];
    const r = sc(RR);
    ctx.globalAlpha = alpha != null ? alpha : 1;
    ctx.beginPath(); ctx.arc(pcx, pcy, r, 0, Math.PI * 2);
    if (dashed) ctx.setLineDash([sc(0.02), sc(0.02)]);
    ctx.fillStyle = face; ctx.fill();
    ctx.strokeStyle = edge; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;
}

function drawOtherRobots(s) {
    if (!s.other_robots) return;
    for (const r of s.other_robots) {
        const [rx, ry, method, id, ] = r;
        const rid     = parseInt(id);
        const isAlly  = s.ally_id != null && rid === s.ally_id;
        const pred    = method === 'predicted';
        const rgb     = isAlly ? ALLY_RGB : ENEMY_RGB;
        const faceA   = pred ? 0.15 : 0.3;
        drawRobot(rx, ry, `rgba(${rgb},${faceA})`, `rgb(${rgb})`, pred, pred ? 0.6 : 1.0);
        // Label
        ctx.globalAlpha = pred ? 0.5 : 1;
        ctx.fillStyle = `rgb(${rgb})`;
        ctx.font = 'bold 12px monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText(`#${rid}`, cx(rx), cy(ry + RR + 0.03));
        ctx.globalAlpha = 1;
    }
}

function drawOwnRobot(s) {
    if (!s.robot_pos) return;
    const [rx, ry] = s.robot_pos;
    drawRobot(rx, ry, OWN_FACE, OWN_EDGE);
    // Heading arrow
    const hr = ((s.heading || 0) * Math.PI / 180);
    const alen = RR * 1.8;
    arrow(rx, ry, rx + alen * Math.cos(hr), ry + alen * Math.sin(hr), OWN_EDGE, 2);
}

// ── Ball ──────────────────────────────────────────────────────────────────────
function drawBall(s) {
    // Hidden / extrapolated
    if (!s.ball_pos && s.ball_hidden_pos) {
        const {x, y} = s.ball_hidden_pos;
        const [pcx, pcy] = [cx(x), cy(y)];
        const r = sc(BR);
        ctx.beginPath(); ctx.arc(pcx, pcy, r, 0, Math.PI * 2);
        ctx.setLineDash([sc(0.015), sc(0.01)]);
        ctx.fillStyle  = s.ball_lost ? 'rgba(255,0,0,0.25)' : 'rgba(255,140,0,0.25)';
        ctx.strokeStyle = s.ball_lost ? 'red' : 'darkorange'; ctx.lineWidth = 1.5;
        ctx.fill(); ctx.stroke(); ctx.setLineDash([]);
    }

    // Detected position
    if (s.ball_pos) {
        const {x, y} = s.ball_pos;
        const [pcx, pcy] = [cx(x), cy(y)];
        ctx.beginPath(); ctx.arc(pcx, pcy, sc(BR), 0, Math.PI * 2);
        ctx.fillStyle = 'orange'; ctx.fill();
        ctx.strokeStyle = 'darkorange'; ctx.lineWidth = 1.5; ctx.stroke();
    }

    // Velocity arrow
    const orig = s.ball_pos || s.ball_hidden_pos;
    if (orig && s.ball_vx != null && s.ball_vy != null
            && Math.hypot(s.ball_vx, s.ball_vy) > 0.1) {
        arrow(orig.x, orig.y,
              orig.x + s.ball_vx * 0.5, orig.y + s.ball_vy * 0.5,
              'darkorange', 1.5);
    }
}

// ── Raw detections ────────────────────────────────────────────────────────────
function drawRawDetections(s) {
    if (s.ball_raw && s.ball_raw.global_pos) {
        cross(s.ball_raw.global_pos.x, s.ball_raw.global_pos.y, 0.03, 'darkorange', 2);
    }
    if (s.raw_robots) {
        for (const r of s.raw_robots) cross(r.x, r.y, 0.03, 'red', 2);
    }
}

// ── Ally markers ──────────────────────────────────────────────────────────────
function drawAllyMarkers(s) {
    if (!s.ally_pos_raw) return;
    const c = `rgb(${ALLY_RGB})`;
    const mp = s.ally_pos_raw['ally_main_robot_pos'];
    if (mp) diamond(mp.x, mp.y, 0.04, c, 2);
    for (let i = 1; i <= 3; i++) {
        const p = s.ally_pos_raw[`ally_other_pos_${i}`];
        if (p) xcross(p.x, p.y, 0.03, c, 2);
    }
    const bp = s.ally_pos_raw['ally_ball_pos'];
    if (bp) star(bp.x, bp.y, 0.035, c);
}

// ── Status text ───────────────────────────────────────────────────────────────
function drawStatus(s) {
    const pos = s.robot_pos
        ? `(${s.robot_pos[0].toFixed(2)}, ${s.robot_pos[1].toFixed(2)})`
        : 'unknown';
    const hdg = s.heading != null ? s.heading.toFixed(1) : '0.0';
    const walls = s.walls ? s.walls.length : 0;
    const bots  = s.other_robots ? s.other_robots.length : 0;
    const lines = [
        `pos=${pos}  heading=${hdg}°`,
        `walls=${walls}  bots=${bots}`
    ];
    const pad = 4, lh = 15;
    ctx.font = '12px monospace';
    const maxW = Math.max(...lines.map(l => ctx.measureText(l).width));
    ctx.globalAlpha = 0.75;
    ctx.fillStyle = 'white';
    ctx.fillRect(5, 5, maxW + pad * 2, lines.length * lh + pad * 2);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#333';
    ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    lines.forEach((l, i) => ctx.fillText(l, 5 + pad, 5 + pad + i * lh));
}

// ── Strategy points ───────────────────────────────────────────────────────────
function drawStrategyArrow(x1, y1, x2, y2, color) {
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.hypot(dx, dy);
    if (len < 1) return;
    const angle = Math.atan2(dy, dx);
    const head = Math.max(8, Math.min(14, len * 0.18));
    const spread = Math.PI / 6;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(angle - spread),
               y2 - head * Math.sin(angle - spread));
    ctx.lineTo(x2 - head * Math.cos(angle + spread),
               y2 - head * Math.sin(angle + spread));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
}

function drawStrategyPoints(s) {
    const pts = s.strategy_points;
    if (!pts || pts.length === 0) return;
    const EDGE = '#22aa22';
    const FACE = 'rgba(136,221,136,0.85)';
    const r = sc(0.025);
    // Connecting lines (robot → points → ...)
    if (pts.length >= 1) {
        ctx.beginPath();

        // Start at robot if available
        if (s.robot_pos) {
            ctx.moveTo(cx(s.robot_pos[0]), cy(s.robot_pos[1]));
        } else {
            ctx.moveTo(cx(pts[0].x), cy(pts[0].y));
        }

        // Then go through all strategy points
        for (let i = 0; i < pts.length; i++) {
            ctx.lineTo(cx(pts[i].x), cy(pts[i].y));
        }

        ctx.strokeStyle = EDGE;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.stroke();
    }
    // Dir arrows (dashed, from each point toward its target)
    for (let i = 0; i < pts.length; i++) {
        const pt = pts[i];
        if (pt.dir) {
            drawStrategyArrow(cx(pt.x), cy(pt.y), cx(pt.dir.x), cy(pt.dir.y), EDGE);
        }
    }
    // Circles + index labels (drawn on top of arrows)
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let i = 0; i < pts.length; i++) {
        const pcx = cx(pts[i].x), pcy = cy(pts[i].y);
        ctx.beginPath();
        ctx.arc(pcx, pcy, r, 0, 2 * Math.PI);
        ctx.fillStyle = FACE;
        ctx.fill();
        ctx.strokeStyle = EDGE;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.stroke();
        ctx.fillStyle = 'black';
        ctx.fillText(String(i), pcx, pcy);
    }
}

// ── Game state HTML panel ─────────────────────────────────────────────────────
const gsEl = document.getElementById('gs');
function updateGameState(s) {
    if (!s.game_state) { gsEl.textContent = 'ball: —'; return; }
    const ctrl     = s.game_state.ball_control;
    const ctrlTeam = s.game_state.controlling_team;
    const bp       = s.game_state.ball;

    const ballStr = bp ? `(${bp.x.toFixed(2)}, ${bp.y.toFixed(2)})` : '—';
    const teamStr = ctrlTeam != null ? `T${ctrlTeam}` : '—';

    let ctrlStr = 'none';
    if (ctrl) {
        const cid = ctrl.id, ct = ctrl.team;
        ctrlStr = cid == null ? `self (T${ct})` : `#${cid} (T${ct})`;
    }
    gsEl.textContent = `ball: ${ballStr}\nctrl: ${ctrlStr}  [${teamStr}]`;
}

// ── Main render ───────────────────────────────────────────────────────────────
function render(s) {
    ctx.clearRect(0, 0, W, H);
    drawField();
    drawWalls(s);
    drawLidar(s);

    // History trails
    drawTrail(s.position_history,     sc(0.018), '43,122,43',  0.7);
    drawRobotHistory(s);
    drawTrail(s.ball_history,         sc(0.014), '255,140,0',  0.7);

    drawRawDetections(s);
    drawAllyMarkers(s);
    drawOtherRobots(s);
    drawOwnRobot(s);
    drawStrategyPoints(s);
    drawBall(s);
    drawStatus(s);
    updateGameState(s);
}

// ── SSE connection ────────────────────────────────────────────────────────────
const fpsEl = document.getElementById('fps');
let frameCount = 0, lastFpsTime = performance.now();
let lastState  = null;

function connect() {
    const es = new EventSource('/events');
    es.onmessage = e => {
        try {
            lastState = JSON.parse(e.data);
        } catch (_) {}
    };
    es.onerror = () => {
        fpsEl.textContent = 'reconnecting…';
        es.close();
        setTimeout(connect, 1000);
    };
}
connect();

// Render loop driven by rAF — decouples rendering from SSE delivery rate
function loop() {
    if (lastState) {
        render(lastState);
        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
            fpsEl.textContent = `${frameCount} fps`;
            frameCount = 0;
            lastFpsTime = now;
        }
    }
    requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

})();
</script>
</body>
</html>"""

# ── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request access log

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            body = _HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/events":
            self.send_response(200)
            self.send_header("Content-Type",  "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection",    "keep-alive")
            self.end_headers()
            last_seq = -1
            while True:
                with _push_cond:
                    _push_cond.wait_for(lambda: _push_seq != last_seq, timeout=5.0)
                    cur_seq = _push_seq
                if cur_seq == last_seq:
                    try:
                        with _perf.measure("sse_send"):
                            self.wfile.write(b": ka\n\n")
                            self.wfile.flush()
                    except Exception:
                        break
                    continue
                last_seq = cur_seq
                with _state_lock:
                    data = _build_state()
                try:
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except Exception:
                    break

        else:
            self.send_response(404)
            self.end_headers()


# ── Broker callbacks (identical to node_dev_twin_vis) ─────────────────────────

def on_update(key, value):
    global _lidar, _detection_origin, _detection_heading, _imu_pitch
    global _robot_pos, _other_robots, _walls
    global _position_history, _other_robots_history
    global _ball_pos, _ball_hidden_pos, _ball_lost, _ball_vx, _ball_vy, _ball_history
    global _ally_id, _ally_pos_raw, _raw_robots, _ball_raw, _game_state, _strategy_points

    if value is None:
        return


    with _perf.measure("update"):
        try:
            with _state_lock:
                if key == "lidar":
                    raw = __import__('json').loads(value)
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
                elif key == "ball_history":
                    _ball_history = json.loads(value)
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
            print(f"[WEB-VIS] parse error on {key!r}: {e}")
            return

    _notify()


# ── Entry point ───────────────────────────────────────────────────────────────

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
        "other_robots":         lambda v: [[float(r["x"]), float(r["y"]), r.get("method", ""),
                                            int(r.get("id", 0)), bool(r.get("ally", False))]
                                           for r in (lambda p: p.get("robots", p)
                                                     if isinstance(p, dict) else p)(json.loads(v))],
        "lidar_walls":          lambda v: json.loads(v),
        "position_history":     lambda v: json.loads(v),
        "other_robots_history": lambda v: json.loads(v),
        "ball":                 lambda v: json.loads(v),
        "ball_history":         lambda v: json.loads(v),
        "ally_id":              lambda v: int(v) if v and v.strip() else None,
        "raw_robots":           lambda v: json.loads(v),
        "ball_raw":             lambda v: json.loads(v),
        "game_state":           lambda v: json.loads(v),
        "robot_strategy_points":   lambda v: json.loads(v),
    }
    _TARGETS = {
        "imu_pitch":               "_imu_pitch",
        "lidar":                   "_lidar",
        "robot_position":          "_robot_pos",
        "other_robots":            "_other_robots",
        "lidar_walls":             "_walls",
        "position_history":        "_position_history",
        "other_robots_history":    "_other_robots_history",
        "ball":                    "_ball_pos",
        "ball_history":            "_ball_history",
        "ally_id":                 "_ally_id",
        "raw_robots":              "_raw_robots",
        "ball_raw":                "_ball_raw",
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

    _ally_keys = ["ally_main_robot_pos", "ally_other_pos_1",
                  "ally_other_pos_2", "ally_other_pos_3", "ally_ball_pos"]
    mb.setcallback(list(_SEEDS.keys()) + _ally_keys + ["robot_strategy_points"], on_update)

    threading.Thread(target=mb.receiver_loop, daemon=True, name="broker").start()

    import socket
    print(f"[WEB-VIS] Serving on port {PORT} — available at:")
    try:
        addrs = set()
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addr = info[4][0]
            if not addr.startswith("127.") and not addr.startswith("::"):
                addrs.add(addr)
        addrs.add("127.0.0.1")
        for addr in sorted(addrs):
            print(f"  http://{addr}:{PORT}/")
    except Exception:
        print(f"  http://0.0.0.0:{PORT}/")
    class _Server(HTTPServer):
        def handle_error(self, request, client_address):
            import sys
            if isinstance(sys.exc_info()[1],
                          (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
                return  # normal client disconnect — not an error
            super().handle_error(request, client_address)

    server = _Server((HOST, PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[WEB-VIS] Stopped.")
        mb.close()

        print("\n[WEB-VIS] Stopped.")
        mb.close()
