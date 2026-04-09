from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import numpy as np
import threading
import time

# ── Camera specifications (Raspberry Pi Cam V2) ────────────────────────────────
FOV_DEG    = 62.2          # Horizontal field of view in degrees
RES_WIDTH  = 640           # Frame width in pixels
RES_HEIGHT = 480           # Frame height in pixels
CENTER_X   = RES_WIDTH / 2.0

# ── Colour filter (HSV) ────────────────────────────────────────────────────────
# Tune these values with a colour-picker / slider script for your lighting.
LOWER_ORANGE = (5,  120, 100)
UPPER_ORANGE = (25, 255, 255)

# ── Detection thresholds ───────────────────────────────────────────────────────
DEADZONE_PIXELS = 40   # Centre dead-zone width (pixels) before steering
MIN_RADIUS      = 5    # Minimum ball radius in pixels (filters noise)

# ── Ball physical size ─────────────────────────────────────────────────────────
BALL_RADIUS_MM = 21.0  # Real-world ball radius in mm (used for distance calc)

# ── Robot geometry ────────────────────────────────────────────────────────────
ROBOT_RADIUS = 0.09   # metres — camera sits one radius ahead of the robot centre

# ── Field dimensions (used by SimBall physics) ────────────────────────────────
FIELD_WIDTH  = 1.58   # metres — playing field only
FIELD_HEIGHT = 2.19

# ── Broker key ────────────────────────────────────────────────────────────────
BROKER_KEY = "ball_raw"

SIM_REPLACE = True  # Set True to force simulated ball even if a camera is found

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_dev_vision", broker=mb, print_every=100)

_robot_pos = None   # (x, y) metres — from robot_position broker key
_imu_pitch = None   # degrees       — from imu_pitch broker key
_sim_state = None   # {"robot": [x,y], "obstacles": [[x,y],...]} — from sim_state broker key


_hw_available = False
try:
    import cv2
    _hw_available = True
except ImportError as e:
    print(f"[VISION] OpenCV not available ({e}) — node will not run.")


def _process_frame(frame):
    """
    Detect the orange ball in a BGR frame.

    Returns a dict:
        command      – "FORWARD" | "LEFT" | "RIGHT" | "NO_BALL"
        distance_cm  – estimated distance in centimetres (0.0 if no ball)
        x_center     – detected pixel x-centre (-1 if no ball)
        radius       – detected pixel radius (-1 if no ball)
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array(LOWER_ORANGE),
                       np.array(UPPER_ORANGE))

    # Noise reduction: erode then dilate
    mask = cv2.erode(mask,  None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    largest = max(contours, key=cv2.contourArea)
    (x_center, _), radius = cv2.minEnclosingCircle(largest)

    if radius <= MIN_RADIUS:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    # ── Distance estimate via angular diameter ─────────────────────────────────
    # The ball subtends an angle of Ow_deg degrees in the image.
    # Using: distance = ball_radius / tan(half_angle)
    diameter_px = radius * 2.0
    Ow_deg = (FOV_DEG / RES_WIDTH) * diameter_px
    Ow_rad = math.radians(Ow_deg)
    distance_cm = (BALL_RADIUS_MM / math.tan(Ow_rad / 2.0)) / 10.0 if Ow_rad > 0 else 0.0

    # ── Angle to ball (horizontal) ─────────────────────────────────────────────
    # Map pixel offset from centre to degrees using the horizontal FOV.
    # Positive = ball is to the right, negative = ball is to the left.
    error_x   = x_center - CENTER_X
    angle_deg = (error_x / RES_WIDTH) * FOV_DEG

    # ── Steering command ───────────────────────────────────────────────────────
    if error_x < -DEADZONE_PIXELS:
        command = "LEFT"
    elif error_x > DEADZONE_PIXELS:
        command = "RIGHT"
    else:
        command = "FORWARD"

    return {
        "command":     command,
        "distance_cm": round(distance_cm, 1),
        "angle_deg":   round(angle_deg, 1),
        "x_center":    round(float(x_center), 1),
        "radius":      round(float(radius), 1),
    }


class _SimBall:
    """
    Physics-based ball simulation that renders synthetic BGR camera frames.

    The ball lives in global field coordinates (metres) and bounces off the
    field walls and all simulated robots.  Each call to render() advances the
    physics, checks whether any obstacle robot occludes the camera→ball line,
    then projects the ball onto the virtual sensor and returns a BGR frame
    that _process_frame() can analyse exactly like a real one.

    Robot positions are read from the module-level _sim_state dict (published
    by node_lidar when running in simulation mode).
    """

    FIELD_W  = 1.58
    FIELD_H  = 2.19
    BALL_R   = BALL_RADIUS_MM / 1000.0   # 0.021 m
    MARGIN   = BALL_R + 0.02             # minimum distance from walls
    SPEED    = 0.6                        # m/s — initial ball speed

    def __init__(self):
        import random

        # Focal length in pixels — derived from horizontal FOV
        self._focal_px = (RES_WIDTH / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))

        # Initial ball position: random, not overlapping any robot
        self._x, self._y = self._random_position()
        angle = random.uniform(0, 2 * math.pi)
        self._vx = math.cos(angle) * self.SPEED
        self._vy = math.sin(angle) * self.SPEED

        self._last_t = time.monotonic()

        # Pre-compute a solid orange BGR colour within LOWER/UPPER_ORANGE.
        # HSV(15, 200, 220) is safely mid-range for both hue and saturation.
        _hsv = np.array([[[15, 200, 220]]], dtype=np.uint8)
        bgr  = cv2.cvtColor(_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        self._orange_bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _all_robots(self):
        """Return [(x, y)] for every simulated robot (own + obstacles)."""
        state = _sim_state
        if state is None:
            return []
        robots = []
        r = state.get("robot")
        if r:
            robots.append((float(r[0]), float(r[1])))
        robots += [(float(p[0]), float(p[1])) for p in state.get("obstacles", [])]
        return robots

    def _obstacle_robots(self):
        """Return [(x, y)] for obstacle robots only (potential occluders)."""
        state = _sim_state
        if state is None:
            return []
        return [(float(p[0]), float(p[1])) for p in state.get("obstacles", [])]

    def _random_position(self):
        """Find a random field position not overlapping any robot."""
        import random
        robots  = self._all_robots()
        min_sep = ROBOT_RADIUS + self.BALL_R + 0.05
        for _ in range(500):
            x = random.uniform(self.MARGIN, self.FIELD_W - self.MARGIN)
            y = random.uniform(self.MARGIN, self.FIELD_H - self.MARGIN)
            if all(math.hypot(x - rx, y - ry) >= min_sep for rx, ry in robots):
                return x, y
        return self.FIELD_W / 2.0, self.FIELD_H / 2.0   # fallback: field centre

    def _is_occluded(self, cx, cy, bx, by):
        """
        Return True if any obstacle robot's body crosses the camera→ball segment.
        Uses a 2-D ray–circle intersection against each obstacle.
        """
        dx, dy     = bx - cx, by - cy
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return False
        for rx, ry in self._obstacle_robots():
            # Scalar projection of occluder centre onto the segment, clamped to [0,1]
            t = max(0.0, min(1.0,
                ((rx - cx) * dx + (ry - cy) * dy) / seg_len_sq))
            closest_dist = math.hypot(cx + t * dx - rx, cy + t * dy - ry)
            if closest_dist < ROBOT_RADIUS:
                return True
        return False

    # ── Physics + render ──────────────────────────────────────────────────────

    def render(self):
        """Advance physics and return a synthetic BGR frame."""
        now = time.monotonic()
        dt  = now - self._last_t
        self._last_t = now

        # ── Advance ball position ─────────────────────────────────────────────
        self._x += self._vx * dt
        self._y += self._vy * dt

        # ── Bounce off field walls ────────────────────────────────────────────
        if self._x < self.MARGIN:
            self._x  = self.MARGIN;              self._vx =  abs(self._vx)
        elif self._x > self.FIELD_W - self.MARGIN:
            self._x  = self.FIELD_W - self.MARGIN; self._vx = -abs(self._vx)
        if self._y < self.MARGIN:
            self._y  = self.MARGIN;              self._vy =  abs(self._vy)
        elif self._y > self.FIELD_H - self.MARGIN:
            self._y  = self.FIELD_H - self.MARGIN; self._vy = -abs(self._vy)

        # ── Elastic bounce off every simulated robot ──────────────────────────
        sep = ROBOT_RADIUS + self.BALL_R
        for rx, ry in self._all_robots():
            dx, dy = self._x - rx, self._y - ry
            dist   = math.hypot(dx, dy)
            if 0 < dist < sep:
                nx, ny    = dx / dist, dy / dist
                self._x   = rx + nx * sep           # push out of overlap
                self._y   = ry + ny * sep
                dot        = self._vx * nx + self._vy * ny
                if dot < 0:                         # only reflect if approaching
                    self._vx -= 2 * dot * nx
                    self._vy -= 2 * dot * ny

        # ── Camera pose in global field frame ─────────────────────────────────
        if _robot_pos is not None and _imu_pitch is not None:
            heading_rad = math.radians(_imu_pitch)
            cam_x = _robot_pos[0] + ROBOT_RADIUS * math.cos(heading_rad)
            cam_y = _robot_pos[1] + ROBOT_RADIUS * math.sin(heading_rad)
        else:
            # Robot position not yet known — use field centre facing right
            heading_rad = 0.0
            cam_x, cam_y = self.FIELD_W / 2.0, self.FIELD_H / 2.0

        # ── Occlusion check ───────────────────────────────────────────────────
        if self._is_occluded(cam_x, cam_y, self._x, self._y):
            return np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype=np.uint8)

        # ── Project ball into camera-local frame ──────────────────────────────
        # Rotate global offset by -heading to get (right, forward) camera axes
        dx, dy  = self._x - cam_x, self._y - cam_y
        cos_h   = math.cos(heading_rad)
        sin_h   = math.sin(heading_rad)
        local_z =  dx * cos_h + dy * sin_h   # depth  (forward)
        local_x = -dx * sin_h + dy * cos_h   # lateral (right = positive)

        if local_z < 0.05:   # ball is behind or too close to the camera
            return np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype=np.uint8)

        local_z_mm = local_z * 1000.0
        local_x_mm = local_x * 1000.0

        px        = int(CENTER_X + (local_x_mm / local_z_mm) * self._focal_px)
        radius_px = max(1, int((BALL_RADIUS_MM / local_z_mm) * self._focal_px))

        # ── Render ────────────────────────────────────────────────────────────
        frame = np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype=np.uint8)
        cv2.circle(frame, (px, RES_HEIGHT // 2), radius_px, self._orange_bgr, -1)
        return frame

    @property
    def pos(self):
        """True ball position in global field coordinates (metres)."""
        return {"x": round(self._x, 3), "y": round(self._y, 3)}



def _on_broker_update(key, value):
    global _robot_pos, _imu_pitch, _sim_state
    if value is None:
        return
    if key == "robot_position":
        try:
            p = json.loads(value)
            _robot_pos = (float(p["x"]), float(p["y"]))
        except Exception:
            pass
    elif key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except Exception:
            pass
    elif key == "sim_state":
        try:
            _sim_state = json.loads(value)
        except Exception:
            pass


def _compute_global_pos(distance_cm, angle_deg):
    """
    Project the detected ball into the global field frame.

    The camera is mounted one robot radius ahead of the robot centre,
    facing in the heading direction.  The ball bearing combines the
    heading with the camera's measured horizontal angle to the ball.
    Returns {"x": float, "y": float} or None if position/heading unknown.
    """
    if _robot_pos is None or _imu_pitch is None:
        return None
    rx, ry      = _robot_pos
    heading_rad = math.radians(_imu_pitch)
    cam_x = rx + ROBOT_RADIUS * math.cos(heading_rad)
    cam_y = ry + ROBOT_RADIUS * math.sin(heading_rad)
    bearing = heading_rad + math.radians(angle_deg)
    dist_m  = distance_cm / 100.0
    return {
        "x": round(cam_x + dist_m * math.cos(bearing), 3),
        "y": round(cam_y + dist_m * math.sin(bearing), 3),
    }


if __name__ == "__main__":
    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    if not _hw_available:
        raise SystemExit("[VISION] Cannot start: OpenCV is not installed.")

    # Seed broker values and start receiver thread
    try:
        val = mb.get("robot_position")
        if val is not None:
            p = json.loads(val)
            _robot_pos = (float(p["x"]), float(p["y"]))
    except Exception:
        pass
    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass
    try:
        val = mb.get("sim_state")
        if val is not None:
            _sim_state = json.loads(val)
    except Exception:
        pass
    mb.setcallback(["robot_position", "imu_pitch", "sim_state"], _on_broker_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    print("[VISION] Starting headless vision system...")

    if SIM_REPLACE:
        cap = None
        sim = _SimBall()
        print("[VISION] SIM_REPLACE=True — using simulated ball.")
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RES_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)

        if cap.isOpened():
            print(f"[VISION] Camera opened ({RES_WIDTH}x{RES_HEIGHT}). "
                  "Running detection loop (Ctrl+C to stop)...")
            sim = None
        else:
            cap.release()
            cap = None
            sim = _SimBall()
            print("[VISION] Camera not available — falling back to simulated ball.")

    last_log_time = time.time()

    try:
        while True:
            if sim is not None:
                frame = sim.render()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[VISION] ERROR: Camera connection lost!")
                    break

            with _perf.measure("frame"):
                result = _process_frame(frame)

                gpos = None
                if result["command"] != "NO_BALL":
                    gpos = _compute_global_pos(result["distance_cm"], result["angle_deg"])

                result["global_pos"] = gpos

                if sim is not None:
                    result["sim_pos"] = sim.pos
                mb.set(BROKER_KEY, json.dumps(result))

            # Log to console at most once per second
            now = time.time()
            if now - last_log_time >= 1.0:
                if result["command"] == "NO_BALL":
                    print("[VISION] Status: NO BALL")
                else:
                    print(f"[VISION] Status: {result['command']} | "
                          f"Distance: {result['distance_cm']:.1f} cm | "
                          f"Angle: {result['angle_deg']:.1f} deg")
                last_log_time = now

    except KeyboardInterrupt:
        print("\n[VISION] Stopped by user.")
    finally:
        if cap is not None:
            cap.release()
        mb.close()
        print("[VISION] Camera closed. System stopped.")
