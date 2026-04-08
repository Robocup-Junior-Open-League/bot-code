from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import numpy as np
import threading
import time
import sys
import cv2

# ── Camera specifications (Raspberry Pi Camera V2 / Webcam) ──────────────────
FOV_DEG    = 62.2   # Horizontaler Blickwinkel der Pi Camera V2 (für PC-Webcam oft ~52.0°)
RES_WIDTH  = 160    # Native 4:3 Format für maximale CPU-Schonung auf RPi Zero
RES_HEIGHT = 120    
CENTER_X   = RES_WIDTH / 2.0

# ── Image Masking (Zuschauer-Zensur) ───────────────────────────────────────────
CENSOR_TOP_HEIGHT_PX = 30  # Oberste 30 Pixel schwärzen (Bunte T-Shirts ausblenden)

# ── Colour filter (HSV) ────────────────────────────────────────────────────────
# Trage hier eure perfekten Werte ein, die ihr mit dem Kalibrierungs-Skript gefunden habt
LOWER_ORANGE = (5, 31, 166)
UPPER_ORANGE = (19, 151, 255)

# ── Detection thresholds ───────────────────────────────────────────────────────
DEADZONE_PIXELS = 10  # Toleranz-Bereich für "FORWARD"
MIN_RADIUS      = 1   # Mindestradius in Pixel

# ── Ball physical size & Robot geometry ────────────────────────────────────────
BALL_RADIUS_MM = 21.0
ROBOT_RADIUS = 0.09
FIELD_WIDTH  = 1.82
FIELD_HEIGHT = 2.43
BROKER_KEY = "ball_raw"

SIM_REPLACE = True  # Setze dies auf True, um ABSICHTLICH den Digital Twin zu nutzen

# ── ADAPTIVE EXPONENTIAL MOVING AVERAGE (AEMA) SETUP ──────────────────────────
AEMA_ALPHA_MIN = 0.08  # Starke Glättung bei Rauschen
AEMA_ALPHA_MAX = 0.5   # Schnelle Reaktion bei echter Bewegung
AEMA_THRESHOLD = 0.15  

class AdaptiveEMA:
    """Intelligenter Filter, der Wackeln ignoriert, aber bei Bewegung sofort reagiert."""
    def __init__(self, alpha_min=AEMA_ALPHA_MIN, alpha_max=AEMA_ALPHA_MAX, threshold=AEMA_THRESHOLD):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.threshold = threshold
        self.estimate = None
    
    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return measurement
        
        if abs(self.estimate) > 1e-6:
            relative_change = abs(measurement - self.estimate) / abs(self.estimate)
        else:
            relative_change = abs(measurement - self.estimate)
        
        alpha = self.alpha_max if relative_change > self.threshold else self.alpha_min  
        self.estimate = alpha * measurement + (1.0 - alpha) * self.estimate
        return self.estimate
    
    def reset(self):
        self.estimate = None

_aema_dist = AdaptiveEMA()
_aema_angle = AdaptiveEMA()
_aema_x = AdaptiveEMA()
_aema_radius = AdaptiveEMA()

def _reset_filters():
    global _aema_dist, _aema_angle, _aema_x, _aema_radius
    _aema_dist.reset()
    _aema_angle.reset()
    _aema_x.reset()
    _aema_radius.reset()
# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_vision", broker=mb, print_every=100)

_robot_pos = None
_imu_pitch = None
_sim_state = None

def _process_frame(frame):
    """Das Gehirn der Bildverarbeitung (aus dem Kalibrierungsskript übernommen)."""
    # 🚫 ZUSCHAUER-ZENSUR
    if CENSOR_TOP_HEIGHT_PX > 0:
        frame[0:CENSOR_TOP_HEIGHT_PX, :] = (0, 0, 0)

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(LOWER_ORANGE), np.array(UPPER_ORANGE))

    mask = cv2.erode(mask,  None, iterations=1) 
    mask = cv2.dilate(mask, None, iterations=1)

    # 📡 RADAR-PIXEL FÜR DEN PC (PIXEL TRANSLATOR)
    small_mask = cv2.resize(mask, (30, 30), interpolation=cv2.INTER_NEAREST)
    y_coords, x_coords = np.where(small_mask > 0)
    
    radar_payload = {}
    for i in range(min(len(x_coords), 50)):
        radar_payload[f"p{i+1}"] = [int(x_coords[i]), int(y_coords[i])]
    mb.set("radar_pixels", json.dumps(radar_payload))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    largest = max(contours, key=cv2.contourArea)
    (x_center, _), radius = cv2.minEnclosingCircle(largest)

    if radius <= MIN_RADIUS:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    # Trigonometrische Mathematik für den Pi Zero
    diameter_px = radius * 2.0
    Ow_deg = (FOV_DEG / RES_WIDTH) * diameter_px
    Ow_rad = math.radians(Ow_deg)
    distance_cm = (BALL_RADIUS_MM / math.tan(Ow_rad / 2.0)) / 10.0 if Ow_rad > 0 else 0.0

    error_x   = x_center - CENTER_X
    angle_deg = (error_x / RES_WIDTH) * FOV_DEG

    return {
        "command":     "FOUND", 
        "distance_cm": distance_cm,
        "angle_deg":   angle_deg,
        "x_center":    float(x_center),
        "radius":      float(radius),
    }


class _SimBall:
    """Der geniale Digital Twin von eurem Projekt."""
    FIELD_W  = 1.82
    FIELD_H  = 2.43
    BALL_R   = BALL_RADIUS_MM / 1000.0
    MARGIN   = BALL_R + 0.02
    SPEED    = 0.6

    CAPTURE_DURATION = 10.0   # seconds the ball stays locked to the robot
    CAPTURE_CHANCE   = 0.5    # probability of capture on collision

    def __init__(self):
        import random
        self._focal_px = (RES_WIDTH / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))
        self._x, self._y = self._random_position()
        angle = random.uniform(0, 2 * math.pi)
        self._vx = math.cos(angle) * self.SPEED
        self._vy = math.sin(angle) * self.SPEED
        self._last_t = time.monotonic()
        _hsv = np.array([[[15, 200, 220]]], dtype=np.uint8)
        bgr  = cv2.cvtColor(_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        self._orange_bgr = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
        self._captured_robot_idx = None   # index into _all_robots()
        self._captured_offset    = (0.0, 0.0)
        self._capture_end_t      = 0.0

    def _all_robots(self):
        state = _sim_state
        if state is None: return []
        robots = []
        r = state.get("robot")
        if r: robots.append((float(r[0]), float(r[1])))
        robots += [(float(p[0]), float(p[1])) for p in state.get("obstacles", [])]
        return robots

    def _obstacle_robots(self):
        state = _sim_state
        if state is None: return []
        return [(float(p[0]), float(p[1])) for p in state.get("obstacles", [])]

    def _random_position(self):
        import random
        robots  = self._all_robots()
        min_sep = ROBOT_RADIUS + self.BALL_R + 0.05
        for _ in range(500):
            x = random.uniform(self.MARGIN, self.FIELD_W - self.MARGIN)
            y = random.uniform(self.MARGIN, self.FIELD_H - self.MARGIN)
            if all(math.hypot(x - rx, y - ry) >= min_sep for rx, ry in robots):
                return x, y
        return self.FIELD_W / 2.0, self.FIELD_H / 2.0

    def _is_occluded(self, cx, cy, bx, by):
        dx, dy     = bx - cx, by - cy
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12: return False
        for rx, ry in self._obstacle_robots():
            t = max(0.0, min(1.0, ((rx - cx) * dx + (ry - cy) * dy) / seg_len_sq))
            closest_dist = math.hypot(cx + t * dx - rx, cy + t * dy - ry)
            if closest_dist < ROBOT_RADIUS: return True
        return False

    def render(self):
        import random
        now = time.monotonic()
        dt  = now - self._last_t
        self._last_t = now

        self._x += self._vx * dt
        self._y += self._vy * dt

        if self._x < self.MARGIN: self._x = self.MARGIN; self._vx = abs(self._vx)
        elif self._x > self.FIELD_W - self.MARGIN: self._x = self.FIELD_W - self.MARGIN; self._vx = -abs(self._vx)
        if self._y < self.MARGIN: self._y = self.MARGIN; self._vy = abs(self._vy)
        elif self._y > self.FIELD_H - self.MARGIN: self._y = self.FIELD_H - self.MARGIN; self._vy = -abs(self._vy)

        sep = ROBOT_RADIUS + self.BALL_R
        for rx, ry in self._all_robots():
            dx, dy = self._x - rx, self._y - ry
            dist   = math.hypot(dx, dy)
            if 0 < dist < sep:
                nx, ny  = dx / dist, dy / dist
                self._x = rx + nx * sep
                self._y = ry + ny * sep
                dot     = self._vx * nx + self._vy * ny
                if dot < 0:
                    self._vx -= 2 * dot * nx
                    self._vy -= 2 * dot * ny

        if _robot_pos is not None and _imu_pitch is not None:
            heading_rad = math.radians(_imu_pitch)
            cam_x = _robot_pos[0] + ROBOT_RADIUS * math.cos(heading_rad)
            cam_y = _robot_pos[1] + ROBOT_RADIUS * math.sin(heading_rad)
        else:
            heading_rad = 0.0
            cam_x, cam_y = self.FIELD_W / 2.0, self.FIELD_H / 2.0

        if self._is_occluded(cam_x, cam_y, self._x, self._y):
            return np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype=np.uint8)

        dx, dy  = self._x - cam_x, self._y - cam_y
        cos_h   = math.cos(heading_rad)
        sin_h   = math.sin(heading_rad)
        local_z =  dx * cos_h + dy * sin_h
        local_x = -dx * sin_h + dy * cos_h

        if local_z < 0.05:
            return np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype=np.uint8)

        local_z_mm = local_z * 1000.0
        local_x_mm = local_x * 1000.0

        px        = int(CENTER_X + (local_x_mm / local_z_mm) * self._focal_px)
        radius_px = max(1, int((BALL_RADIUS_MM / local_z_mm) * self._focal_px))

        frame = np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype=np.uint8)
        cv2.circle(frame, (px, RES_HEIGHT // 2), radius_px, self._orange_bgr, -1)
        return frame

    @property
    def pos(self):
        return {"x": round(self._x, 3), "y": round(self._y, 3)}


def _on_broker_update(key, value):
    global _robot_pos, _imu_pitch, _sim_state
    if value is None: return
    if key == "robot_position":
        try:
            p = json.loads(value)
            _robot_pos = (float(p["x"]), float(p["y"]))
        except: pass
    elif key == "imu_pitch":
        try: _imu_pitch = float(value)
        except: pass
    elif key == "sim_state":
        try: _sim_state = json.loads(value)
        except: pass


def _compute_global_pos(distance_cm, angle_deg):
    if _robot_pos is None or _imu_pitch is None: return None
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

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":

    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    if not _hw_available:
        raise SystemExit("[VISION] Cannot start: OpenCV is not installed.")

    try:
        val = mb.get("robot_position")
        if val is not None:
            p = json.loads(val)
            _robot_pos = (float(p["x"]), float(p["y"]))
    except: pass
    try:
        val = mb.get("imu_pitch")
        if val is not None: _imu_pitch = float(val)
    except: pass
    try:
        val = mb.get("sim_state")
        if val is not None: _sim_state = json.loads(val)
    except: pass
    
    mb.setcallback(["robot_position", "imu_pitch", "sim_state"], _on_broker_update)
    threading.Thread(target=mb.receiver_loop, daemon=True, name="broker-receiver").start()

    print("[VISION] Starting headless vision system...")

    # 🚀 ROBUSTES KAMERA/SIMULATION SETUP
    picam2 = None
    cap = None
    sim = None

    if SIM_REPLACE:
        sim = _SimBall()
        print("[VISION] SIM_REPLACE=True — using simulated ball (Digital Twin).")
    else:
        try:
            # 1. VERSUCH: Raspberry Pi Picamera2
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_video_configuration(main={"size": (RES_WIDTH, RES_HEIGHT), "format": "BGR888"})
            picam2.configure(config)
            picam2.start()
            print(f"[VISION] 🚀 Hardware Picamera2 gestartet ({RES_WIDTH}x{RES_HEIGHT}).")
        except Exception as e:
            # 2. VERSUCH: USB-Webcam (Für den PC)
            print(f"[VISION] Picamera2 nicht gefunden ({e}). Teste USB-Webcam...")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
            
            if cap.isOpened():
                print(f"[VISION] USB Webcam geoeffnet. (Achtung: FOV muss evtl. angepasst werden)")
            else:
                # 3. VERSUCH: Fallback auf Simulation (Digital Twin)
                cap.release()
                cap = None
                sim = _SimBall()
                print("[VISION] KEINE KAMERA GEFUNDEN! Wechsle sicherheitshalber in den Digital Twin-Modus.")

    print(f"[VISION] Censor Box Active: Top {CENSOR_TOP_HEIGHT_PX} pixels will be ignored.")
    last_log_time = time.time()

    try:
        while True:
            # 📸 ROBUSTES EINLESEN (Egal welche Kamera / Simulation aktiv ist)
            if sim is not None:
                frame = sim.render()
            elif picam2 is not None:
                frame = picam2.capture_array()
            elif cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("[VISION] ERROR: Camera connection lost!")
                    break
                # Wenn es die Webcam ist, erzwinge die richtige Groesse!
                frame = cv2.resize(frame, (RES_WIDTH, RES_HEIGHT), interpolation=cv2.INTER_NEAREST)

            with _perf.measure("frame"):
                result = _process_frame(frame)

                if result["command"] != "NO_BALL":
                    raw_dist = result["distance_cm"]
                    raw_angle = result["angle_deg"]
                    raw_x = result["x_center"]
                    raw_rad = result["radius"]

                    # AEMA Glättung anwenden
                    result["distance_cm"] = round(_aema_dist.update(raw_dist), 1)
                    result["angle_deg"]   = round(_aema_angle.update(raw_angle), 1)
                    result["x_center"]    = round(_aema_x.update(raw_x), 1)
                    result["radius"]      = round(_aema_radius.update(raw_rad), 1)

                    error_x = result["x_center"] - CENTER_X
                    if error_x < -DEADZONE_PIXELS:
                        result["command"] = "LEFT"
                    elif error_x > DEADZONE_PIXELS:
                        result["command"] = "RIGHT"
                    else:
                        result["command"] = "FORWARD"
                else:
                    _reset_filters()

                # --- GLOBAL POSITION & BROKER ---
                gpos = None
                if result["command"] != "NO_BALL":
                    gpos = _compute_global_pos(result["distance_cm"], result["angle_deg"])

                result["global_pos"] = gpos

                if sim is not None:
                    result["sim_pos"] = sim.pos
                mb.set(BROKER_KEY, json.dumps(result))

            # Logging
            now = time.time()
            if now - last_log_time >= 1.0:
                if result["command"] == "NO_BALL":
                    print("[VISION] Status: NO_BALL")
                else:
                    print(f"[VISION] Status: {result['command']} | "
                          f"Distance: {result['distance_cm']:.1f} cm | "
                          f"Angle: {result['angle_deg']:.1f} deg")
                last_log_time = now

    except KeyboardInterrupt:
        print("\n[VISION] Stopped by user.")
    finally:
        # Sauberes Beenden von jeglicher Hardware
        if picam2 is not None:
            picam2.stop()
        if cap is not None:
            cap.release()
        mb.close()
        print("[VISION] Camera closed. System stopped.")