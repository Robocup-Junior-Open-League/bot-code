import sys
import os
import json
import math
import threading
import time
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))
from robus_core.libs.lib_telemtrybroker import TelemetryBroker

# ==============================================================================
# KAMERA-KONFIGURATION (Raspberry Pi Camera V2 / USB-Webcam)
# ==============================================================================
FOV_DEG    = 62.2   # Horizontaler Blickwinkel Pi Cam V2 (Webcam oft ~52.0°)
RES_WIDTH  = 160
RES_HEIGHT = 120
CENTER_X   = RES_WIDTH / 2.0

# ==============================================================================
# BILD-MASKIERUNG (Zuschauer-Zensur)
# ==============================================================================
CENSOR_TOP_HEIGHT_PX = 30  # Oberste N Pixel schwärzen

# ==============================================================================
# FARB-FILTER (HSV) — aus sweep_calibration_no_gui.py übernehmen/anpassen
# ==============================================================================
LOWER_ORANGE = np.array([5,  31, 166])
UPPER_ORANGE = np.array([19, 151, 255])

# ==============================================================================
# ERKENNUNGS-SCHWELLWERTE
# ==============================================================================
DEADZONE_PIXELS = 10   # Toleranz-Bereich für "FORWARD"
MIN_RADIUS      = 1    # Mindestradius in Pixel

# ==============================================================================
# PHYSIK & GEOMETRIE
# ==============================================================================
BALL_RADIUS_MM = 21.0
ROBOT_RADIUS   = 0.09
FIELD_WIDTH    = 1.58
FIELD_HEIGHT   = 2.19
BROKER_KEY     = "ball_raw"

# ==============================================================================
# SIMULATION (Digital Twin)
# ==============================================================================
SIM_REPLACE = True   # True  → immer Simulation nutzen
                     # False → Hardware-Kamera, Simulation nur als Fallback

# ==============================================================================
# ADAPTIVE EXPONENTIAL MOVING AVERAGE (AEMA)
# ==============================================================================
AEMA_ALPHA_MIN = 0.08   # Starke Glättung bei Rauschen
AEMA_ALPHA_MAX = 0.5    # Schnelle Reaktion bei echter Bewegung
AEMA_THRESHOLD = 0.15   # Relativer Sprung ab dem alpha_max gilt


class AdaptiveEMA:
    """Ignoriert Wackeln, reagiert sofort auf echte Bewegung."""

    def __init__(self, alpha_min=AEMA_ALPHA_MIN, alpha_max=AEMA_ALPHA_MAX,
                 threshold=AEMA_THRESHOLD):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.threshold = threshold
        self.estimate  = None

    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return measurement
        base = abs(self.estimate)
        relative_change = (abs(measurement - self.estimate) / base
                           if base > 1e-6 else abs(measurement - self.estimate))
        alpha = self.alpha_max if relative_change > self.threshold else self.alpha_min
        self.estimate = alpha * measurement + (1.0 - alpha) * self.estimate
        return self.estimate

    def reset(self):
        self.estimate = None


_aema_dist   = AdaptiveEMA()
_aema_angle  = AdaptiveEMA()
_aema_x      = AdaptiveEMA()
_aema_radius = AdaptiveEMA()


def _reset_filters():
    _aema_dist.reset()
    _aema_angle.reset()
    _aema_x.reset()
    _aema_radius.reset()


# ==============================================================================
# GLOBALER ZUSTAND (wird per Broker-Callback aktuell gehalten)
# ==============================================================================
_robot_pos: tuple | None = None   # (x, y) in Meter
_imu_pitch: float | None = None   # Heading in Grad
_sim_state: dict  | None = None   # Simulationszustand vom Broker


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


# ==============================================================================
# BILDVERARBEITUNG
# ==============================================================================
def _process_frame(frame: np.ndarray) -> dict:
    """Führt HSV-Filterung und Kreis-Detektion auf einem BGR-Frame durch."""

    # Zuschauer-Zensur
    if CENSOR_TOP_HEIGHT_PX > 0:
        frame[:CENSOR_TOP_HEIGHT_PX, :] = 0

    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)

    # Morphologisches Glätten (Rauschen entfernen)
    mask = cv2.erode(mask,  None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    # Radar-Pixel für Debug/Visualisierung
    small_mask = cv2.resize(mask, (30, 30), interpolation=cv2.INTER_NEAREST)
    y_coords, x_coords = np.where(small_mask > 0)
    radar_payload = {
        f"p{i + 1}": [int(x_coords[i]), int(y_coords[i])]
        for i in range(min(len(x_coords), 50))
    }
    mb.set("radar_pixels", json.dumps(radar_payload))

    # Kontur-Suche
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    largest = max(contours, key=cv2.contourArea)
    (x_center, _), radius = cv2.minEnclosingCircle(largest)

    if radius <= MIN_RADIUS:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    # Distanz-Berechnung per Winkel-Trigonometrie
    Ow_rad = math.radians((FOV_DEG / RES_WIDTH) * (radius * 2.0))
    distance_cm = (
        (BALL_RADIUS_MM / math.tan(Ow_rad / 2.0)) / 10.0
        if Ow_rad > 0 else 0.0
    )
    angle_deg = ((x_center - CENTER_X) / RES_WIDTH) * FOV_DEG

    return {
        "command":     "FOUND",
        "distance_cm": distance_cm,
        "angle_deg":   angle_deg,
        "x_center":    float(x_center),
        "radius":      float(radius),
    }


# ==============================================================================
# GLOBALE POSITION BERECHNEN
# ==============================================================================
def _compute_global_pos(distance_cm: float, angle_deg: float) -> dict | None:
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


# ==============================================================================
# DIGITAL TWIN (SimBall)
# ==============================================================================
class _SimBall:
    """Simuliert einen beweglichen Ball auf dem Spielfeld."""

    MARGIN = BALL_RADIUS_MM / 1000.0 + 0.02
    SPEED  = 0.6

    def __init__(self):
        import random
        self._focal_px = (
            (RES_WIDTH / 2.0) / math.tan(math.radians(FOV_DEG / 2.0))
        )
        self._x, self._y = self._random_position()
        angle    = random.uniform(0, 2 * math.pi)
        self._vx = math.cos(angle) * self.SPEED
        self._vy = math.sin(angle) * self.SPEED
        self._last_t = time.monotonic()

        # Orange-Farbe für den simulierten Ball berechnen (HSV→BGR)
        hsv_px = np.array([[[15, 200, 220]]], dtype=np.uint8)
        bgr_px = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0]
        self._orange_bgr = (int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2]))

    # ── Hilfsfunktionen ───────────────────────────────────────────────────────

    def _all_robots(self) -> list[tuple[float, float]]:
        robots = []
        if _sim_state is None:
            return robots
        r = _sim_state.get("robot")
        if r:
            robots.append((float(r[0]), float(r[1])))
        robots += [(float(p[0]), float(p[1]))
                   for p in _sim_state.get("obstacles", [])]
        return robots

    def _obstacle_robots(self) -> list[tuple[float, float]]:
        if _sim_state is None:
            return []
        return [(float(p[0]), float(p[1]))
                for p in _sim_state.get("obstacles", [])]

    def _random_position(self) -> tuple[float, float]:
        import random
        min_sep = ROBOT_RADIUS + BALL_RADIUS_MM / 1000.0 + 0.05
        for _ in range(500):
            x = random.uniform(self.MARGIN, FIELD_WIDTH  - self.MARGIN)
            y = random.uniform(self.MARGIN, FIELD_HEIGHT - self.MARGIN)
            if all(math.hypot(x - rx, y - ry) >= min_sep
                   for rx, ry in self._all_robots()):
                return x, y
        return FIELD_WIDTH / 2.0, FIELD_HEIGHT / 2.0

    def _is_occluded(self, cx, cy, bx, by) -> bool:
        dx, dy     = bx - cx, by - cy
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return False
        for rx, ry in self._obstacle_robots():
            t = max(0.0, min(1.0,
                ((rx - cx) * dx + (ry - cy) * dy) / seg_len_sq))
            if math.hypot(cx + t * dx - rx, cy + t * dy - ry) < ROBOT_RADIUS:
                return True
        return False

    # ── Haupt-Render ─────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        now = time.monotonic()
        dt  = now - self._last_t
        self._last_t = now

        # Bewegung
        self._x += self._vx * dt
        self._y += self._vy * dt

        # Wand-Kollision
        if self._x < self.MARGIN:
            self._x  = self.MARGIN;              self._vx =  abs(self._vx)
        elif self._x > FIELD_WIDTH - self.MARGIN:
            self._x  = FIELD_WIDTH - self.MARGIN; self._vx = -abs(self._vx)
        if self._y < self.MARGIN:
            self._y  = self.MARGIN;               self._vy =  abs(self._vy)
        elif self._y > FIELD_HEIGHT - self.MARGIN:
            self._y  = FIELD_HEIGHT - self.MARGIN; self._vy = -abs(self._vy)

        # Roboter-Kollision
        sep = ROBOT_RADIUS + BALL_RADIUS_MM / 1000.0
        for rx, ry in self._all_robots():
            dx, dy = self._x - rx, self._y - ry
            dist   = math.hypot(dx, dy)
            if 0 < dist < sep:
                nx, ny   = dx / dist, dy / dist
                self._x  = rx + nx * sep
                self._y  = ry + ny * sep
                dot = self._vx * nx + self._vy * ny
                if dot < 0:
                    self._vx -= 2 * dot * nx
                    self._vy -= 2 * dot * ny

        # Kamera-Position
        if _robot_pos is not None and _imu_pitch is not None:
            heading_rad = math.radians(_imu_pitch)
            cam_x = _robot_pos[0] + ROBOT_RADIUS * math.cos(heading_rad)
            cam_y = _robot_pos[1] + ROBOT_RADIUS * math.sin(heading_rad)
        else:
            heading_rad = 0.0
            cam_x, cam_y = FIELD_WIDTH / 2.0, FIELD_HEIGHT / 2.0

        # Verdeckungs-Check
        if self._is_occluded(cam_x, cam_y, self._x, self._y):
            return np.zeros((RES_HEIGHT, RES_WIDTH, 3), dtype=np.uint8)

        # Projektion in Kamera-Koordinaten
        dx, dy  = self._x - cam_x, self._y - cam_y
        cos_h, sin_h = math.cos(heading_rad), math.sin(heading_rad)
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
    def pos(self) -> dict:
        return {"x": round(self._x, 3), "y": round(self._y, 3)}


# ==============================================================================
# KAMERA-INITIALISIERUNG
#   Gibt zurück: ("picam2", picam2_obj) | ("webcam", cap_obj) | ("sim", sim_obj)
# ==============================================================================
def _init_camera():
    if SIM_REPLACE:
        print("[VISION] SIM_REPLACE=True — Digital Twin aktiv.")
        return "sim", _SimBall()

    # 1. Versuch: Picamera2 (Raspberry Pi)
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        cfg = picam2.create_video_configuration(
            main={"size": (RES_WIDTH, RES_HEIGHT), "format": "BGR888"}
        )
        picam2.configure(cfg)
        picam2.start()
        time.sleep(0.5)  # Sensor-Aufwärmzeit
        print(f"[VISION] Picamera2 gestartet ({RES_WIDTH}x{RES_HEIGHT}).")
        return "picam2", picam2
    except Exception as e:
        print(f"[VISION] Picamera2 nicht verfügbar ({e}).")

    # 2. Versuch: USB-Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RES_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("[VISION] USB-Webcam geöffnet. (FOV ggf. anpassen!)")
            return "webcam", cap
    cap.release()
    print("[VISION] Keine Kamera gefunden — wechsle in Digital Twin-Modus.")

    # 3. Fallback: Simulation
    return "sim", _SimBall()


# ==============================================================================
# FRAME EINLESEN (einheitliche Schnittstelle für alle Quellen)
# ==============================================================================
def _read_frame(cam_type: str, cam_obj) -> np.ndarray | None:
    if cam_type == "sim":
        return cam_obj.render()

    if cam_type == "picam2":
        # Picamera2 liefert je nach Konfiguration RGB oder BGR.
        # Mit format="BGR888" kommt BGR — kein Konvertierung nötig.
        frame = cam_obj.capture_array()
        # Sicherheits-Check: falls doch RGB geliefert wird, korrigieren
        # (passiert bei manchen Pi-OS-Versionen mit alten Picamera2-Treibern)
        return frame

    if cam_type == "webcam":
        ret, frame = cam_obj.read()
        if not ret or frame is None:
            return None
        # Webcam liefert immer BGR, ggf. auf Ziel-Auflösung skalieren
        if frame.shape[1] != RES_WIDTH or frame.shape[0] != RES_HEIGHT:
            frame = cv2.resize(frame, (RES_WIDTH, RES_HEIGHT),
                               interpolation=cv2.INTER_NEAREST)
        return frame

    return None


# ==============================================================================
# KAMERA SCHLIESSEN
# ==============================================================================
def _close_camera(cam_type: str, cam_obj):
    if cam_type == "picam2":
        try:
            cam_obj.stop()
        except Exception:
            pass
    elif cam_type == "webcam":
        try:
            cam_obj.release()
        except Exception:
            pass


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # ── Argument-Parsing ──────────────────────────────────────────────────────
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-output", action="store_true",
                    help="Alle stdout-Ausgaben unterdrücken")
    args = ap.parse_args()
    if args.no_output:
        sys.stdout = open(os.devnull, "w")

    # ── TelemetryBroker ───────────────────────────────────────────────────────
    mb = TelemetryBroker()

    # ── Initialer Broker-Zustand lesen ────────────────────────────────────────
    try:
        val = mb.get("robot_position")
        if val:
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
        if val:
            _sim_state = json.loads(val)
    except Exception:
        pass

    # ── Broker-Callback-Thread starten ────────────────────────────────────────
    mb.setcallback(["robot_position", "imu_pitch", "sim_state"], _on_broker_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    # ── Kamera öffnen ─────────────────────────────────────────────────────────
    cam_type, cam_obj = _init_camera()
    print(f"[VISION] Censor-Box aktiv: obere {CENSOR_TOP_HEIGHT_PX} Pixel werden ignoriert.")
    print("[VISION] Vision-System läuft.")

    last_log_t = time.time()

    try:
        while True:
            frame = _read_frame(cam_type, cam_obj)

            if frame is None:
                print("[VISION] FEHLER: Kamera-Verbindung verloren!")
                break

            # ── Bildverarbeitung ──────────────────────────────────────────────
            result = _process_frame(frame)

            if result["command"] != "NO_BALL":
                # AEMA-Glättung
                result["distance_cm"] = round(_aema_dist.update(result["distance_cm"]), 1)
                result["angle_deg"]   = round(_aema_angle.update(result["angle_deg"]),   1)
                result["x_center"]    = round(_aema_x.update(result["x_center"]),        1)
                result["radius"]      = round(_aema_radius.update(result["radius"]),      1)

                # Richtungs-Kommando
                error_x = result["x_center"] - CENTER_X
                if error_x < -DEADZONE_PIXELS:
                    result["command"] = "LEFT"
                elif error_x > DEADZONE_PIXELS:
                    result["command"] = "RIGHT"
                else:
                    result["command"] = "FORWARD"
            else:
                _reset_filters()

            # ── Globale Position ──────────────────────────────────────────────
            result["global_pos"] = (
                _compute_global_pos(result["distance_cm"], result["angle_deg"])
                if result["command"] != "NO_BALL" else None
            )

            # Simulations-Position anhängen (falls aktiv)
            if cam_type == "sim":
                result["sim_pos"] = cam_obj.pos

            # ── Broker-Publish ────────────────────────────────────────────────
            mb.set(BROKER_KEY, json.dumps(result))

            # ── Log (1 Hz) ────────────────────────────────────────────────────
            now = time.time()
            if now - last_log_t >= 1.0:
                if result["command"] == "NO_BALL":
                    print("[VISION] Status: NO_BALL")
                else:
                    print(f"[VISION] Status: {result['command']} | "
                          f"Distanz: {result['distance_cm']:.1f} cm | "
                          f"Winkel: {result['angle_deg']:.1f}°")
                last_log_t = now

    except KeyboardInterrupt:
        print("\n[VISION] Gestoppt.")
    finally:
        _close_camera(cam_type, cam_obj)
        mb.close()
        print("[VISION] Kamera geschlossen. System beendet.")