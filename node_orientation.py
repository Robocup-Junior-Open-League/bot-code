import time
import math
import json
import board
import numpy as np
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_LINEAR_ACCELERATION,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_ROTATION_VECTOR,
    BNO_REPORT_GRAVITY,
)
from robus_core.libs.lib_telemtrybroker import TelemetryBroker

# ── Configuration ─────────────────────────────────────────────────────────────
PUBLISH_ACCELERATION = True   # Linear acceleration (m/s²) → "linear_acceleration"
PUBLISH_ANGULAR_VEL  = True   # Angular velocity (°/s)    → "angular_velocity"
PUBLISH_ORIENTATION  = True   # Roll / pitch / yaw (°)    → "orientation"
AUTOCALIBRATE        = True   # Align axes to startup gravity and heading;
                               # publishes additional "*_cal" keys for each active value
UPDATE_RATE_HZ       = 20
# ──────────────────────────────────────────────────────────────────────────────

_SLEEP        = 1.0 / UPDATE_RATE_HZ
_NEED_QUAT    = PUBLISH_ORIENTATION or AUTOCALIBRATE

# ── Sensor setup ──────────────────────────────────────────────────────────────
i2c = board.I2C()
bno = BNO08X_I2C(i2c, address=0x4b)

if _NEED_QUAT:
    bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
if PUBLISH_ACCELERATION:
    bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
if PUBLISH_ANGULAR_VEL:
    bno.enable_feature(BNO_REPORT_GYROSCOPE)
if AUTOCALIBRATE:
    bno.enable_feature(BNO_REPORT_GRAVITY)

mb = TelemetryBroker()

# ── Math helpers ──────────────────────────────────────────────────────────────
def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _quat_to_matrix(q):
    """Quaternion (x, y, z, w) → 3×3 rotation matrix (body → world)."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _quaternion_to_euler(q):
    """Quaternion (x, y, z, w) → (roll, pitch, yaw) in degrees. Yaw in [0, 360)."""
    x, y, z, w = q
    roll  = math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    sinp  = 2*(w*y - z*x)
    pitch = math.degrees(
        math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    )
    yaw   = math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))) % 360
    return roll, pitch, yaw


# ── Autocalibration setup ─────────────────────────────────────────────────────
# Calibrated frame axes:
#   Z  = up (opposite to gravity vector)
#   X  = startup forward direction, projected onto the horizontal plane
#   Y  = startup right  (= X × Z)
#
# _R_cal rows are the calibrated axes expressed in the sensor fusion world frame,
# giving the transform:  v_cal = _R_cal @ v_world
_R_cal       = None
_startup_yaw = None   # degrees; used for the orientation_cal yaw offset

if AUTOCALIBRATE:
    print("Calibrating axes – hold sensor still...")
    time.sleep(1.0)

    gravity_raw, q_startup = None, None
    while gravity_raw is None or q_startup is None:
        gravity_raw = bno.gravity
        q_startup   = bno.quaternion
        time.sleep(0.05)

    # Z axis: opposite to gravity = "up"
    Z_cal = _normalize(-np.array(gravity_raw))

    # X axis: sensor +X at startup, projected onto the horizontal plane
    R_startup    = _quat_to_matrix(q_startup)
    sensor_fwd   = R_startup[:, 0]                                # sensor +X in world frame
    X_cal        = _normalize(sensor_fwd - np.dot(sensor_fwd, Z_cal) * Z_cal)

    # Y axis: startup right = X × Z
    Y_cal        = _normalize(np.cross(X_cal, Z_cal))

    _R_cal       = np.array([X_cal, Y_cal, Z_cal])                # world → calibrated

    _, _, _startup_yaw = _quaternion_to_euler(q_startup)
    print(f"Calibration done.  Startup yaw: {_startup_yaw:.1f}°")
    print(f"  Z (up):      {Z_cal.round(3)}")
    print(f"  X (forward): {X_cal.round(3)}")
    print(f"  Y (right):   {Y_cal.round(3)}")


def _to_cal_frame(v_body, q_current):
    """Rotate a body-frame vector into the calibrated world frame."""
    v_world = _quat_to_matrix(q_current) @ np.array(v_body)
    return _R_cal @ v_world


# ── Main loop ─────────────────────────────────────────────────────────────────
print("Orientation node running (Ctrl+C to stop)...")

try:
    while True:
        q = bno.quaternion if _NEED_QUAT else None

        # Skip this cycle if the fusion hasn't produced a quaternion yet
        if _NEED_QUAT and q is None:
            time.sleep(_SLEEP)
            continue

        data = {}

        # ── Orientation ───────────────────────────────────────────────────────
        if PUBLISH_ORIENTATION:
            roll, pitch, yaw = _quaternion_to_euler(q)
            data["orientation"] = json.dumps({
                "roll":  round(roll,  1),
                "pitch": round(pitch, 1),
                "yaw":   round(yaw,   1),
            })
            if AUTOCALIBRATE:
                # Roll and pitch are gravity-referenced (same in cal frame).
                # Yaw is offset so that startup heading = 0°.
                data["orientation_cal"] = json.dumps({
                    "roll":  round(roll,  1),
                    "pitch": round(pitch, 1),
                    "yaw":   round((yaw - _startup_yaw) % 360, 1),
                })

        # ── Linear acceleration ───────────────────────────────────────────────
        if PUBLISH_ACCELERATION:
            la = bno.linear_acceleration
            if la is not None:
                data["linear_acceleration"] = json.dumps({
                    "x": round(la[0], 3),
                    "y": round(la[1], 3),
                    "z": round(la[2], 3),
                })
                if AUTOCALIBRATE:
                    cx, cy, cz = _to_cal_frame(la, q)
                    data["linear_acceleration_cal"] = json.dumps({
                        "x": round(float(cx), 3),
                        "y": round(float(cy), 3),
                        "z": round(float(cz), 3),
                    })

        # ── Angular velocity ──────────────────────────────────────────────────
        if PUBLISH_ANGULAR_VEL:
            gyr = bno.gyro    # rad/s in body frame
            if gyr is not None:
                data["angular_velocity"] = json.dumps({
                    "x": round(math.degrees(gyr[0]), 2),
                    "y": round(math.degrees(gyr[1]), 2),
                    "z": round(math.degrees(gyr[2]), 2),
                })
                if AUTOCALIBRATE:
                    cx, cy, cz = _to_cal_frame(gyr, q)
                    data["angular_velocity_cal"] = json.dumps({
                        "x": round(math.degrees(float(cx)), 2),
                        "y": round(math.degrees(float(cy)), 2),
                        "z": round(math.degrees(float(cz)), 2),
                    })

        if data:
            mb.setmulti(data)

        time.sleep(_SLEEP)

except KeyboardInterrupt:
    print("\nOrientation node stopped.")
    mb.close()
