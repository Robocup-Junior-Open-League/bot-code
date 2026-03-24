import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import time
import math
from robus_core.libs.lib_telemtrybroker import TelemetryBroker

try:
    import board
    import busio
    import digitalio
    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_ROTATION_VECTOR,
        PacketError,
    )
    from adafruit_bno08x.i2c import BNO08X_I2C
except ImportError as e:
    print(f"[IMU] Hardware libraries not available ({e}) — is this running on the robot? Exiting.")
    sys.exit(0)

# --- CONFIG ---
I2C_ADDRESS  = 0x4b
RESET_PIN    = board.D17
POLL_RATE    = 0.01   # seconds (100 Hz)

# --- BAUDRATE CHECK ---
def check_baudrate():
    try:
        with open("/sys/class/i2c-adapter/i2c-1/of_node/clock-frequency", "rb") as f:
            current_baud = int.from_bytes(f.read(), byteorder="big")
        print(f"[IMU] I2C baudrate: {current_baud} Hz")
        if current_baud > 50000:
            print("!" * 60)
            print("WARNING: I2C baudrate is too high for BNO08x.")
            print("Set 'dtparam=i2c_arm=on,i2c_arm_baudrate=40000' in /boot/config.txt")
            print("!" * 60)
    except Exception:
        print("[IMU] Could not read I2C baudrate, skipping check.")

# --- HARDWARE RESET ---
def reset_sensor(reset_pin):
    print("[IMU] Resetting sensor...")
    reset_pin.value = False
    time.sleep(0.2)
    reset_pin.value = True
    time.sleep(0.8)

# --- SENSOR INIT ---
def init_bno(i2c, reset_pin):
    reset_sensor(reset_pin)
    try:
        bno = BNO08X_I2C(i2c, address=I2C_ADDRESS)
        bno.enable_feature(BNO_REPORT_ACCELEROMETER)
        time.sleep(0.1)
        bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        print("[IMU] Sensor initialised.")
        return bno
    except Exception as e:
        print(f"[IMU] Init failed: {e}")
        return None

# --- QUATERNION → EULER ---
def quaternion_to_euler(i, j, k, w):
    sinr_cosp = 2 * (w * i + j * k)
    cosr_cosp = 1 - 2 * (i * i + j * j)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * j - k * i)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

    siny_cosp = 2 * (w * k + i * j)
    cosy_cosp = 1 - 2 * (j * j + k * k)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    yaw_deg = math.degrees(yaw)
    if yaw_deg < 0:
        yaw_deg += 360  # normalise to 0–360

    return math.degrees(roll), math.degrees(pitch), yaw_deg

# --- MAIN ---
check_baudrate()

broker = TelemetryBroker()

reset_pin = digitalio.DigitalInOut(RESET_PIN)
reset_pin.direction = digitalio.Direction.OUTPUT
reset_pin.value = True

i2c    = busio.I2C(board.SCL, board.SDA)
sensor = init_bno(i2c, reset_pin)

print("[IMU] Starting orientation stream...")

while True:
    if sensor is None:
        sensor = init_bno(i2c, reset_pin)
        time.sleep(1)
        continue

    try:
        accel_x, accel_y, accel_z   = sensor.acceleration
        quat_i, quat_j, quat_k, quat_w = sensor.quaternion

        roll, pitch, yaw = quaternion_to_euler(quat_i, quat_j, quat_k, quat_w)

        broker.setmulti({
            "imu_roll":    round(roll,  2),
            "imu_pitch":   round(pitch, 2),
            "imu_yaw":     round(yaw,   2),
            "imu_quat_i":  round(quat_i, 4),
            "imu_quat_j":  round(quat_j, 4),
            "imu_quat_k":  round(quat_k, 4),
            "imu_quat_w":  round(quat_w, 4),
            "imu_accel_x": round(accel_x, 3),
            "imu_accel_y": round(accel_y, 3),
            "imu_accel_z": round(accel_z, 3),
        })

        time.sleep(POLL_RATE)

    except KeyboardInterrupt:
        print("\n[IMU] Stopped.")
        broker.close()
        break
    except Exception as e:
        print(f"[IMU] {type(e).__name__}: {e} — reinitialising...")
        sensor = None
