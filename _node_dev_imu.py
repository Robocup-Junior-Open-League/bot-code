import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--no-output", action="store_true")
if _ap.parse_args().no_output:
    sys.stdout = open(os.devnull, "w")

import time
import math
import random
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor

# ── Configuration ──────────────────────────────────────────────────────────────
I2C_ADDRESS = 0x4b
RESET_PIN   = None   # set below after conditional hardware import
POLL_RATE   = 0.01   # seconds (100 Hz)
# ──────────────────────────────────────────────────────────────────────────────

_hw_available = False
try:
    import board
    import busio
    import digitalio
    from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR, PacketError
    from adafruit_bno08x.i2c import BNO08X_I2C
    RESET_PIN     = board.D17
    _hw_available = True
except ImportError as e:
    print(f"[IMU] Hardware libraries not available ({e}) — using simulated pitch.")


# ── I2C baudrate check ─────────────────────────────────────────────────────────
def _check_baudrate():
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


# ── Hardware reset ─────────────────────────────────────────────────────────────
def _reset_sensor(reset_pin):
    print("[IMU] Resetting sensor...")
    reset_pin.value = False
    time.sleep(0.2)
    reset_pin.value = True
    time.sleep(0.8)


# ── Sensor init ────────────────────────────────────────────────────────────────
def _init_bno(i2c, reset_pin):
    _reset_sensor(reset_pin)
    try:
        bno = BNO08X_I2C(i2c, address=I2C_ADDRESS)
        bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        time.sleep(0.05)
        print("[IMU] Sensor initialised.")
        return bno
    except Exception as e:
        print(f"[IMU] Init failed: {e}")
        return None


# ── Quaternion → pitch ─────────────────────────────────────────────────────────
def _quaternion_to_pitch(i, j, k, w):
    sinp = 2 * (w * j - k * i)
    return math.degrees(
        math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    )


# ── Simulated pitch (random walk) ──────────────────────────────────────────────
class _SimPitch:
    """Slow random walk bounded to ±30°."""
    def __init__(self):
        self._pitch = 0.0

    def read(self):
        self._pitch += random.uniform(-0.5, 0.5)
        self._pitch  = max(-30.0, min(30.0, self._pitch))
        return round(self._pitch, 2)


# ── Main ───────────────────────────────────────────────────────────────────────
broker = TelemetryBroker()
_perf  = PerfMonitor("node_dev_imu", broker=broker, print_every=1000)

if _hw_available:
    _check_baudrate()
    _reset_pin           = digitalio.DigitalInOut(RESET_PIN)
    _reset_pin.direction = digitalio.Direction.OUTPUT
    _reset_pin.value     = True
    _i2c                 = busio.I2C(board.SCL, board.SDA)
    _sensor              = _init_bno(_i2c, _reset_pin)
    _sim                 = None
    print("[IMU] Starting pitch stream (hardware)...")
else:
    _sensor = None
    _sim    = _SimPitch()
    print("[IMU] Starting pitch stream (simulated)...")

while True:
    # ── Hardware reinit if needed ──────────────────────────────────────────────
    if _hw_available and _sensor is None:
        _sensor = _init_bno(_i2c, _reset_pin)
        time.sleep(1)
        continue

    try:
        with _perf.measure("poll"):
            if _sim is not None:
                pitch = _sim.read()
            else:
                quat = _sensor.quaternion
                if quat is None:
                    time.sleep(POLL_RATE)
                    continue
                pitch = _quaternion_to_pitch(*quat)
                pitch = round(pitch, 2)

            broker.set("imu_pitch", str(pitch))
        time.sleep(POLL_RATE)

    except KeyboardInterrupt:
        print("\n[IMU] Stopped.")
        broker.close()
        break
    except Exception as e:
        print(f"[IMU] {type(e).__name__}: {e} — reinitialising...")
        _sensor = None
