import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import time
import math
import random
import json
import queue
import threading

from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
from utils import lidar_read_usb, lidar_read_uart, lidar_sim

# ── IMU configuration ─────────────────────────────────────────────────────────
I2C_ADDRESS   = 0x4b
IMU_POLL_RATE = 0.01   # seconds (100 Hz)

# ── Lidar configuration ───────────────────────────────────────────────────────
LIDAR_SIM_REPLACE = True
LIDAR_BATCH_SIZE  = 360

# ─────────────────────────────────────────────────────────────────────────────

_hw_imu_available = False
_imu_reset_pin_id = None
try:
    import board
    import busio
    import digitalio
    from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR, PacketError
    from adafruit_bno08x.i2c import BNO08X_I2C
    _imu_reset_pin_id = board.D17
    _hw_imu_available = True
except ImportError as e:
    print(f"[SENSOR] IMU hardware libraries not available ({e}) — using simulated pitch.")

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_sensor", broker=mb, print_every=500)

# Shared: IMU thread writes, lidar sim reads via get_heading lambda
_imu_pitch   = None
_angle_dict  = {}
_batch_count = 0


# ── IMU helpers ───────────────────────────────────────────────────────────────

def _check_imu_baudrate():
    try:
        with open("/sys/class/i2c-adapter/i2c-1/of_node/clock-frequency", "rb") as f:
            current_baud = int.from_bytes(f.read(), byteorder="big")
        print(f"[SENSOR/IMU] I2C baudrate: {current_baud} Hz")
        if current_baud > 50000:
            print("!" * 60)
            print("WARNING: I2C baudrate is too high for BNO08x.")
            print("Set 'dtparam=i2c_arm=on,i2c_arm_baudrate=40000' in /boot/config.txt")
            print("!" * 60)
    except Exception:
        print("[SENSOR/IMU] Could not read I2C baudrate, skipping check.")


def _reset_imu(reset_pin):
    print("[SENSOR/IMU] Resetting sensor...")
    reset_pin.value = False
    time.sleep(0.2)
    reset_pin.value = True
    time.sleep(0.8)


def _init_bno(i2c, reset_pin):
    _reset_imu(reset_pin)
    try:
        bno = BNO08X_I2C(i2c, address=I2C_ADDRESS)
        bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
        time.sleep(0.05)
        print("[SENSOR/IMU] Sensor initialised.")
        return bno
    except Exception as e:
        print(f"[SENSOR/IMU] Init failed: {e}")
        return None


def _quaternion_to_pitch(i, j, k, w):
    sinp = 2 * (w * j - k * i)
    return math.degrees(
        math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    )


class _SimPitch:
    def __init__(self):
        self._pitch = 0.0

    def read(self):
        self._pitch += random.uniform(-0.5, 0.5)
        self._pitch  = max(-30.0, min(30.0, self._pitch))
        return round(self._pitch, 2)


def _imu_loop(sensor_ref, sim):
    """Continuously polls IMU and publishes imu_pitch. Runs in a daemon thread."""
    global _imu_pitch
    while True:
        if _hw_imu_available and sensor_ref[0] is None:
            sensor_ref[0] = _init_bno(sensor_ref[1], sensor_ref[2])
            time.sleep(1)
            continue
        try:
            with _perf.measure("imu"):
                if sim is not None:
                    pitch = sim.read()
                else:
                    quat = sensor_ref[0].quaternion
                    if quat is None:
                        time.sleep(IMU_POLL_RATE)
                        continue
                    pitch = round(_quaternion_to_pitch(*quat), 2)
                _imu_pitch = pitch
                mb.set("imu_pitch", str(pitch))
            time.sleep(IMU_POLL_RATE)
        except Exception as e:
            print(f"[SENSOR/IMU] {type(e).__name__}: {e} — reinitialising...")
            sensor_ref[0] = None


# ── Lidar helpers ─────────────────────────────────────────────────────────────

def _on_measurement(angle, distance, quality):
    global _batch_count
    _angle_dict[int(round(angle))] = distance
    _batch_count += 1
    if _batch_count >= LIDAR_BATCH_SIZE:
        with _perf.measure("lidar_hw"):
            mb.set("lidar", json.dumps(_angle_dict))
        _batch_count = 0


def _on_scan(batch):
    with _perf.measure("lidar_sim"):
        _angle_dict.update(batch)
        mb.set("lidar", json.dumps(_angle_dict))


def _on_sim_state(rx, ry, obs_snap):
    mb.set("sim_state", json.dumps({
        "robot":     [round(rx, 4), round(ry, 4)],
        "obstacles": [[round(float(p[0]), 4), round(float(p[1]), 4)]
                      for p in obs_snap],
    }))


if __name__ == "__main__":
    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    # ── IMU setup ─────────────────────────────────────────────────────────────
    if _hw_imu_available:
        _check_imu_baudrate()
        _reset_pin_hw           = digitalio.DigitalInOut(_imu_reset_pin_id)
        _reset_pin_hw.direction = digitalio.Direction.OUTPUT
        _reset_pin_hw.value     = True
        _i2c                    = busio.I2C(board.SCL, board.SDA)
        _sensor_ref             = [_init_bno(_i2c, _reset_pin_hw), _i2c, _reset_pin_hw]
        _sim_imu                = None
        print("[SENSOR/IMU] Starting pitch stream (hardware)...")
    else:
        _sensor_ref = [None, None, None]
        _sim_imu    = _SimPitch()
        print("[SENSOR/IMU] Starting pitch stream (simulated)...")

    threading.Thread(target=_imu_loop, args=(_sensor_ref, _sim_imu),
                     daemon=True, name="imu-loop").start()

    # ── Lidar setup ───────────────────────────────────────────────────────────
    raw_queue = queue.Queue(maxsize=36000)

    for _reader in (lidar_read_usb, lidar_read_uart):
        try:
            producer = _reader.start_producer(raw_queue)
            print(f"[SENSOR/LIDAR] Sensor opened on {_reader.PORT}")
            print("[SENSOR/LIDAR] Reading measurements (Ctrl+C to stop)...")
            try:
                while True:
                    result = _reader.parse_packet(raw_queue.get())
                    if result:
                        _on_measurement(*result)
            except KeyboardInterrupt:
                print("\n[SENSOR] Stopped.")
                producer.stop()
                mb.close()
            break
        except _reader.SensorUnavailableError as e:
            print(f"[SENSOR/LIDAR] {_reader.PORT} not available: {e}")
    else:
        if LIDAR_SIM_REPLACE:
            print("[SENSOR/LIDAR] Falling back to simulated sensor data.")
            try:
                lidar_sim.read_lidar_data(
                    _on_measurement,
                    get_heading=lambda: _imu_pitch if _imu_pitch is not None else 0.0,
                    on_scan=_on_scan,
                    on_state=_on_sim_state,
                )
            except KeyboardInterrupt:
                print("\n[SENSOR] Stopped.")
                mb.close()
        else:
            raise lidar_read_usb.SensorUnavailableError("No sensor found on any port.")
