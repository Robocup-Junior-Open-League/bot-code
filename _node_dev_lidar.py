from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils import lidar_read_usb, lidar_read_uart, lidar_sim
from utils.perf_monitor import PerfMonitor
import json
import queue
import threading

SIM_REPLACE = True  # Use simulation if sensor is not found
BATCH_SIZE  = 360   # Publish to broker every N measurements

mb    = TelemetryBroker()
_perf = PerfMonitor("node_dev_lidar", broker=mb, print_every=50)

angle_dict   = {}
_batch_count = 0
_imu_pitch   = None  # degrees — read from broker, never written here


def _on_broker_update(key, value):
    global _imu_pitch
    if key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except (ValueError, TypeError):
            pass


def on_measurement(angle, distance, quality):
    global _batch_count
    angle_dict[int(round(angle))] = distance
    _batch_count += 1
    if _batch_count >= BATCH_SIZE:
        with _perf.measure("hardware"):
            mb.set("lidar", json.dumps(angle_dict))
        _batch_count = 0


def on_scan(batch):
    """Batch callback for simulation: receives a full {angle: dist_mm} dict at once."""
    with _perf.measure("sim"):
        angle_dict.update(batch)
        mb.set("lidar", json.dumps(angle_dict))


if __name__ == "__main__":
    import argparse, sys, os
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--no-output", action="store_true")
    if _ap.parse_args().no_output:
        sys.stdout = open(os.devnull, "w")

    # Seed imu_pitch and subscribe — runs in a daemon thread so it stays live
    # alongside whichever blocking path (hardware loop or sim) runs below.
    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass
    mb.setcallback(["imu_pitch"], _on_broker_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    raw_queue = queue.Queue(maxsize=36000)  # ~10 full scans of headroom

    # Try USB first, then UART0, then simulation.
    for _reader in (lidar_read_usb, lidar_read_uart):
        try:
            producer = _reader.start_producer(raw_queue)
            print(f"Sensor opened on {_reader.PORT}")
            print("Reading measurements (Ctrl+C to stop)...")
            try:
                while True:
                    result = _reader.parse_packet(raw_queue.get())
                    if result:
                        on_measurement(*result)
            except KeyboardInterrupt:
                print("\nStopping...")
                producer.stop()
            break
        except _reader.SensorUnavailableError as e:
            print(f"[{_reader.PORT}] not available: {e}")
    else:
        if SIM_REPLACE:
            print("Falling back to simulated sensor data.")

            def on_sim_state(rx, ry, obs_snap):
                mb.set("sim_state", json.dumps({
                    "robot":     [round(rx, 4), round(ry, 4)],
                    "obstacles": [[round(float(p[0]), 4), round(float(p[1]), 4)]
                                  for p in obs_snap],
                }))

            lidar_sim.read_lidar_data(
                on_measurement,
                get_heading=lambda: _imu_pitch if _imu_pitch is not None else 0.0,
                on_scan=on_scan,
                on_state=on_sim_state,
                # Per-ray fallback (realistic drip-feed, matches real hardware):
                # on_scan=None,
            )
        else:
            raise lidar_read_usb.SensorUnavailableError("No sensor found on any port.")
