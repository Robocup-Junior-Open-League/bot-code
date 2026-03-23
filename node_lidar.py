from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from lidar_utils.lidar_read import start_producer, parse_packet, SensorUnavailableError
from lidar_utils import lidar_sim
import json
import queue

SIM_REPLACE = True  # Use simulation if sensor is not found
BATCH_SIZE  = 360   # Publish to broker every N measurements

mb = TelemetryBroker()

angle_dict   = {}
_batch_count = 0


def on_measurement(angle, distance, quality):
    global _batch_count
    angle_dict[int(round(angle))] = distance
    _batch_count += 1
    if _batch_count >= BATCH_SIZE:
        mb.set("lidar", json.dumps(angle_dict))
        _batch_count = 0


if __name__ == "__main__":
    raw_queue = queue.Queue(maxsize=36000)  # ~10 full scans of headroom
    try:
        producer = start_producer(raw_queue)
        print("Reading measurements (Ctrl+C to stop)...")
        try:
            while True:
                result = parse_packet(raw_queue.get())
                if result:
                    on_measurement(*result)
        except KeyboardInterrupt:
            print("\nStopping...")
            producer.stop()
    except SensorUnavailableError:
        if SIM_REPLACE:
            print("Falling back to simulated sensor data.")
            lidar_sim.read_lidar_data(on_measurement)
        else:
            raise
