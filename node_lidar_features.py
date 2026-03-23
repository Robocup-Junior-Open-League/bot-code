from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from lidar_utils.lidar_analysis import simple_corners
import json
import math

mb = TelemetryBroker()


def on_lidar_update(key, value):
    if value is None:
        return

    raw = json.loads(value)
    sorted_angles = sorted(int(k) for k in raw.keys())
    points = [
        (
            (raw[str(a)] / 1000) * math.cos(math.radians(a)),
            (raw[str(a)] / 1000) * math.sin(math.radians(a))
        )
        for a in sorted_angles
    ]
    corner_xy = simple_corners(points)
    corners = [
        (math.degrees(math.atan2(y, x)) % 360, math.hypot(x, y) * 1000)
        for x, y in corner_xy
    ]
    mb.set("lidar_corners", json.dumps(corners))


if __name__ == "__main__":
    mb.setcallback(["lidar"], on_lidar_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping corner detection.")
        mb.close()
