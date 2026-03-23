from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from lidar_utils.lidar_vis import LiveVisualiser
import json

mb = TelemetryBroker()
vis = LiveVisualiser()

# Local mirror of broker state, updated by callbacks
_angle_dict = {}
_corners = []


def on_data_change(key, value):
    global _angle_dict, _corners
    if value is None:
        return

    if key == "lidar":
        raw = json.loads(value)
        # JSON keys are always strings — restore to int degrees
        _angle_dict = {int(k): v for k, v in raw.items()}
    elif key == "lidar_corners":
        _corners = json.loads(value)

    vis.update(_angle_dict, _corners)


if __name__ == "__main__":
    mb.setcallback(["lidar", "lidar_corners"], on_data_change)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping visualisation.")
        mb.close()
