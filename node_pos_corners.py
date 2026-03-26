from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from lidar_utils.lidar_analysis import simple_corners
import json
import math

# Corners within this distance (metres) of each other are considered "similar".
# If the largest such group is a majority (> half of all detected corners),
# lone outliers outside that group are discarded.
OUTLIER_DIST = 0.15

mb = TelemetryBroker()


def _filter_outliers(corner_xy):
    """
    If more than 2 corners are detected and a majority cluster exists,
    discard corners that do not belong to it.
    """
    n = len(corner_xy)
    if n <= 2:
        return corner_xy

    # Build adjacency between corners within OUTLIER_DIST
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(corner_xy[i][0] - corner_xy[j][0],
                          corner_xy[i][1] - corner_xy[j][1]) <= OUTLIER_DIST:
                adj[i].add(j)
                adj[j].add(i)

    # Find connected components via BFS
    visited = [False] * n
    components = []
    for i in range(n):
        if visited[i]:
            continue
        component, queue = [], [i]
        visited[i] = True
        while queue:
            curr = queue.pop()
            component.append(curr)
            for nb in adj[curr]:
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        components.append(component)

    largest = max(components, key=len)
    if len(largest) > n / 2:
        return [corner_xy[i] for i in largest]
    return corner_xy


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

    corner_xy = _filter_outliers(simple_corners(points))
    corners = [
        (int(round(math.degrees(math.atan2(y, x)) % 360)), int(round(math.hypot(x, y) * 1000)))
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
