"""
Modular cooperation data readers.

To swap the transport layer, subclass BaseCooperationReader and implement
start() and stop(), then return an instance from _make_reader() in
node_cooperation.py.

Expected frame schema (one JSON object per newline for SerialCooperationReader):

    {
      "main_robot_pos": {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pos_1":    {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pos_2":    {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pos_3":    {"x": <float>, "y": <float>, "confidence": <float>},
      "ball_pos":       {"x": <float>, "y": <float>, "confidence": <float>},
      "ball_pred":      {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pred_1":   {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pred_2":   {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pred_3":   {"x": <float>, "y": <float>, "confidence": <float>}
    }

Detections (other_pos_*, ball_pos) are freshly observed positions.
Predictions (other_pred_*, ball_pred) are forward-projected estimates for
objects not detected this frame.  ball_pos and ball_pred are mutually
exclusive: send whichever applies.  All fields are optional; missing ones
are simply ignored by the node.
"""

import json
import threading
import random as _random

try:
    import serial as _serial
    _serial_available = True
except ImportError:
    _serial_available = False


class BaseCooperationReader:
    """
    Abstract base for cooperation data readers.

    Subclasses must implement start(on_frame) and stop().
    on_frame(data: dict) is called from a background thread for each frame.
    send(data: dict) transmits a frame to the remote; default is a no-op.
    """

    def start(self, on_frame):
        """Start reading.  Must return immediately; run I/O on a background thread."""
        raise NotImplementedError

    def stop(self):
        """Signal the reader to stop and release all resources."""
        raise NotImplementedError

    def send(self, data: dict):
        """Send a JSON frame to the remote.  No-op by default."""
        pass


class SerialCooperationReader(BaseCooperationReader):
    """Reads newline-delimited JSON frames from a serial port."""

    DEFAULT_PORT = "/dev/ttyUSB1"
    DEFAULT_BAUD = 115200

    def __init__(self, port=None, baud=None):
        if not _serial_available:
            raise ImportError("pyserial is required for SerialCooperationReader")
        self._port    = port or self.DEFAULT_PORT
        self._baud    = baud or self.DEFAULT_BAUD
        self._stop_ev = threading.Event()
        self._thread  = None
        self._ser     = None
        self._ser_lock = threading.Lock()

    def start(self, on_frame):
        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_frame,),
            daemon=True, name="coop-serial",
        )
        self._thread.start()

    def stop(self):
        self._stop_ev.set()

    def send(self, data: dict):
        """Write a JSON frame (newline-terminated) to the serial port."""
        with self._ser_lock:
            ser = self._ser
        if ser is None:
            return
        try:
            line = json.dumps(data, separators=(",", ":")) + "\n"
            with self._ser_lock:
                ser.write(line.encode("utf-8"))
        except Exception as e:
            print(f"[COOP] Send error: {e}")

    def _run(self, on_frame):
        try:
            ser = _serial.Serial(self._port, self._baud, timeout=1)
            print(f"[COOP] Serial opened on {self._port} at {self._baud} baud.")
        except _serial.SerialException as e:
            print(f"[COOP] Could not open {self._port}: {e}")
            return

        with self._ser_lock:
            self._ser = ser

        buf = b""
        try:
            while not self._stop_ev.is_set():
                chunk = ser.read(ser.in_waiting or 1)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8", errors="replace"))
                        on_frame(data)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"[COOP] Parse error: {e} — {line[:80]}")
        finally:
            with self._ser_lock:
                self._ser = None
            ser.close()
            print("[COOP] Serial closed.")


class SimCooperationReader(BaseCooperationReader):
    """
    Simulation fallback: generates cooperation frames by applying Gaussian
    jitter to simulator state positions.

    *get_sim_state* and *get_ball_sim* are zero-argument callables returning
    the current sim_state dict and ball sim_pos dict respectively (or None).

    The ally robot slot is chosen randomly at startup and stays consistent
    for the process lifetime.
    """

    RATE_HZ    = 10.0
    JITTER_STD = 0.03  # metres

    def __init__(self, get_sim_state, get_ball_sim):
        self._get_sim_state = get_sim_state
        self._get_ball_sim  = get_ball_sim
        self._ally_idx      = _random.randint(0, 2)
        self._stop_ev       = threading.Event()
        self._thread        = None

    def start(self, on_frame):
        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_frame,),
            daemon=True, name="coop-sim",
        )
        self._thread.start()
        print(f"[COOP] No serial port found — using simulation fallback (ally slot {self._ally_idx}).")

    def stop(self):
        self._stop_ev.set()

    def _jitter(self, x, y):
        return (
            round(x + _random.gauss(0, self.JITTER_STD), 4),
            round(y + _random.gauss(0, self.JITTER_STD), 4),
        )

    def _run(self, on_frame):
        interval = 1.0 / self.RATE_HZ
        while not self._stop_ev.is_set():
            sim  = self._get_sim_state()
            ball = self._get_ball_sim()

            if sim is not None:
                obstacles = sim.get("obstacles", [])
                if obstacles:
                    ally_idx = self._ally_idx % len(obstacles)
                    ax, ay   = self._jitter(float(obstacles[ally_idx][0]),
                                            float(obstacles[ally_idx][1]))
                    data = {"main_robot_pos": {"x": ax, "y": ay}}

                    slot = 1
                    for i, obs in enumerate(obstacles):
                        if slot > 3:
                            break
                        if i == ally_idx:
                            continue
                        ox, oy = self._jitter(float(obs[0]), float(obs[1]))
                        d = ((ox - ax) ** 2 + (oy - ay) ** 2) ** 0.5
                        data[f"other_pos_{slot}"] = {"x": ox, "y": oy, "confidence": 5 / d}
                        slot += 1

                    if ball is not None:
                        try:
                            bx, by = self._jitter(float(ball["x"]), float(ball["y"]))
                            data["ball_pos"] = {"x": bx, "y": by}
                        except (KeyError, TypeError, ValueError):
                            pass

                    on_frame(data)

            self._stop_ev.wait(interval)
