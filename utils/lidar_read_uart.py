import threading
import queue as _queue_module

try:
    import serial
    _serial_import_error = None
except ImportError as e:
    print(e, "\nTrying pyserial...")
    try:
        import pyserial as serial
        _serial_import_error = None
    except ImportError as e:
        print(e, "\nBoth serial and pyserial failed.")
        serial = None
        _serial_import_error = e

PORT = '/dev/ttyS0'   # UART0 on Raspberry Pi
BAUD = 460800

_PACKET_SIZE = 5
_READ_CHUNK = 250  # 50 packets per read; must stay a multiple of _PACKET_SIZE


class SensorUnavailableError(Exception):
    pass


def parse_packet(raw: bytes):
    """
    Parse a raw 5-byte RPLidar C1 packet.
    Returns (angle, distance, quality) if the packet passes validity checks,
    or None if it should be discarded.
    """
    if len(raw) < _PACKET_SIZE:
        return None

    quality   = raw[0] >> 2
    angle_raw = (raw[1] >> 1) | (raw[2] << 7)
    angle     = int(round(angle_raw / 64.0))
    dist_raw  = raw[3] | (raw[4] << 8)
    distance  = int(dist_raw / 4.0)   # C1 uses quarter-millimetres

    # 128 is a sentinel for invalid distance; quality < 30 is unreliable
    if quality >= 30 and 0.0 <= angle <= 360.0 and 50 <= distance <= 12000 and distance != 128:
        return angle, distance, quality
    return None


def start_producer(raw_queue: _queue_module.Queue) -> threading.Thread:
    """
    Open the sensor and start a background daemon thread that reads raw bytes
    from the serial port as fast as possible, placing 5-byte packets onto
    raw_queue.

    Raises SensorUnavailableError immediately (before spawning the thread) if
    the serial module is missing or the port cannot be opened.

    Returns the running thread; call thread.stop() to request a clean shutdown.
    """
    if serial is None:
        raise SensorUnavailableError(
            f"serial module not available: {_serial_import_error}"
        )

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
    except serial.SerialException as e:
        raise SensorUnavailableError(f"Sensor not found on {PORT}: {e}") from e

    ser.flushInput()
    print("Starting scan...")
    ser.write(b'\xa5\x20')                        # Scan command
    descriptor = ser.read(7)                       # 7-byte response header
    print(f"Lidar response: {descriptor.hex()}")

    stop_event = threading.Event()

    def _producer():
        leftover = b''
        print("Producer thread running...")
        try:
            while not stop_event.is_set():
                chunk = ser.read(_READ_CHUNK)
                if not chunk:
                    continue
                data = leftover + chunk
                i = 0
                while i + _PACKET_SIZE <= len(data):
                    try:
                        raw_queue.put_nowait(data[i:i + _PACKET_SIZE])
                    except _queue_module.Full:
                        pass  # Consumer is too slow; drop the packet
                    i += _PACKET_SIZE
                leftover = data[i:]   # At most 4 bytes; kept for next iteration
        finally:
            ser.write(b'\xa5\x25')   # Stop command
            ser.close()
            print("Producer thread stopped.")

    thread = threading.Thread(target=_producer, daemon=True, name="lidar-producer")
    thread.stop = stop_event.set
    thread.start()
    return thread
