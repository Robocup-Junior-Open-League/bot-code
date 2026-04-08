import math
import random
import threading
import time
import numpy as np

SIM_QUALITY      = 15              # simulated quality value passed to on_update
SIM_JITTER_MM    = 4               # Gaussian noise std dev (mm) per reading
SIM_MAX_SPEED    = 0.4             # m/s  — maximum speed (all robots)
SIM_OBS_ACCEL    = 0.3             # m/s² — random 2-D acceleration for obstacles
SIM_ROB_ACCEL    = 0.6             # m/s² — scalar accel/decel along robot heading
SIM_MAX_TURN_RAD = math.radians(90)  # rad/s — maximum heading turn rate
SIM_PHYSICS_HZ   = 60             # physics updates per second
SIM_SCAN_HZ      = 10             # complete lidar scan cycles per second

# ── Field geometry ─────────────────────────────────────────────────────────────
# Playing field (inside white line): FIELD_WIDTH × FIELD_HEIGHT = 1.58 × 2.19 m
# Outer area margin:  OUTER_MARGIN = 0.12 m on all sides (outside field dims)
# Total space:        1.82 × 2.43 m  (= playing field + 2 × outer margin)
# Goals (centred on each short side): GOAL_WIDTH = 0.60 m wide, in the outer margin.
#   The outer wall is the goal back wall.  Goal side walls run from the outer
#   wall to the playing-field border (white line).  No concave notch in outer wall.

OUTER_MARGIN = 0.12   # m  (exported — used by other nodes)


def _build_wall_segments(field_w, field_h):
    """
    Return wall segments for a playing field of given dimensions (white-line to
    white-line), with goals centred on each short side in the outer margin.

    The outer walls form a complete rectangle at ±OUTER_MARGIN beyond the playing
    field.  Goal side walls are the only additional structure; the outer wall
    itself serves as the goal back wall.

    Each segment is a tuple:
        (is_horiz, fixed_val, rng_min, rng_max)

    is_horiz=True  → horizontal wall at y=fixed_val, x ∈ [rng_min, rng_max]
    is_horiz=False → vertical   wall at x=fixed_val, y ∈ [rng_min, rng_max]

    All coordinates use the playing-field origin: (0,0) = bottom-left white-line
    corner.  The outer walls sit at −OUTER_MARGIN and field_w/h + OUTER_MARGIN.
    """
    ox0 = -OUTER_MARGIN               # left / bottom outer wall
    ox1 =  field_w + OUTER_MARGIN     # right outer wall
    oy0 = -OUTER_MARGIN               # bottom outer wall
    oy1 =  field_h + OUTER_MARGIN     # top outer wall

    return [
        # Outer walls — complete rectangle; the goal opening has no additional
        # obstructions at lidar height (goal side walls are below the beam).
        (False, ox0,  oy0, oy1),   # left
        (False, ox1,  oy0, oy1),   # right
        (True,  oy0,  ox0, ox1),   # bottom
        (True,  oy1,  ox0, ox1),   # top
    ]


# ── Ray casting ───────────────────────────────────────────────────────────────

def _cast_rays_np(px, py, heading_deg, obstacles, wall_segments, angles_deg, obstacle_radius):
    """
    Cast all rays at once using numpy.

    angles_deg    : 1-D numpy array of sensor angles in degrees.
    heading_deg   : scalar field-heading offset (degrees).
    wall_segments : list of (is_horiz, fixed_val, rng_min, rng_max) from
                    _build_wall_segments().  Coordinates are in the playing-field
                    frame: (0,0) = bottom-left white-line corner.
    Returns a 1-D numpy array of distances in metres, one per angle.
    """
    rad = np.radians(angles_deg + heading_deg)
    ux  = np.cos(rad)
    uy  = np.sin(rad)

    dists = np.full(len(angles_deg), np.inf)

    with np.errstate(divide='ignore', invalid='ignore'):
        for is_horiz, fixed_val, rng_min, rng_max in wall_segments:
            if is_horiz:
                # horizontal wall at y = fixed_val
                t     = np.where(np.abs(uy) > 1e-9, (fixed_val - py) / uy, np.inf)
                hit_x = px + t * ux
                valid = (t > 1e-9) & (hit_x >= rng_min) & (hit_x <= rng_max)
            else:
                # vertical wall at x = fixed_val
                t     = np.where(np.abs(ux) > 1e-9, (fixed_val - px) / ux, np.inf)
                hit_y = py + t * uy
                valid = (t > 1e-9) & (hit_y >= rng_min) & (hit_y <= rng_max)
            dists = np.where(valid, np.minimum(dists, t), dists)

    for ox, oy in obstacles:
        ocx  = px - ox
        ocy  = py - oy
        # a = 1 (unit directions), so the quadratic simplifies to:
        # t = (-b ± sqrt(b² - 4c)) / 2
        b      = 2 * (ocx * ux + ocy * uy)
        c      = ocx**2 + ocy**2 - obstacle_radius**2
        disc   = b**2 - 4 * c
        valid  = disc >= 0
        sqrt_d = np.sqrt(np.where(valid, disc, 0.0))
        t1     = (-b - sqrt_d) / 2
        t2     = (-b + sqrt_d) / 2
        t_hit  = np.where(valid & (t1 > 0), t1,
                 np.where(valid & (t2 > 0), t2, np.inf))
        dists  = np.minimum(dists, t_hit)

    return dists


def _cast_rays(px, py, angle_f, obstacles, wall_segments, step_size, obstacle_radius):
    """Full 360° scan — used by get_boundary_distances."""
    angles = np.arange(0, 360, step_size, dtype=float)
    dists  = _cast_rays_np(px, py, angle_f, obstacles, wall_segments,
                           angles, obstacle_radius)
    return list(zip(angles.astype(int), dists.tolist()))


# ── Physics ───────────────────────────────────────────────────────────────────

def _elastic_collide(pos_a, vel_a, pos_b, vel_b, sep):
    """Equal-mass elastic collision between two circles. Modifies in-place."""
    dx, dy = pos_b[0] - pos_a[0], pos_b[1] - pos_a[1]
    dist = math.hypot(dx, dy)
    if not (1e-9 < dist < sep):
        return
    nx, ny = dx / dist, dy / dist
    push   = (sep - dist) / 2
    pos_a[0] -= nx * push;  pos_a[1] -= ny * push
    pos_b[0] += nx * push;  pos_b[1] += ny * push
    dv = (vel_a[0] - vel_b[0]) * nx + (vel_a[1] - vel_b[1]) * ny
    vel_a[0] -= dv * nx;  vel_a[1] -= dv * ny
    vel_b[0] += dv * nx;  vel_b[1] += dv * ny


def _wall_bounce(pos, vel, radius, width, length):
    """Reflect off field boundaries. Modifies in-place."""
    if pos[0] < radius:
        pos[0] = radius;          vel[0] = abs(vel[0])
    elif pos[0] > width - radius:
        pos[0] = width - radius;  vel[0] = -abs(vel[0])
    if pos[1] < radius:
        pos[1] = radius;          vel[1] = abs(vel[1])
    elif pos[1] > length - radius:
        pos[1] = length - radius; vel[1] = -abs(vel[1])


def _physics_step(rob_pos, rob_vel, rob_heading, obs_pos, obs_vel,
                  robot_radius, obstacle_radius, width, length, dt):
    """Advance all robots one physics step. All pos/vel/heading lists modified in-place."""

    # ── Main robot: heading-based steering + scalar accel ─────────────────────
    rob_heading[0] += random.uniform(-SIM_MAX_TURN_RAD, SIM_MAX_TURN_RAD) * dt
    accel = random.uniform(-SIM_ROB_ACCEL, SIM_ROB_ACCEL) * dt
    rob_vel[0] += math.cos(rob_heading[0]) * accel
    rob_vel[1] += math.sin(rob_heading[0]) * accel
    speed = math.hypot(rob_vel[0], rob_vel[1])
    if speed > SIM_MAX_SPEED:
        s = SIM_MAX_SPEED / speed
        rob_vel[0] *= s;  rob_vel[1] *= s

    # ── Obstacles: batched random acceleration + vectorised speed clamp ─────────
    obs_vel += np.random.uniform(-SIM_OBS_ACCEL, SIM_OBS_ACCEL, obs_vel.shape) * dt
    speeds   = np.linalg.norm(obs_vel, axis=1)
    over     = speeds > SIM_MAX_SPEED
    if np.any(over):
        scale = np.where(over, SIM_MAX_SPEED / np.maximum(speeds, 1e-9), 1.0)
        obs_vel *= scale[:, np.newaxis]

    # ── Integrate positions ───────────────────────────────────────────────────
    rob_pos[0] += rob_vel[0] * dt
    rob_pos[1] += rob_vel[1] * dt
    obs_pos    += obs_vel * dt

    # ── Wall bounce — main robot (scalar) ─────────────────────────────────────
    _wall_bounce(rob_pos, rob_vel, robot_radius, width, length)

    # ── Wall bounce — obstacles (vectorised) ──────────────────────────────────
    lo_x, hi_x = obstacle_radius, width  - obstacle_radius
    lo_y, hi_y = obstacle_radius, length - obstacle_radius
    hit_x_lo = obs_pos[:, 0] < lo_x;  hit_x_hi = obs_pos[:, 0] > hi_x
    hit_y_lo = obs_pos[:, 1] < lo_y;  hit_y_hi = obs_pos[:, 1] > hi_y
    np.clip(obs_pos[:, 0], lo_x, hi_x, out=obs_pos[:, 0])
    np.clip(obs_pos[:, 1], lo_y, hi_y, out=obs_pos[:, 1])
    obs_vel[hit_x_lo, 0] =  np.abs(obs_vel[hit_x_lo, 0])
    obs_vel[hit_x_hi, 0] = -np.abs(obs_vel[hit_x_hi, 0])
    obs_vel[hit_y_lo, 1] =  np.abs(obs_vel[hit_y_lo, 1])
    obs_vel[hit_y_hi, 1] = -np.abs(obs_vel[hit_y_hi, 1])

    # ── All-pairs elastic collisions ──────────────────────────────────────────
    # Row views into obs_pos/obs_vel are in-place-modifiable numpy array views.
    all_pos = [rob_pos] + [obs_pos[i] for i in range(len(obs_pos))]
    all_vel = [rob_vel] + [obs_vel[i] for i in range(len(obs_vel))]
    all_rad = [robot_radius] + [obstacle_radius] * len(obs_pos)
    n = len(all_pos)
    for i in range(n):
        for j in range(i + 1, n):
            _elastic_collide(all_pos[i], all_vel[i],
                             all_pos[j], all_vel[j],
                             all_rad[i] + all_rad[j])


def _physics_loop(rob_pos, rob_vel, rob_heading, obs_pos, obs_vel,
                  robot_radius, obstacle_radius, width, length,
                  lock, stop_event):
    """Physics thread: runs _physics_step at SIM_PHYSICS_HZ."""
    dt = 1.0 / SIM_PHYSICS_HZ
    while not stop_event.is_set():
        t0 = time.monotonic()
        with lock:
            _physics_step(rob_pos, rob_vel, rob_heading, obs_pos, obs_vel,
                          robot_radius, obstacle_radius, width, length, dt)
        sleep = dt - (time.monotonic() - t0)
        if sleep > 0:
            time.sleep(sleep)


# ── Public interface ──────────────────────────────────────────────────────────

def read_lidar_data(on_update, on_ready=None, get_heading=None, on_scan=None, on_state=None,
                    width=1.58, length=2.19, step_size=1, proximity=0.1, robot_radius=0.09):
    """
    Simulates RPLidar C1 with independent physics and lidar threads.

    Physics runs at SIM_PHYSICS_HZ; the lidar spins through angles one ray at a
    time at SIM_SCAN_HZ, reading the current world state for each ray.

    Calls on_update(angle, distance_mm, quality) per ray.
    Calls on_ready(px, py, angle_f) once with the initial robot state.
    Calls on_heading(heading_deg) once per scan with the current robot heading.
    Stops on KeyboardInterrupt.
    """
    print("Starting simulated scan...")
    wall_segments = _build_wall_segments(width, length)
    px, py, angle_f, _, _, init_obstacles, _ = get_boundary_distances(
        width, length, step_size, proximity, robot_radius
    )
    print(f"  Robot position : ({px:.3f}, {py:.3f}) m")
    print(f"  Sensor heading : {angle_f:.1f}°")
    print(f"  Obstacles      : {[(round(ox, 3), round(oy, 3)) for ox, oy in init_obstacles]}")

    if on_ready:
        on_ready(px, py, angle_f)

    # Shared mutable state — all lists so physics thread edits in-place
    rob_pos     = [px, py]
    rob_vel     = [0.0, 0.0]
    rob_heading = [math.radians(angle_f)]   # stored in radians for physics
    obs_pos     = np.array([[ox, oy] for ox, oy in init_obstacles], dtype=float)
    obs_vel     = np.zeros((len(init_obstacles), 2))

    lock       = threading.Lock()
    stop_event = threading.Event()

    physics_thread = threading.Thread(
        target=_physics_loop,
        args=(rob_pos, rob_vel, rob_heading, obs_pos, obs_vel,
              robot_radius, proximity, width, length,
              lock, stop_event),
        daemon=True,
        name="sim-physics",
    )
    physics_thread.start()

    angles_np    = np.arange(0, 360, step_size, dtype=float)
    angles_int   = angles_np.astype(int).tolist()
    scan_interval = 1.0 / SIM_SCAN_HZ
    scan_count   = 0

    try:
        while True:
            t_scan = time.monotonic()

            # Single lock acquisition for the whole scan snapshot
            with lock:
                rx, ry           = rob_pos[0], rob_pos[1]
                _physics_heading = math.degrees(rob_heading[0])
                obs_snap         = obs_pos.copy()

            if on_state is not None:
                on_state(rx, ry, obs_snap)

            # Use externally-supplied heading if provided (e.g. from imu_pitch),
            # otherwise fall back to the internal physics heading.
            heading_deg = get_heading() if get_heading is not None else _physics_heading

            # Vectorised raycasting — all 360 rays in one numpy call
            dists_m  = _cast_rays_np(rx, ry, heading_deg, obs_snap,
                                     wall_segments, angles_np, proximity)
            dists_mm = (dists_m * 1000.0)
            if SIM_JITTER_MM > 0:
                dists_mm += np.random.normal(0.0, SIM_JITTER_MM, len(angles_np))
            dists_mm = dists_mm.astype(int).tolist()

            if on_scan:
                # Batch path: deliver the full scan as a single dict
                batch = {a: d for a, d in zip(angles_int, dists_mm)
                         if 50 <= d <= 12000}
                on_scan(batch)
            else:
                # Per-ray path: realistic drip-feed (matches real hardware contract)
                for angle_deg, dist_mm in zip(angles_int, dists_mm):
                    if 50 <= dist_mm <= 12000:
                        on_update(angle_deg, dist_mm, SIM_QUALITY)

            scan_count += 1
            if scan_count % SIM_SCAN_HZ == 0:   # print once per simulated second
                obs_log = [(round(p[0], 3), round(p[1], 3)) for p in obs_snap]
                src = "external" if get_heading is not None else "physics"
                print(f"  [SIM] robot=({rx:.3f}, {ry:.3f})"
                      f"  heading={heading_deg % 360:.1f}° ({src})"
                      f"  obs={obs_log}")

            # Sleep the remainder of the scan cycle
            remainder = scan_interval - (time.monotonic() - t_scan)
            if remainder > 0:
                time.sleep(remainder)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        stop_event.set()
        physics_thread.join(timeout=1.0)


def get_boundary_distances(width=1.58, length=2.19, step_size=1, proximity=0.1, robot_radius=0.1):
    # Spawn robot inside the playing field (clear of the white line)
    margin = robot_radius + 0.05
    px      = random.uniform(margin, width  - margin)
    py      = random.uniform(margin, length - margin)
    angle_f = random.uniform(0, 360)

    obstacles = []
    for _ in range(3):
        for _ in range(1000):
            ox = random.uniform(margin, width  - margin)
            oy = random.uniform(margin, length - margin)
            if (math.hypot(ox - px, oy - py) >= robot_radius + proximity and
                    all(math.hypot(ox - ex, oy - ey) >= 2 * proximity
                        for ex, ey in obstacles)):
                obstacles.append((ox, oy))
                break

    wall_segments = _build_wall_segments(width, length)
    results = _cast_rays(px, py, angle_f, obstacles, wall_segments, step_size, proximity)
    return px, py, angle_f, width, length, obstacles, results
