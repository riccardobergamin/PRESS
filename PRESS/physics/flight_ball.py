"""
Ball flight physics and arrival matching utilities.
"""

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint

from .times import time_to_intercept


def ball_trajectory(
    v0,
    theta,
    t_span,
    z0=0.0,
    rho=1.2,
    Cd=0.25,
    A=0.038,
    m=0.43,
    g=9.81,
):
    """
    Integrate a single ball trajectory.

    State vector:
        y[0] : horizontal position
        y[1] : vertical position
        y[2] : horizontal velocity
        y[3] : vertical velocity

    Airborne motion includes gravity and aerodynamic drag. Below 5 cm while
    descending, height is frozen and horizontal drag continues. The latent vz
    state is not reset and is not a physical post-impact velocity. Only position
    and height are used by the pass benchmark. Bounce and rolling friction
    are not modelled; this retains the fitted forward model.
    """

    # Aerodynamic drag coefficient
    k = 0.5 * rho * Cd * A / m

    def ode(y, t):

        z = y[1]
        v_d = y[2]
        v_z = y[3]

        # Ball speed magnitude
        v_norm = jnp.sqrt(
            v_d**2 + v_z**2
        )

        def airborne(_):
            """
            Ball flight dynamics.

            Horizontal and vertical motion
            are affected by aerodynamic drag.
            """

            return jnp.array(
                [
                    v_d,
                    v_z,
                    -k * v_d * v_norm,
                    -k * v_z * v_norm - g,
                ]
            )

        def ground_contact(_):
            """
            Freeze height and vertical acceleration after ground contact.

            The latent vertical velocity is retained, not reset to zero.
            Horizontal motion continues with aerodynamic drag.
            """

            return jnp.array(
                [
                    v_d,
                    0.0,
                    -k * v_d * jnp.abs(v_d),
                    0.0,
                ]
            )

        # Ground contact occurs when the ball
        # reaches the ground while moving downward.

        on_ground = (z <= 0.05) & (v_z <= 0.0)

        return jax.lax.cond(
            on_ground,
            ground_contact,
            airborne,
            operand=None,
        )

    # Initial state
    # odeint requires matching state/time dtypes. Mixed Python scalars and
    # float32 sample times otherwise crash when global x64 is enabled.

    dtype = jnp.result_type(v0, theta, z0, t_span)
    y0 = jnp.array(
        [
            0.0,                        # initial horizontal position
            z0,                         # initial height
            v0 * jnp.cos(theta),         # horizontal velocity
            v0 * jnp.sin(theta),         # vertical velocity
        ], dtype=dtype,
    )
    t_span = jnp.asarray(t_span, dtype=dtype)

    return odeint(
        ode,
        y0,
        t_span,
    )


def generate_ball_trajectories(
    v0_candidates,
    theta_candidates,
    t_span,
    rho=1.2,
    Cd=0.25,
    A=0.038,
    m=0.43,
    g=9.81
):
    """
    Generate candidate ball trajectories.
    """

    vv, tt = jnp.meshgrid(
        v0_candidates,
        theta_candidates,
    )

    def solve(v0, theta):

        return ball_trajectory(
            v0,
            theta,
            t_span,
            z0=0,
            rho=rho,
            Cd=Cd,
            A=A,
            m=m,
            g=g
        )

    return jax.vmap(
        solve
    )(
        vv.ravel(),
        tt.ravel(),
    )


def build_flight_lookup(
    r_limit,
    trajectories,
    t_span,
    grid_step,
    z_max,
):
    """
    Build lookup table:
    radius -> possible ball arrival times.
    """

    dist_phys = trajectories[:, :, 0]
    z_phys = trajectories[:, :, 1]

    n_radii = (
        int(r_limit / grid_step)
        + 2
    )

    # Indexing uses round(distance/grid_step), requiring that exact spacing.
    radii = jnp.arange(n_radii) * grid_step

    def interpolate(radius):

        def single(
            distance,
            height,
        ):

            t_hit = jnp.interp(
                radius,
                distance,
                t_span,
            )

            z_hit = jnp.interp(
                t_hit,
                t_span,
                height,
            )

            valid = (
                (z_hit <= z_max)
                &
                (radius <= jnp.max(distance))
            )

            return jnp.where(
                valid,
                t_hit,
                jnp.inf,
            )

        return jax.vmap(
            single
        )(
            dist_phys,
            z_phys,
        )

    lookup = jax.vmap(
        interpolate
    )(
        radii
    )

    return radii, lookup

# def player_arrival_time(
#     r_target,
#     p_attackers,
#     v_attackers,
#     tau_r,
#     alpha,
#     v_max,
# ):
#     """
#     Minimum player arrival time.
#
#     Computes the fastest player arrival
#     for each target position.
#     """
#
#     # position after reaction
#     p_react =  p_attackers[:, None, :] + v_attackers[:, None, :] * tau_r
#
#
#     # player-target vectors
#     delta = r_target[None, :, :] - p_react
#
#     # distances
#     distance = jnp.linalg.norm(delta, axis=-1)
#
#
#     tau_move = distance / v_max
#
#
#     # rotation time
#     tau_theta = rotation_time(p_attackers, v_attackers, r_target, alpha)
#
#    # running time
#
#     tau_players =  tau_r + tau_theta + tau_move
#
#
#     # fastest player per target
#     return jnp.min(tau_players,axis=0)


def player_arrival_time(
    r_target,
    p_attackers,
    v_attackers,
    v_max,
    a_max,
    *,
    a_turn=None,
    preserve_initial_overspeed=True,
):
    """
    Arrival time of the player closest to each target position.

    The receiving player is selected based on current position
    (p_attackers), not on who can reach the target fastest.

    After selecting the closest player, its arrival time is
    computed using the same time-to-intercept model used by F().
    """

    # --------------------------------------------------------
    # Player-target vectors
    # --------------------------------------------------------

    if p_attackers.shape[0] == 0:
        return jnp.full(r_target.shape[0], jnp.inf)

    delta = (
        r_target[None, :, :]
        - p_attackers[:, None, :]
    )

    # --------------------------------------------------------
    # Current distance from each player to each target
    # --------------------------------------------------------

    distance = jnp.linalg.norm(
        delta,
        axis=-1,
    )
    valid = jnp.isfinite(p_attackers).all(axis=-1) & (jnp.linalg.norm(p_attackers,axis=-1)<500.)
    distance = jnp.where(valid[:,None],distance,jnp.inf)

    # --------------------------------------------------------
    # Player closest to each target
    # --------------------------------------------------------

    player_idx = jnp.argmin(
        distance,
        axis=0,
    )

    # --------------------------------------------------------
    # Time to intercept for every player-target pair
    # --------------------------------------------------------

    tau_players = time_to_intercept(
        p_attackers,
        v_attackers,
        r_target,
        v_max=v_max,
        a_max=a_max,
        a_turn=a_turn,
        preserve_initial_overspeed=preserve_initial_overspeed,
    )

    # --------------------------------------------------------
    # Keep only the arrival time of the
    # player closest to each target
    # --------------------------------------------------------

    tau_selected = jnp.take_along_axis(
        tau_players,
        player_idx[None, :],
        axis=0,
    )[0]

    return jnp.where(jnp.any(valid),tau_selected,jnp.inf)


def exact_ball_arrival_times(distances, player_times, trajectories, t_span, z_max=2.):
    """Select candidate times at the exact radius, without a radial lookup.

    Candidates are interpolated in monotone horizontal distance. Unreachable
    or too-high candidates have infinite time. The flight closest to the
    nearest receiver's arrival is selected, as in the final benchmark.
    """
    def select(radius, wanted):
        times=jax.vmap(lambda tr:jnp.interp(radius,tr[:,0],t_span))(trajectories)
        heights=jax.vmap(lambda tr,t:jnp.interp(t,t_span,tr[:,1]))(trajectories,times)
        available=(heights<=z_max) & (radius<=trajectories[:,:,0].max(axis=1))
        times=jnp.where(available,times,jnp.inf)
        return jnp.where(jnp.isfinite(wanted),times[jnp.argmin(jnp.abs(times-wanted))],jnp.inf)

    return jax.vmap(select)(distances,player_times)


def match_ball_player_time(
    distances,
    player_times,
    lookup,
    grid_step,
):
    """
    Select ball trajectory time closest
    to player arrival time.
    """

    n_radii = lookup.shape[0]

    def select(
        distance,
        player_time,
    ):

        idx = jnp.int32(
            jnp.clip(
                jnp.rint(
                    distance / grid_step
                ),
                0,
                n_radii - 1,
            )
        )

        candidate_times = lookup[idx]

        best_idx = jnp.argmin(
            jnp.abs(
                candidate_times
                -
                player_time
            )
        )

        return candidate_times[best_idx]

    return jax.vmap(
        select
    )(
        distances,
        player_times,
    )


def time_flight(
    r_bc,
    r_target,
    p_attackers,
    v_attackers,
    v0_candidates=jnp.linspace(
        10.0,
        30.0,
        20,
    ),
    theta_candidates=jnp.linspace(
        0.0,
        jnp.deg2rad(60),
        20,
    ),
    t_max_flight=10.0,
    dt_ball=0.04,
    grid_step=0.5,
    v_max=5.0,
    a_max=5.0,
    z_max=2.0,
    rho=1.2,
    Cd=0.25,
    A=0.038,
    m=0.43,
    g=9.81,
    distance_mode="exact",
    a_turn=None,
    preserve_initial_overspeed=True,
):
    """
    Compute final ball arrival time.

    Steps:
    1. Generate physical ball trajectories.
    2. Compute arrival of the player currently nearest each target.
    3. Evaluate candidate times at the exact target distance.
    4. Match ball trajectory with player arrival.

    distance_mode='lookup' is an explicitly approximate, correctly spaced
    table for sensitivities. The final benchmark uses 'exact'.
    """

    if r_target.shape[0] == 0:
        return jnp.empty((0,))

    t_span = jnp.arange(
        0.0,
        t_max_flight + dt_ball,
        dt_ball,
    )

    trajectories = generate_ball_trajectories(
        v0_candidates,
        theta_candidates,
        t_span,
        rho=rho,
        Cd=Cd,
        A=A,
        m=m,
        g=g,
    )

    distances = jnp.linalg.norm(r_target - r_bc, axis=-1)

    tau_players = player_arrival_time(
        r_target,
        p_attackers,
        v_attackers,
        v_max,
        a_max,
        a_turn=a_turn,
        preserve_initial_overspeed=preserve_initial_overspeed,
    )

    if distance_mode == "exact":
        return exact_ball_arrival_times(distances,tau_players,trajectories,t_span,z_max)
    if distance_mode != "lookup":
        raise ValueError("distance_mode must be 'exact' or 'lookup'.")
    _, lookup = build_flight_lookup(
        r_limit=jnp.max(distances),trajectories=trajectories,
        t_span=t_span,grid_step=grid_step,z_max=z_max,
    )

    return match_ball_player_time(
        distances,
        tau_players,
        lookup,
        grid_step,
    )
