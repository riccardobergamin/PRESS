"""
Spearman Pitch control dynamics
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from PRESS.physics.arrival_probabilities import F
from PRESS.physics.angles import compute_angle


def ppcf_step(
    P,
    t,
    p_pos,
    p_vel,
    r_grid,
    T_ball,
    lambda_control,
    s,
    tau_r,
    v_max,
    alpha,
    dt,
):
    """
    Single Euler integration step.

    Parameters
    ----------
    P : array, shape (n_players, n_grid)
        Current player control probabilities.

    Returns
    -------
    array, shape (n_players, n_grid)
        Updated player control probabilities.
    """

    # Player arrival probability
    F_int = F(
        p_pos,
        p_vel,
        r_grid,
        t,
        s,
        tau_r,
        v_max,
        alpha,
    )

    # Probability that each grid cell is still uncontrolled
    U = jnp.clip(
        1.0
        - jnp.sum(
            P,
            axis=0,
            keepdims=True,
        ),
        0.0,
        1.0,
    )

    # Ball availability at each grid cell
    ball_available = jnp.where(
        t >= T_ball[None, :],
        1.0,
        0.0,
    )

    # Euler increment
    dP = (
        lambda_control
        * U
        * F_int
        * ball_available
    )

    return jnp.clip(
        P + dt * dP,
        0.0,
        1.0,
    )



def integrate_pitch_control(
    r_grid,
    p_pos,
    p_vel,
    ball_carrier_idx,
    T_ball,
    lambda_control=4.3,
    s=0.45,
    tau_r=0.5,
    v_max=5.0,
    alpha=1.0,
    t_max_pc=8.0,
    dt_pc=0.01,
    tol=1e-6,
):
    """
    Integrate pitch control dynamics.

    Returns
    -------
    array, shape (n_grid,)
        Attacking pitch control probability.
    """

    n_players = p_pos.shape[0]
    n_grid = r_grid.shape[0]

    # Initial control probabilities
    P0 = jnp.zeros((n_players, n_grid))

    # Time discretization
    time_steps = jnp.arange(
        0.0,
        t_max_pc,
        dt_pc,
    )

    def scan_body(P, t):

        # Cells that are not yet fully controlled
        active = (
            jnp.sum(
                P,
                axis=0,
                keepdims=True,
            )
            < 1.0 - tol
        )

        P_next = ppcf_step(
            P,
            t,
            p_pos,
            p_vel,
            r_grid,
            T_ball,
            lambda_control,
            s,
            tau_r,
            v_max,
            alpha,
            dt_pc,
        )
        
        P_next = P_next.at[ball_carrier_idx].set(0.0)

        P = jnp.where(
            active,
            P_next,
            P,
        )

        return P, None
      

    P_final, _ = jax.lax.scan(
        scan_body,
        P0,
        time_steps,
    )
    
    return P_final


def relevant_ppcf(
    r_target,
    x_bc,
    v_bc,
    P_attackers,
    kappa_surface,
    sigma=14.0,
):
    """
    Compute transition probabilities and
    transition-weighted pitch control.

    kappa is obtained from a data-derived
    kappa(v, distance) surface.
    """

    from PRESS.utils.grid import lookup_kappa

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------

    delta = r_target - x_bc

    distance = (
        jnp.linalg.norm(
            delta,
            axis=-1,
        )
        + 1e-6
    )

    speed = jnp.linalg.norm(
        v_bc,
    )

    theta = compute_angle(
        x_bc[None, :],
        v_bc[None, :],
        r_target,
    )[0]

    # ---------------------------------------------------------
    # Data-derived Von Mises concentration
    # ---------------------------------------------------------

    kappa = lookup_kappa(
        speed=speed,
        distance=distance,
        kappa_surface=kappa_surface,
    )
    
    
    # ---------------------------------------------------------
    # Gaussian spatial term
    # ---------------------------------------------------------

    log_spatial = (
        -jnp.log(sigma)
        -0.5 * jnp.log(2.0 * jnp.pi)
        -0.5
        * (
            distance / sigma
        ) ** 2
    )

    # ---------------------------------------------------------
    # Log Von Mises angular term
    # ---------------------------------------------------------

    log_i0 = (
        jnp.log(
            jsp.special.i0e(kappa)
        )
        + kappa
    )

    log_angular = (
        kappa * jnp.cos(theta)
        - log_i0
    )

    # ---------------------------------------------------------
    # Transition probabilities
    # ---------------------------------------------------------

    log_weights = (
        log_spatial
        + log_angular
    )

    transition_probabilities = jnp.exp(
        log_weights
        - jsp.special.logsumexp(log_weights)
    )

    # ---------------------------------------------------------
    # Relevant pitch control
    # ---------------------------------------------------------

    return (
        transition_probabilities
        * P_attackers
    )
