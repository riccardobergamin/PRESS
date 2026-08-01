"""
Spearman Pitch control dynamics
"""

import jax
import jax.numpy as jnp
from PRESS.physics.arrival_probabilities import F


def ppcf_step(
    P,
    t,
    p_pos,
    p_vel,
    r_grid,
    T_ball,
    lambda_control,
    tau_r,
    s,
    v_max,
    alpha_rot,
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
        alpha_rot,
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
    T_ball,
    attacker_idx,
    ball_carrier_idx,
    t_max_pc=8.0,
    dt_pc=0.01,
    lambda_control=4.3,
    tau_r=0.5,
    s=0.45,
    v_max=5.0,
    alpha_rot=1.0,
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

    # Attacking players (excluding the ball carrier)
    attacker_mask = (
        jnp.zeros(n_players, dtype=bool)
        .at[attacker_idx].set(True)
        .at[ball_carrier_idx].set(False)
    )[:, None]

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
            tau_r,
            s,
            v_max,
            alpha_rot,
            dt_pc,
        )

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

    # Sum attacking player control
    return jnp.sum(
        jnp.where(
            attacker_mask,
            P_final,
            0.0,
        ),
        axis=0,
    )


# def relevant_ppcf(
#     r_target,
#     r_ball,
#     P_attackers,
#     sigma=14.0,
#     alpha=1.0,
# ):
#     """
#     Compute transition probabilities and
#     transition-weighted pitch control.
# 
#     Parameters
#     ----------
#     r_target : array, shape (n_grid, dim)
#         Candidate receiving positions.
# 
#     r_ball : array, shape (dim,)
#         Current ball position.
# 
#     P_attackers : array, shape (n_grid,)
#         Attacking pitch control map.
# 
#     sigma : float
#         Spatial pass dispersion.
# 
#     alpha : float
#         Pitch control weighting exponent.
# 
#     Returns
#     -------
#     transition_probabilities : array, shape (n_grid,)
#         Probability distribution over receiving positions.
# 
#     relevant_ppcf : array, shape (n_grid,)
#         Pitch control weighted by transition probability.
#         The sum represents the expected pitch control
#         over the pass distribution.
#     """
# 
#     # Pass spatial likelihood
#     delta = r_target - r_ball
# 
#     gaussian = jnp.exp(
#         -0.5
#         * jnp.sum(delta**2, axis=1)
#         / sigma**2
#     )
# 
#     # Transition model
#     weighted = (
#         gaussian
#         * (P_attackers + 1e-8) ** alpha
#     )
# 
#     transition_probabilities = (
#         weighted
#         /
#         jnp.sum(weighted)
#     )
# 
#     # Relevant PPCF
#     relevant_ppcf = (
#         transition_probabilities
#         *
#         P_attackers
#     )
# 
#     return (
#         transition_probabilities,
#         relevant_ppcf,
#     )
