"""
Spearman Pitch control dynamics
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from PRESS.physics.arrival_probabilities import F
from PRESS.physics.angles import compute_angle

def pass_prob_step(
    P,
    t,
    p_pos,
    p_vel,
    x_ball,
    lambda_control,
    s,
    v_max,
    a_max,
    dt,
    z_max,
):
    """
    Single Euler integration step.

    Parameters
    ----------
    x_ball : shape (3,)
        Current 3D ball position [x, y, z].

    The x/y components are used as the 2D target
    passed to F.

    Control probability is accumulated only when
    the ball is at or below z_max.
    """

    # --------------------------------------------------------
    # Ball position -> 2D target
    # --------------------------------------------------------

    r_target = x_ball[:2][None, :]

    # --------------------------------------------------------
    # Player arrival probability
    # --------------------------------------------------------

    F_int = F(
        p_pos,
        p_vel,
        r_target,
        t,
        s,
        v_max,
        a_max,
    ).squeeze(-1)

    # --------------------------------------------------------
    # Probability that the ball position
    # is still uncontrolled
    # --------------------------------------------------------

    U = jnp.clip(
        1.0 - jnp.sum(P),
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Euler increment
    #
    # No control while the ball is above z_max.
    # --------------------------------------------------------

    dP = jnp.where(
        x_ball[2] <= z_max,
        lambda_control * U * F_int,
        0.0,
    )

    return jnp.clip(
        P + dt * dP,
        0.0,
        1.0,
    )


def integrate_pass_prob(
    p_pos,
    p_vel,
    attacker_idx,
    ball_carrier_idx,
    ball_trajectory,
    lambda_control=4.3,
    s=0.45,
    v_max=5.0,
    a_max=5.0,
    dt=0.04,
    z_max=2.0,
    tol=1e-6,
):
    """
    Integrate pass probability along the full 3D
    ball trajectory.

    The ball trajectory determines the number of
    integration steps. The same dt is used both for
    the trajectory time axis and for Euler integration.

    Parameters
    ----------
    ball_trajectory : array, shape (T, 3)
        Full 3D ball trajectory.

    dt : float
        Time step of ball_trajectory in seconds.

    Returns
    -------
    pass_prob : scalar
        Total attacking control probability.
    """

    n_players = p_pos.shape[0]

    # --------------------------------------------------------
    # INITIAL CONTROL PROBABILITIES
    # --------------------------------------------------------

    P0 = jnp.zeros(
        (n_players,)
    )

    # --------------------------------------------------------
    # TIME DISCRETIZATION
    #
    # Number of steps is determined directly by the
    # ball trajectory.
    # --------------------------------------------------------
    
    n_steps = ball_trajectory.shape[0]

    time_steps = (
        jnp.arange(n_steps)
        * dt
    )

    # --------------------------------------------------------
    # SCAN INTEGRATION
    # --------------------------------------------------------

    def scan_body(
        P,
        inputs,
    ):

        t, x_ball = inputs

        P_next = pass_prob_step(
            P=P,
            t=t,
            p_pos=p_pos,
            p_vel=p_vel,
            x_ball=x_ball,
            lambda_control=lambda_control,
            s=s,
            v_max=v_max,
            a_max=a_max,
            dt=dt,
            z_max=z_max,
        )

        # ----------------------------------------------------
        # BALL CARRIER DOES NOT PARTICIPATE IN CONTROL
        # ----------------------------------------------------

        P_next = P_next.at[
            ball_carrier_idx
        ].set(0.0)

        # ----------------------------------------------------
        # STOP UPDATING ONCE TOTAL CONTROL REACHES ~1
        # ----------------------------------------------------

        active = (
            jnp.sum(P)
            < 1.0 - tol
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
        (
            time_steps,
            ball_trajectory,
        ),
    )

    # --------------------------------------------------------
    # TOTAL ATTACKING CONTROL PROBABILITY
    # --------------------------------------------------------

    pass_prob = jnp.sum(
        P_final[attacker_idx]
    )

    return pass_prob
  
  
def ppcf_step(
    P,
    t,
    p_pos,
    p_vel,
    r_grid,
    T_ball,
    lambda_control,
    s,
    v_max,
    a_max,
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
        v_max,
        a_max,
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
    v_max=5.0,
    a_max=5.0,
    t_max_pc=8.0,
    dt_pc=0.01,
    tol=1e-6,
):
    """
    Integrate pitch control dynamics.

    Returns
    -------
    array, shape (n_players, n_grid)
        Player-specific control probabilities. Sum the attacking rows
        to obtain attacking pitch control, as PRESS.compute_ppcf does.
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
            v_max,
            a_max,
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
  
  
def Platt_scaling(
    P_attackers,
    a=1.96,
    b=0.42,
):
  
    P_attackers = jnp.clip(
        P_attackers,
        1e-6,
        1 - 1e-6,
    )

    return jax.nn.sigmoid(
        a * (
            jnp.log(P_attackers)
            - jnp.log1p(-P_attackers)
        )
        + b
    )

