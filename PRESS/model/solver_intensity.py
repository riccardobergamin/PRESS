import jax
import jax.numpy as jnp
from PRESS.physics.arrival_probabilities import F
from PRESS.physics.times import rotation_time


def loss_prob_step(
    P,
    t,
    p_pos,
    p_vel,
    x_bc0,
    v_bc,
    lambda_0,
    beta_v,
    beta_B,
    s=0.45,
    dt=0.04,
    v_max=5.0,
    a_max=7.0,
):

    bounds = jnp.array(
        [
            [-52.5, 52.5],
            [-34.0, 34.0],
        ]
    )

    # --------------------------------------------------------
    # Ball carrier position
    # --------------------------------------------------------

    x_bc = (
        x_bc0
        + v_bc * t
    )

    v_bc_norm = jnp.linalg.norm(
        v_bc
    )

    # --------------------------------------------------------
    # Survival probability
    # --------------------------------------------------------

    U = jnp.clip(
        1.0 - jnp.sum(P),
        0.0,
        1.0,
    )

    # ========================================================
    # REAL DEFENDERS
    # ========================================================

    F_def = F(
        p_pos,
        p_vel,
        x_bc[None, :],
        t,
        s,
        v_max,
        a_max,
    ).squeeze(-1)


    # --------------------------------------------------------
    # Boundary distance
    #
    # Minimum distance from BC to any pitch boundary.
    # --------------------------------------------------------

    x_B = jnp.min(
        jnp.abs(
            jnp.array(
                [
                    x_bc[0] - bounds[0, 0],
                    bounds[0, 1] - x_bc[0],
                    x_bc[1] - bounds[1, 0],
                    bounds[1, 1] - x_bc[1],
                ]
            )
        )
    )
    

    # --------------------------------------------------------
    # Proportional hazard
    # --------------------------------------------------------

    lambda_def = (
        lambda_0
        * jnp.exp(
            beta_v * v_bc_norm
            + beta_B * x_B,
        )
    )

    # --------------------------------------------------------
    # Defensive loss probability increment
    # --------------------------------------------------------

    dP = (
        U
        * F_def
        * lambda_def
    )

    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    return jnp.clip(
        P + dt * dP,
        0.0,
        1.0,
    )


def integrate_loss_prob(
    p_pos,
    p_vel,
    x_bc0,
    v_bc0,
    delta_t,
    lambda_0,
    beta_v,
    beta_B,
    s=0.45,
    v_max=5.0,
    a_max=7.0,
    dt=0.04,
    tol=1e-6,
):

    n_steps = int(
        jnp.ceil(
            delta_t / dt
        )
    )

    time_steps = (
        jnp.arange(n_steps)
        * dt
    )

    P0 = jnp.zeros(
        p_pos.shape[0],
    )

    def scan_body(P, t):

        P_next = loss_prob_step(
            P=P,
            t=t,
            p_pos=p_pos,
            p_vel=p_vel,
            x_bc0=x_bc0,
            v_bc=v_bc0,
            lambda_0=lambda_0,
            beta_v=beta_v,
            beta_B=beta_B,
            s=s,
            dt=dt,
            v_max=v_max,
            a_max=a_max,
        )

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
        time_steps,
    )

    return P_final
