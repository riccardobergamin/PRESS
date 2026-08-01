"""
Numerical solvers for competing risks model.
"""

import jax
import jax.numpy as jnp
from functools import partial

from PRESS.physics.bcos_path import build_path
from PRESS.physics.hazard_rates_bcos import (
    ballcarrier_hazard,
    defender_hazard,
)

# ============================================================
# Euler integration step
# ============================================================

@jax.jit
def solver_step(
    state,
    x_bc0,
    v_bc0,
    r_def,
    v_def,
    x_path,
    mask,
    s,
    tau_r_bc,
    tau_r_def,
    v_bc_max,
    v_def_max,
    alpha_rot,
    dt,
    tol=1e-6,
):
    """
    Perform one Euler step.

    State:
        S       : ball carrier event probability
        P_def   : defender interception probabilities (N_def,)
        t       : current time
    """

    S, P_def, t = state

    is_valid = mask > 0.0

    idx = jnp.argmax(
        jnp.diff(
            is_valid.astype(jnp.float32),
            append=0.0,
        ) < 0
    )

    x_target = x_path[idx]

    h_BC = ballcarrier_hazard(
        x_bc0,
        v_bc0,
        x_target,
        t,
        s,
        tau_r_bc,
        v_bc_max,
        alpha_rot,
    )

    h_def = defender_hazard(
        x_bc0,
        v_bc0,
        r_def,
        v_def,
        x_path,
        mask,
        t,
        s,
        tau_r_def,
        tau_r_bc,
        v_def_max,
        v_bc_max,
        alpha_rot,
    )

    U = jnp.clip(
        1.0 - S - jnp.sum(P_def),
        0.0,
        1.0,
    )

    dS = U * h_BC
    dP_def = U * h_def

    S_new = S + dt * dS
    P_def_new = P_def + dt * dP_def

    active = (
        S + jnp.sum(P_def)
    ) < (1.0 - tol)

    S_new = jnp.where(
        active,
        jnp.clip(S_new, 0.0, 1.0),
        S,
    )

    P_def_new = jnp.where(
        active,
        jnp.clip(P_def_new, 0.0, 1.0),
        P_def,
    )

    new_state = (
        S_new,
        P_def_new,
        t + dt,
    )

    return new_state, None


# ============================================================
# Integrate one target point
# ============================================================

@partial(
    jax.jit,
    static_argnums=(5, 13, 14),
)
def integrate_point(
    x_target,
    x_bc0,
    v_bc0,
    r_def,
    v_def,
    MAX_POINTS=500,
    dx=0.1,
    s=0.45,
    tau_r_bc=0.2,
    tau_r_def=0.7,
    alpha_rot=1.0,
    v_bc_max=7.5,
    v_def_max=8.5,
    dt=0.05,
    T_max=15.0,
    tol=1e-6,
):
    """
    Integrate competing risks for one target point.

    Returns:
        trajectory[0]: S(t)
        trajectory[1]: P_def(t)
        trajectory[2]: t
    """

    x_path, mask = build_path(
        x_target,
        x_bc0,
        MAX_POINTS=MAX_POINTS,
        dx=dx,
    )

    n_def = r_def.shape[0]

    state0 = (
        jnp.array(0.0),
        jnp.zeros(n_def),
        jnp.array(0.0),
    )

    num_steps = int(T_max / dt)

    def step(state, _):

        next_state, _ = solver_step(
            state,
            x_bc0,
            v_bc0,
            r_def,
            v_def,
            x_path,
            mask,
            s,
            tau_r_bc,
            tau_r_def,
            v_bc_max,
            v_def_max,
            alpha_rot,
            dt,
            tol,
        )

        return next_state, next_state

    _, trajectory = jax.lax.scan(
        step,
        state0,
        None,
        length=num_steps,
    )

    trajectory = (
        jnp.concatenate(
            [state0[0][None], trajectory[0]]
        ),
        jnp.concatenate(
            [state0[1][None, :], trajectory[1]],
            axis=0,
        ),
        jnp.concatenate(
            [state0[2][None], trajectory[2]]
        ),
    )

    return trajectory


# ============================================================
# Grid solver
# ============================================================

def solve_grid(
    x_grid,
    x_bc0,
    v_bc0,
    r_def,
    v_def,
    MAX_POINTS=500,
    dx=0.1,
    s=0.45,
    tau_r_bc=0.2,
    tau_r_def=0.7,
    alpha_rot=1.0,
    v_bc_max=7.5,
    v_def_max=8.5,
    dt=0.05,
    T_max=15.0,
    tol=1e-6,
):
    """
    Solve the model over a spatial grid.

    Only the final time step is returned.

    Returns
    -------
    S_final:
        (N_target,)

    P_final:
        (N_target, N_def)
    """

    def solver(target):

        trajectory = integrate_point(
            target,
            x_bc0,
            v_bc0,
            r_def,
            v_def,
            MAX_POINTS,
            dx,
            s,
            tau_r_bc,
            tau_r_def,
            alpha_rot,
            v_bc_max,
            v_def_max,
            dt,
            T_max,
            tol,
        )

        S_final = trajectory[0][-1]
        P_final = trajectory[1][-1]

        return S_final, P_final

    return jax.vmap(
        solver
    )(x_grid)
