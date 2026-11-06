"""Optional Ball Carrier Open Space model.

BCOS is kept separate from the main :class:`PRESS` class. Its scalar
summary remains an input feature of the existing action-selection booster.
"""
from __future__ import annotations

import jax.numpy as jnp


class BCOS:
    """Compute and store Ball Carrier Open Space fields for tracking frames."""

    def __init__(self, tracking_data):
        self.tracking_data = tracking_data
        self.results: dict[int, dict] = {}

    def compute_bcos(self, frame_id, r_max=30, step=0.5, physics_args=None):
        """Compute BCOS for one frame and return its result mapping."""
        from PRESS.model.solver_bcos import solve_grid
        from PRESS.utils.grid import generate_circular_grid
        from PRESS.utils.state import get_pv_players

        if physics_args is None:
            physics_args = {
                "MAX_POINTS": int(r_max / 0.5) + 30,
                "dx": 1.0,
                "s": 0.45,
                "v_bc_max": 6.5,
                "v_def_max": 7.0,
                "a_def_max": 7.0,
                "a_bc_max": 7.0,
                "dt": 0.04,
                "T_max": 15.0,
                "tol": 1e-7,
            }

        state = get_pv_players(frame_id, self.tracking_data)
        x_bc = state["x_bc"]
        v_bc = state["v_bc"]
        r_defs = state["r_defs"]
        v_defs = state["v_defs"]

        grid = generate_circular_grid(r_max, step)
        grid_world = x_bc + grid
        bounds = jnp.array([[-state["pitch_length"] / 2, state["pitch_length"] / 2],
                            [-state["pitch_width"] / 2, state["pitch_width"] / 2]])
        mask = jnp.all(
            (grid_world >= bounds[:, 0]) & (grid_world <= bounds[:, 1]),
            axis=1,
        )
        r_target = grid_world[mask]

        open_space, defensive_control = solve_grid(
            r_target,
            x_bc,
            v_bc,
            r_defs,
            v_defs,
            **physics_args,
        )

        result = {
            "Ball carrier open space": open_space,
            "Defensive control": defensive_control,
            "grid": r_target,
            "params": {"physics_args": physics_args, "r_max": r_max, "step": step},
        }
        self.results[frame_id] = result
        return result

    def get_result(self, frame_id):
        """Return BCOS output stored for ``frame_id``."""
        if frame_id not in self.results:
            raise KeyError(f"No BCOS result stored for frame {frame_id!r}.")
        return self.results[frame_id]
