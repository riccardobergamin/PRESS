# BCOS/core/bcos.py
import jax.numpy as jnp


class BCOS:
    def __init__(
        self,
        event_data,
        tracking_data,
    ):
        """
        Initialize BCOS model.
        """
        self.event_data = event_data
        self.tracking_data = tracking_data
        self.possessions = None
        self.results = {}

    def segment_possessions(self):
        """
        Segment ball possessions.
        """
        from BCOS.utils.possessions import (
            get_possessions,
        )
        self.possessions = get_possessions(
            self
        )

    def get_possession_state(
        self,
        possession_id,
    ):
        """
        Extract BC state from possession.
        """
        from BCOS.utils.state import (
            get_pv_players,
        )
        if self.possessions is None:
            self.segment_possessions()
        possession = self.possessions.iloc[
            possession_id
        ]
        return get_pv_players(
            possession,
            self.tracking_data,
            self.event_data,
        )

    def compute(
        self,
        possession_id,
        r_max=30,
        step=0.5,
        physics_args=None,
    ):
        """
        Compute BCOS field.

        Returns:
            S:
                Ball carrier arrival probability
            P:
                Defender first interception probabilities
                shape:
                    (N_target, N_def)
        """
        from BCOS.model.solver_bcos import (
            solve_grid,
        )
        from BCOS.utils.grid import (
            generate_circular_grid,
        )

        if physics_args is None:
            physics_args = {
                "MAX_POINTS": int(r_max / 0.5) + 10, 
                "dx": 0.5,
                "s": 0.45,
                "tau_r_bc": 0.2,
                "tau_r_def": 0.7,
                "alpha_rot": 1.0,
                "v_bc_max": 7.5,
                "v_def_max": 8.5,
                "dt": 0.04,
                "T_max": 15.0,
                "tol": 1e-7,
            }

        state = self.get_possession_state(
            possession_id
        )
        x_bc = state["x_bc"]
        v_bc = state["v_bc"]
        r_defs = state["r_defs"]
        v_defs = state["v_defs"]

        grid = generate_circular_grid(
            r_max,
            step,
        )
        grid_world = x_bc + grid

        bounds = jnp.array(
            [
                [-52.5, 52.5],
                [-34.0, 34.0],
            ]
        )
        mask = jnp.all(
            (grid_world >= bounds[:, 0])
            &
            (grid_world <= bounds[:, 1]),
            axis=1,
        )
        r_target = grid_world[mask]

        S, P = solve_grid(
            r_target,
            x_bc,
            v_bc,
            r_defs,
            v_defs,
            **physics_args,
        )

        self.results[possession_id] = {
            "Ball carrier open space": S,
            "Defensive control": P,
            "grid": r_target,
            "params": {
                "physics_args": physics_args,
                "r_max": r_max,
                "step": step,
            },
        }

