# PCCF/core/pitch_control.py

import jax.numpy as jnp


class PitchControl:

    def __init__(
        self,
        event_data,
        tracking_data,
    ):
        """
        Initialize Pitch Control object.
        """

        self.event_data = event_data
        self.tracking_data = tracking_data

        self.passes = None
        self.results = {}


    def segment_passes(self):
        """
        Segment passes.
        """

        from PCCF.utils.passes import (
            get_passes,
        )

        self.passes = get_passes(
            self
        )


    def get_pass_state(
        self,
        pass_id,
    ):
        """
        Extract state from pass.
        """

        from PCCF.utils.state import (
            get_pv_players,
        )


        if self.passes is None:
            self.segment_passes()


        passage = self.passes.iloc[
            pass_id
        ]


        return get_pv_players(
            passage,
            self.tracking_data,
            self.event_data,
        )


    def compute(
        self,
        pass_id,
        r_max=40,
        step=1,
        physics_kwargs=None,
    ):
        """
        Compute pitch control field.
        """

        from PCCF.model.solver_pitch_control import (
            integrate_pitch_control,
        )

        from PCCF.model.physics import (
            time_flight,
        )

        from PCCF.utils.grid import (
            generate_circular_grid,
        )

        if physics_kwargs is None:

            physics_kwargs = {
              
                "tau_r": 0.5,
                "alpha_rot": 1.0,
                "v_max": 5.0,

                "v0_candidates": jnp.linspace(
                    10.0,
                    30.0,
                    20,
                ),

                "theta_candidates": jnp.linspace(
                    0.0,
                    jnp.deg2rad(60),
                    20,
                ),

                "t_max_flight": 10.0,
                "dt_ball": 0.04,

                "z_max": 2.0,

                "rho": 1.2,
                "Cd": 0.25,
                "A": 0.038,
                "m": 0.43,
                "g": 9.81,

                "t_max_pc": 12.0,
                "dt_pc": 0.04,

                "lambda_control": 4.3,
                "s": 0.45,
                "tol": 1e-6,
            }


        shared_keys = [
            "tau_r",
            "alpha_rot",
            "v_max",
        ]


        ball_keys = [
            "v0_candidates",
            "theta_candidates",
            "t_max_flight",
            "dt_ball",
            "z_max",
            "rho",
            "Cd",
            "A",
            "m",
            "g",
        ]


        pc_keys = [
            "t_max_pc",
            "dt_pc",
            "lambda_control",
            "s",
            "tol",
        ]


        ball_kwargs = {
            key: physics_kwargs[key]
            for key in shared_keys + ball_keys
        }
        
        ball_kwargs["grid_step"] = step
        
        pc_kwargs = {
            key: physics_kwargs[key]
            for key in shared_keys + pc_keys
        }

        state = self.get_pass_state(
            pass_id
        )


        x_bc = state["x_bc"]
        v_bc = state["v_bc"]

        r_atts = state["r_atts"]
        v_atts = state["v_atts"]

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


        T_ball = time_flight(
            x_bc,
            r_target,
            r_atts,
            v_atts,
            **ball_kwargs,
        )


        p_pos = jnp.vstack(
            [
                r_atts,
                x_bc[None, :],
                r_defs,
            ]
        )


        p_vel = jnp.vstack(
            [
                v_atts,
                v_bc[None, :],
                v_defs,
            ]
        )


        n_atts = r_atts.shape[0]


        attacker_idx = jnp.arange(
            n_atts + 1
        )


        ball_carrier_idx = n_atts


        PC = integrate_pitch_control(
            r_grid=r_target,
            p_pos=p_pos,
            p_vel=p_vel,
            T_ball=T_ball,
            attacker_idx=attacker_idx,
            ball_carrier_idx=ball_carrier_idx,
            **pc_kwargs,
        )


        self.results[pass_id] = {

            "Pitch control field": PC,

            "grid": r_target,

            "params": {

                "ball_kwargs": ball_kwargs,

                "pc_kwargs": pc_kwargs,

                "r_max": r_max,

                "step": step,
            },
        }
