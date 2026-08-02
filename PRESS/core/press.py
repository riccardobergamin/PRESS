import jax.numpy as jnp


class PRESS:
    """
    Physics-informed framework for football pressure modelling.

    Components
    ----------
    BCOS:
        Ball Carrier Open Space

    PPCF:
        Physics-based Potential Pitch Control Field

    PI:
        Pressing Intensity
    """


    def __init__(
        self,
        event_data,
        tracking_data,
    ):

        self.event_data = event_data
        self.tracking_data = tracking_data

        self.results = {
            "BCOS": {},
            "PPCF": {},
            "PI": {},
        }



    def get_state(
        self,
        frame_id,
    ):
        """
        Extract physical state from frame.
        """

        from PRESS.utils.state import (
            get_pv_players,
        )


        return get_pv_players(
            frame_id,
            self.tracking_data,
            self.event_data,
        )



    # ============================================================
    # BALL CARRIER OPEN SPACE
    # ============================================================

    def bcos(
        self,
        frame_id,
        r_max=30,
        step=0.5,
        physics_args=None,
    ):
        """
        Compute Ball Carrier Open Space.
        """

        from PRESS.model.solver_bcos import (
            solve_grid,
        )

        from PRESS.utils.grid import (
            generate_circular_grid,
        )


        if physics_args is None:

            physics_args = {

                "MAX_POINTS": int(r_max / 0.5) + 10,

                "dx": 0.5,

                "s": 0.45,

                "tau_r_bc": 0.2,

                "delta_r_def": 0.5,

                "v_bc_max": 7.5,

                "v_def_max": 8.5,
                
                "alpha": 1.0,

                "dt": 0.04,

                "T_max": 15.0,

                "tol": 1e-7,
            }


        state = self.get_state(
            frame_id
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


        self.results["BCOS"][frame_id] = {

            "Ball carrier open space": S,

            "Defensive control": P,

            "grid": r_target,

            "params": {

                "physics_args": physics_args,

                "r_max": r_max,

                "step": step,
            },
        }



    # ============================================================
    # PITCH CONTROL FIELD
    # ============================================================

    def ppcf(
        self,
        frame_id,
        r_max=40,
        sigma = 14.0,
        k0 = 0.1,
        kv = 0.8,
        kd = 0.08,
        step=1,
        physics_kwargs=None,
    ):
        """
        Compute Physics-based Potential Pitch Control Field.

        Uses ballistic ball flight estimation.
        """


        from PRESS.model.solver_pitch_control import (
            integrate_pitch_control,
            relevant_ppcf,
        )

        from PRESS.physics.flight_ball import (
            time_flight,
        )

        from PRESS.utils.grid import (
            generate_circular_grid,
        )



        if physics_kwargs is None:

            physics_kwargs = {


                # shared response model

                "tau_r": 0.5,

                "alpha": 1.0,

                "v_max": 8.5,


                # ballistic ball model

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


                # pitch control

                "t_max_pc": 12.0,

                "dt_pc": 0.04,

                "lambda_control": 4.3,

                "s": 0.45,

                "tol": 1e-6,
            }



        shared_keys = [

            "tau_r",

            "alpha",

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



        state = self.get_state(
            frame_id
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



        # -------------------------------------
        # Ball flight physics
        # -------------------------------------

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



        # -------------------------------------
        # Relevant PPCF
        # -------------------------------------

        R_PPCF = relevant_ppcf(
            r_target,
            x_bc,
            v_bc,
            PC,
            sigma = sigma,
            k0 = k0,
            kv = kv,
            kd = kd,
        )



        self.results["PPCF"][frame_id] = {


            "Pitch control field": PC,


            "Relevant pitch control field": R_PPCF,


            "grid": r_target,


            "ball flight time": T_ball,


            "params": {

                "ball_kwargs": ball_kwargs,

                "pc_kwargs": pc_kwargs,

                "r_max": r_max,
                
                "sigma" : sigma,
                
                "k0" : k0,
                
                "kv" : kv,
                
                "kd" : kd,

                "step": step,
            },
        }


    # ============================================================
    # PRESSING INTENSITY
    # ============================================================

    def pressing_intensity(
        self,
        frame_id,
        t=1.5,
        s=0.45,
        tau_r_def=0.5,
        v_def_max=8.5,
        alpha=1.0,
        v_thr=2.0,
    ):
        """
        Compute Pressing Intensity.
        """

        from PRESS.model.solver_intensity import (
            interception_probability,
        )

        state = self.get_state(
            frame_id
        )

        p_def = interception_probability(
            state["x_bc"],
            state["v_bc"],
            state["r_defs"],
            state["v_defs"],
            t,
            s,
            tau_r_def,
            v_def_max,
            alpha,
        )

        # Ignore defenders moving slower than v_thr
        v_norm = jnp.linalg.norm(
            state["v_defs"],
            axis=1,
        )

        p_def = jnp.where(
            v_norm >= v_thr,
            p_def,
            0.0,
        )

        p_press = (
            1.0
            - jnp.prod(
                1.0 - p_def
            )
        )

        self.results["PI"][frame_id] = {
            "Pressing intensity": p_press,
            "Individual interception probabilities": p_def,
            "params": {
                "t": t,
                "s": s,
                "tau_r_def": tau_r_def,
                "v_def_max": v_def_max,
                "alpha": alpha,
                "v_thr": v_thr,
            },
        }
