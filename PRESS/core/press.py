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
        tracking_data,
        events_data = None
    ):

        from PRESS.utils.sanitizers import (
            validate_tracking_data,
            validate_events_data
        )

        validate_tracking_data(tracking_data)

        self.tracking_data = tracking_data

        if events_data is not None:

          validate_events_data(events_data)


        self.events_data = events_data


        self.results = {
            "BCOS": {},
            "PPCF": {},
            "PI": {},
            # stores Action Selection output (P(pass)) per frame_id,
            # same one-dict-per-component pattern as BCOS/PPCF/PI.
            "ACTION_SELECTION": {},
            # NEW: stores the instantaneous hazard h(s_t) per frame_id
            # (raw + isotonic-calibrated), same pattern.
            "HAZARD": {},
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
            self.tracking_data
        )


    def get_result(self, component, frame_id):
        """Return a stored component result for one tracking frame.

        ``component`` accepts the result keys ``BCOS``, ``PPCF``, ``PI``,
        ``ACTION_SELECTION``, and ``HAZARD``. The returned value is the
        stored result mapping; callers can inspect it without reaching
        into the internal ``results`` structure directly.
        """
        component = str(component).upper()
        if component not in self.results:
            raise ValueError(
                f"Unknown component {component!r}; expected one of "
                f"{', '.join(self.results)}."
            )
        if frame_id not in self.results[component]:
            raise KeyError(
                f"No {component} result stored for frame {frame_id!r}."
            )
        return self.results[component][frame_id]


    def get_bcos_result(self, frame_id):
        """Return BCOS output stored for ``frame_id``."""
        return self.get_result("BCOS", frame_id)


    def get_ppcf_result(self, frame_id):
        """Return pitch-control and pass output stored for ``frame_id``."""
        return self.get_result("PPCF", frame_id)


    def get_ball_drive_result(self, frame_id):
        """Return ball-drive risk output stored for ``frame_id``."""
        return self.get_result("PI", frame_id)


    def get_action_selection_result(self, frame_id):
        """Return action-selection output stored for ``frame_id``."""
        return self.get_result("ACTION_SELECTION", frame_id)


    def get_hazard_result(self, frame_id):
        """Return composed hazard output stored for ``frame_id``."""
        return self.get_result("HAZARD", frame_id)


    # ============================================================
    # BALL CARRIER OPEN SPACE
    # ============================================================

    def compute_bcos(
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

    def compute_ppcf(
        self,
        frame_id,
        r_max=150,
        step=1.0,
        a_platt=1.96,
        b_platt=0.42,
        physics_kwargs=None,
    ):
        """
        Compute Physics-based Potential Pitch Control Field.

        Uses ballistic ball flight estimation.
        """


        from PRESS.model.solver_pitch_control import (
            integrate_pitch_control, Platt_scaling
        )

        from PRESS.physics.flight_ball import (
            time_flight,
        )

        from PRESS.utils.grid import (
            generate_circular_grid
        )



        if physics_kwargs is None:

            physics_kwargs = {


                # shared response model

                "a_max": 7.0,

                "v_max": 5.0,


                # ballistic ball model

                "v0_candidates": jnp.linspace(
                    5.0,
                    25.0,
                    30,
                ),

                "theta_candidates": jnp.linspace(
                    0.0,
                    jnp.deg2rad(45),
                    30,
                ),


                "t_max_flight": 10.0,

                "dt_ball": 0.01,

                "z_max": 2.0,


                "rho": 1.2,

                "Cd": 0.25,

                "A": 0.038,

                "m": 0.43,

                "g": 9.81,


                # pitch control

                "t_max_pc": 15.0,

                "dt_pc": 0.04,

                "lambda_control": 4.3,

                "s": 0.43,

                "tol": 1e-6
            }



        shared_keys = [

            "a_max",

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

            "tol"
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

            ball_carrier_idx=ball_carrier_idx,

            **pc_kwargs,
        )


        # Sum attacking player control

        attacker_mask = jnp.zeros(
            PC.shape[0],
            dtype=bool,
        ).at[attacker_idx].set(
            True,
        )


        PC_att = jnp.sum(
            jnp.where(
                attacker_mask[:, None],
                PC,
                0.0,
            ),
            axis=0,
        )


        PC_scaled_att = Platt_scaling(
                            P_attackers = PC_att,
                            a = a_platt,
                            b = b_platt)


        self.results["PPCF"][frame_id] = {

                "Pitch control field": PC,

                "Attacking pitch control field": PC_att,

                "Pass probability": PC_scaled_att,

                "attacker_idx": attacker_idx,

                "ball_carrier_idx": ball_carrier_idx,

                "grid": r_target,

                "ball flight time": T_ball,

                "params": {

                    "ball_kwargs": ball_kwargs,

                    "pc_kwargs": pc_kwargs,

                    "r_max": r_max,

                    "step": step,
                },
        }


    def compute_transition_probabilities(
        self,
        frame_id,
        xT_grid,
    ):
        """
        Compute the destination CHOICE probability distribution
        (softmax over the trained XGBoost transition model) for the
        ball carrier at frame_id.

        Requires compute_ppcf() AND compute_ball_drive_loss() to have
        already been run for this frame_id.

        Stores in self.results["PPCF"][frame_id]["Transition probabilities"].
        """

        if frame_id not in self.results["PPCF"]:
            raise ValueError(
                "Compute PPCF before computing transition probabilities"
            )

        if (
            frame_id not in self.results["PI"]
            or "Ball drive loss probability"
            not in self.results["PI"][frame_id]
        ):
            raise ValueError(
                "Compute ball drive loss before computing transition "
                "probabilities (ball_drive_prob is one of the model "
                "features)."
            )

        from PRESS.utils.transition import (
            build_transition_features,
            predict_transition_probabilities,
        )

        state = self.get_state(frame_id)

        ppcf = self.results["PPCF"][frame_id]

        # CONFIRMED (25/9): the transition model's own original
        # feature-engineering script (extract_pass_destination_dataset /
        # compute_ball_drive) uses the LOSS probability directly,
        # unflipped, for the "ball_drive_prob" training column -- no
        # inversion here. (Different from the action_features/ pipeline
        # used for action selection, which does invert it -- same
        # column name, opposite convention across the two original
        # training scripts. Each wrapper matches its own model's real
        # training data.)
        X = build_transition_features(
            x_bc=state["x_bc"],
            v_bc=state["v_bc"],
            r_defs=state["r_defs"],
            r_atts=state["r_atts"],
            attacking_side=state["attacking_side"],
            targets=ppcf["grid"],
            target_pitch_control=ppcf["Attacking pitch control field"],
            ball_drive_prob=self.results["PI"][frame_id][
                "Ball drive loss probability"
            ],
            xT_grid=xT_grid,
        )

        transition_probabilities = predict_transition_probabilities(X)

        self.results["PPCF"][frame_id][
            "Transition probabilities"
        ] = transition_probabilities

        return transition_probabilities


    # ============================================================
    # PRESSING INTENSITY
    # ============================================================

    def compute_ball_drive_loss(
        self,
        frame_id,
        t_press=1.0,
        physics_kwargs=None,
    ):
        """
        Compute ball-drive loss probabilities.
        """

        from PRESS.model.solver_intensity import (
            integrate_loss_prob,
        )

        if physics_kwargs is None:

            physics_kwargs = {

                "v_max": 7.0,

                "a_max": 7.0,

                "lambda_0": 0.36,

                "beta_v": 0.09,

                "beta_B": -0.02,

                "s": 0.43,
            }

        state = self.get_state(
            frame_id,
        )

        p_loss = integrate_loss_prob(
            state["r_defs"],
            state["v_defs"],
            state["x_bc"],
            state["v_bc"],
            t_press,
            **physics_kwargs,
        )

        self.results["PI"][frame_id] = {

            "Individual tackle probabilities":
                p_loss,

            "Ball drive loss probability":
                jnp.sum(p_loss),

            "params": {
                "t_press": t_press,
                "physics_kwargs": physics_kwargs,
            },
        }


    # ============================================================
    # ACTION SELECTION
    # ============================================================

    def compute_action_selection_probability(
        self,
        frame_id,
        xT_grid,
        start_type,
        previous_action,
    ):
        """
        Compute P(pass) vs P(ball_drive) for the ball carrier at
        frame_id (trained XGBoost action selection model, single
        sigmoid over the instantaneous state -- not a per-destination
        choice set like compute_transition_probabilities).

        Requires compute_bcos(), compute_ppcf(), compute_ball_drive_loss()
        AND compute_transition_probabilities() to have already been run
        for this frame_id, in that order (same dependency chain as
        compute_transition_probabilities, plus BCOS which that method
        does not need).

        start_type / previous_action are the raw category strings for
        this action (e.g. "reception_open_play", "ball_drive") -- not
        derivable from tracking data alone, must be supplied by the
        caller from the event annotation for this action.

        NOTE on ball_drive_prob polarity: this method's feature builder
        (PRESS.utils.action_selection) uses 1 - "Ball drive loss
        probability" (= P(success)), matching the action_features/
        pipeline the action selection model was actually trained on --
        the OPPOSITE convention from compute_transition_probabilities
        above, which uses the loss probability directly. Confirmed
        against both original training scripts, not a bug.

        Stores in
        self.results["ACTION_SELECTION"][frame_id]["Pass probability"].
        """

        if frame_id not in self.results["BCOS"]:
            raise ValueError(
                "Compute BCOS before computing action selection "
                "probability (bcos is one of the model features)."
            )

        if frame_id not in self.results["PPCF"]:
            raise ValueError(
                "Compute PPCF before computing action selection "
                "probability."
            )

        if (
            frame_id not in self.results["PPCF"]
            or "Transition probabilities"
            not in self.results["PPCF"][frame_id]
        ):
            raise ValueError(
                "Compute transition probabilities before computing "
                "action selection probability (rppcf / xT_*_pass / "
                "*_pressure_line_index_pass all depend on it)."
            )

        if (
            frame_id not in self.results["PI"]
            or "Ball drive loss probability"
            not in self.results["PI"][frame_id]
        ):
            raise ValueError(
                "Compute ball drive loss before computing action "
                "selection probability."
            )

        from PRESS.utils.action_selection import (
            build_action_selection_features,
            predict_action_selection_probability,
        )

        state = self.get_state(frame_id)

        ppcf = self.results["PPCF"][frame_id]
        bcos = self.results["BCOS"][frame_id][
            "Ball carrier open space"
        ].sum()
        ball_drive_loss = self.results["PI"][frame_id][
            "Ball drive loss probability"
        ]

        X = build_action_selection_features(
            x_bc=state["x_bc"],
            v_bc=state["v_bc"],
            r_defs=state["r_defs"],
            r_atts=state["r_atts"],
            attacking_side=state["attacking_side"],
            targets=ppcf["grid"],
            transition_probabilities=ppcf["Transition probabilities"],
            pass_success_probability=ppcf["Pass probability"],
            bcos=bcos,
            ball_drive_loss=ball_drive_loss,
            xT_grid=xT_grid,
            start_type=start_type,
            previous_action=previous_action,
        )

        pass_probability = predict_action_selection_probability(X)

        self.results["ACTION_SELECTION"][frame_id] = {
            "Pass probability": pass_probability,
            "Ball drive probability": 1.0 - pass_probability,
        }

        return pass_probability


    # ============================================================
    # INSTANTANEOUS HAZARD
    # ============================================================

    def compute_hazard(
        self,
        frame_id,
    ):
        """
        Compute the instantaneous hazard h(s_t) -- the marginal
        (action-unconditional) risk of losing possession at this
        checkpoint, mixing the drive and pass branches by the action
        selection model's P(pass):

            h(s_t) = P(drive|s_t) * drive_risk(s_t)
                   + P(pass|s_t)  * pass_failure_expected(s_t)

        where drive_risk(s_t) = Ball drive loss probability directly
        (no inversion needed here -- see PRESS.utils.hazard docstring
        for why the two inversions cancel out), and
        pass_failure_expected(s_t) = 1 - rppcf (rppcf = sum(tr *
        pass_prob) over the destination grid, using the same
        softmax-sums-to-1 simplification already used elsewhere).

        NOTE (25/9): NO post-hoc calibration is applied here. An
        isotonic calibrator was fitted and evaluated (ECE 0.0201 ->
        0.0032 out-of-fold), but the four components it calibrates
        are all PRODUCTION models fit on all 10 matches -- so both
        h(s_t) and its calibration are evaluated in-sample relative
        to the underlying component training, not genuinely
        out-of-fold. Fixing this properly would require rebuilding
        h(s_t) from per-fold (not production) versions of all four
        components on a shared match-level split. Decided not to do
        that now (low cross-fold variance in all four components
        suggests the effect is likely small) -- but also decided NOT
        to ship an unvalidated calibration layer. h(s_t) is reported
        raw.

        Requires compute_ppcf(), compute_ball_drive_loss(),
        compute_transition_probabilities() AND
        compute_action_selection_probability() to have already been
        run for this frame_id.

        Stores in self.results["HAZARD"][frame_id]:
            "Hazard"
            "Drive risk"
            "Pass failure expected"
            "rppcf"

        Returns the hazard (float).
        """

        if (
            frame_id not in self.results["PI"]
            or "Ball drive loss probability"
            not in self.results["PI"][frame_id]
        ):
            raise ValueError(
                "Compute ball drive loss before computing hazard."
            )

        if (
            frame_id not in self.results["PPCF"]
            or "Transition probabilities"
            not in self.results["PPCF"][frame_id]
        ):
            raise ValueError(
                "Compute transition probabilities before computing "
                "hazard (rppcf depends on it)."
            )

        if frame_id not in self.results["ACTION_SELECTION"]:
            raise ValueError(
                "Compute action selection probability before computing "
                "hazard (P(pass) is the mixing weight)."
            )

        from PRESS.utils.hazard import compute_hazard_raw

        ppcf = self.results["PPCF"][frame_id]
        ball_drive_loss = self.results["PI"][frame_id][
            "Ball drive loss probability"
        ]
        action_selection_pass_probability = self.results[
            "ACTION_SELECTION"
        ][frame_id]["Pass probability"]

        hazard_h, drive_risk, pass_failure_expected, rppcf = (
            compute_hazard_raw(
                ball_drive_loss=ball_drive_loss,
                transition_probabilities=ppcf["Transition probabilities"],
                pass_success_probability=ppcf["Pass probability"],
                action_selection_pass_probability=(
                    action_selection_pass_probability
                ),
            )
        )

        self.results["HAZARD"][frame_id] = {
            "Hazard": hazard_h,
            "Drive risk": drive_risk,
            "Pass failure expected": pass_failure_expected,
            "rppcf": rppcf,
        }

        return hazard_h
