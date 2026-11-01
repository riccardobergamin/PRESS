import jax.numpy as jnp


class PRESS:
    """
    One-match physics-informed model of next-action possession-loss risk.

    Components
    ----------
    PPCF:
        Physics-based Potential Pitch Control Field

    PI:
        Pressing Intensity
    """


    def __init__(
        self,
        tracking_data,
        events_data = None,
        *,
        possessions_data=None,
        actions_data=None,
        player_possessions_data=None,
        ball_drive_model=None,
        pass_risk_model=None,
        pass_selection_model=None,
    ):
        """Initialize one match without running physics or model prediction.

        tracking_data is canonical wide tracking in metres. events_data is
        an optional canonical event table (not raw SkillCorner events).
        possessions_data contains explicit team intervals: possession_id,
        team_id, frame_start, frame_end. Alternatively, player_possessions_data
        can be linked through verified receptions. actions_data contains
        frame_start, frame_end, action, start_type, previous_action. All
        intervals and frame IDs belong to this one match. See api.qmd.
        """

        from PRESS.utils.sanitizers import (
            validate_tracking_data,
            validate_events_data
        )

        validate_tracking_data(tracking_data)

        self.tracking_data = tracking_data.copy()

        if events_data is not None:

          validate_events_data(events_data)


        self.events_data = events_data

        from PRESS.model.ball_drive import BallDriveModel
        self.ball_drive_model = (ball_drive_model if isinstance(ball_drive_model, BallDriveModel)
                                 else BallDriveModel.from_json(ball_drive_model) if ball_drive_model is not None
                                 else None)

        from PRESS.model.pass_risk import PassRiskModel
        self.pass_risk_model = (pass_risk_model if isinstance(pass_risk_model, PassRiskModel)
                                else PassRiskModel.from_json(pass_risk_model) if pass_risk_model is not None
                                else None)

        from PRESS.model.pass_selection import PassSelectionModel
        self.pass_selection_model = (pass_selection_model if isinstance(pass_selection_model, PassSelectionModel)
                                     else PassSelectionModel.from_json(pass_selection_model) if pass_selection_model is not None
                                     else None)

        import numpy as np
        from PRESS.utils.match import link_team_possessions, validate_intervals

        self.player_possessions_data = None
        if player_possessions_data is not None:
            linked, self.player_possessions_data = link_team_possessions(player_possessions_data)
            if possessions_data is not None:
                raise ValueError("Supply team intervals or player intervals, not both.")
            possessions_data = linked
        self.possessions_data = None
        if possessions_data is not None:
            table = validate_intervals(possessions_data, [
                "possession_id", "team_id", "frame_start", "frame_end",
            ])
            if table.possession_id.isna().any() or table.possession_id.duplicated().any():
                raise ValueError("possession_id must be nonmissing and unique within a match.")
            if not table.team_id.isin(["home", "away"]).all():
                raise ValueError("team_id must be home or away.")
            if len(table) > 1 and np.any(table.frame_start.to_numpy()[1:] <= table.frame_end.to_numpy()[:-1]):
                raise ValueError("Team possession intervals must not overlap.")
            self.possessions_data = table
        self.actions_data = None
        if actions_data is not None:
            self.actions_data = validate_intervals(actions_data, [
                "frame_start", "frame_end", "action", "start_type", "previous_action",
            ])


        self.results = {
            "PPCF": {},
            "PI": {},
            # Stores action selection output (P(pass)) per frame_id.
            "ACTION_SELECTION": {},
            # Raw next-action hazard per frame; no post-hoc calibrator.
            "HAZARD": {},
        }

    @classmethod
    def from_skillcorner(cls, match_dir, *, actions_path=None):
        """Load processed <match_id>_tracking/_possession.parquet files.

        match_dir is one directory created by the preprocessing pipeline.
        Optional actions_path points to multi-match actions.csv/parquet; rows
        are filtered to this match before initialization. Requires the io
        extra. Raw SkillCorner JSON/events CSV are not canonical inputs.
        """
        from pathlib import Path
        import pandas as pd

        match_dir = Path(match_dir)
        match_id = match_dir.name
        tracking = pd.read_parquet(match_dir / f"{match_id}_tracking.parquet")
        players = pd.read_parquet(match_dir / f"{match_id}_possession.parquet")
        actions = None
        if actions_path is not None:
            actions = (pd.read_parquet(actions_path) if Path(actions_path).suffix == ".parquet"
                       else pd.read_csv(actions_path, dtype={"match_id": str}))
            if "match_id" not in actions:
                raise ValueError("Actions must contain match_id.")
            actions = actions.loc[actions.match_id.astype(str) == match_id].copy()
        return cls(tracking, player_possessions_data=players, actions_data=actions)

    def get_possessions(self, *, team_id=None):
        """Return a copy of explicit team intervals, optionally filtered by team.

        Raises ValueError when no team/player interval table was supplied.
        Carrier labels alone do not define verified team possessions.
        """
        if self.possessions_data is None:
            raise ValueError("Supply possessions_data or player_possessions_data to navigate possessions.")
        table = self.possessions_data
        if team_id is not None:
            if team_id not in ("home", "away"):
                raise ValueError("team_id must be home or away.")
            table = table.loc[table.team_id == team_id]
        return table.copy().reset_index(drop=True)

    def get_possession(self, possession_id):
        """Return one team interval as a copied Series; unknown ID raises KeyError."""
        table = self.get_possessions()
        selected = table.loc[table.possession_id == possession_id]
        if selected.empty:
            raise KeyError(f"Unknown possession_id {possession_id!r}.")
        return selected.iloc[0].copy()

    def get_player_possessions(self, possession_id=None):
        """Return linked player intervals, optionally restricted to one team possession."""
        if self.player_possessions_data is None:
            raise ValueError("No player_possessions_data was supplied.")
        table = self.player_possessions_data
        if possession_id is not None:
            self.get_possession(possession_id)
            table = table.loc[table.team_possession_id == possession_id]
        return table.copy().reset_index(drop=True)

    def get_frame(self, frame_id):
        """Return one wide tracking row as a copied DataFrame; unknown ID raises KeyError."""
        frame = self.tracking_data.loc[self.tracking_data.frame == frame_id]
        if frame.empty:
            raise KeyError(f"Unknown frame_id {frame_id!r}.")
        return frame.copy().reset_index(drop=True)

    def get_frames(self, possession_id=None, *, valid_only=False):
        """Return sorted wide tracking rows, optionally within a team interval.

        Intervals include both endpoints and in-flight rows. valid_only
        excludes frames lacking a carrier, direction, or finite carrier
        position/velocity; it does not certify full tracking quality.
        """
        import numpy as np
        table = self.tracking_data
        if possession_id is not None:
            interval = self.get_possession(possession_id)
            table = table.loc[table.frame.between(interval.frame_start, interval.frame_end)]
        if valid_only:
            mask = table.player_possession.notna() & table.attacking_side.isin([
                "left_to_right", "right_to_left",
            ])
            for player in table.loc[mask, "player_possession"].unique():
                selected = table.player_possession == player
                columns = [f"{player}_{suffix}" for suffix in ("x", "y", "vx", "vy")]
                mask.loc[selected] &= np.isfinite(table.loc[selected, columns]).all(axis=1)
            table = table.loc[mask]
        return table.sort_values("frame").copy().reset_index(drop=True)

    def get_checkpoints(self, possession_id=None, *, eligible_only=True, include_snapshots=False):
        """Return canonical action starts, optionally with evaluation-only snapshots.

        Adds frame_id, start_type_std, previous_action_std, eligible.
        Canonical rows include drive attempts (also early failures) and pass
        releases. Short non-loss control tails and other terminal snapshots
        are optional and never action-selection training examples.
        Source possession_id, if present, remains a PLAYER-possession ID.
        """
        from PRESS.utils.match import standardize_action_context
        if self.actions_data is None:
            raise ValueError("Supply actions_data to navigate action checkpoints.")
        table = self.actions_data.copy()
        if possession_id is not None:
            interval = self.get_possession(possession_id)
            table = table.loc[table.frame_start.between(interval.frame_start, interval.frame_end)].copy()
        table["frame_id"] = table.frame_start
        if "is_canonical" not in table:
            table["is_canonical"] = table.action.isin(["ball_drive", "pass"])
        table["checkpoint_kind"] = table.is_canonical.map({True: "canonical", False: "snapshot"})
        table["start_type_std"] = table.start_type.map(standardize_action_context)
        table["previous_action_std"] = table.previous_action.map(
            lambda value: standardize_action_context(value, previous=True)
        )
        valid_frames = self.get_frames(possession_id, valid_only=True).frame
        table["eligible"] = (
            ((table.is_canonical & table.action.isin(["ball_drive", "pass"]))
             | (include_snapshots & table.action.eq("snapshot")))
            & table.frame_id.isin(valid_frames)
            & table.start_type_std.notna()
            & table.previous_action_std.notna()
        )
        if eligible_only:
            table = table.loc[table.eligible]
            if table.frame_id.duplicated().any():
                raise ValueError("Ambiguous duplicate checkpoints; resolve actions_data first.")
        return table.reset_index(drop=True)

    def get_frame_context(self, frame_id):
        """Read standardized action context for any annotated tracking frame.

        Uses per-frame tracking context first, then an action interval
        containing this frame. At shared boundaries the latest interval
        start wins; conflicting contexts at that start raise ValueError.
        Context annotations are separate from the frames selected for a
        possession trajectory. Missing/unknown categories are returned as
        None so callers can supply the actual context explicitly.
        """
        from PRESS.utils.match import standardize_action_context

        frame = self.get_frame(frame_id).iloc[0]
        for start_column, previous_column in (
            ("start_type_std", "previous_action_std"),
            ("start_type", "previous_action"),
        ):
            if start_column in frame and previous_column in frame:
                start = standardize_action_context(frame[start_column])
                previous = standardize_action_context(frame[previous_column], previous=True)
                if start is not None and previous is not None:
                    return {"start_type": start, "previous_action": previous,
                            "source": "tracking"}
        if self.actions_data is not None:
            intervals = self.actions_data.loc[
                (self.actions_data.frame_start <= frame_id)
                & (self.actions_data.frame_end >= frame_id)
            ]
            if not intervals.empty:
                intervals = intervals.loc[intervals.frame_start == intervals.frame_start.max()]
                contexts = {
                    (standardize_action_context(row.start_type),
                     standardize_action_context(row.previous_action, previous=True))
                    for row in intervals.itertuples(index=False)
                }
                if len(contexts) != 1:
                    raise ValueError("Conflicting action context for this frame; supply context explicitly.")
                start, previous = contexts.pop()
                return {"start_type": start, "previous_action": previous,
                        "source": "action_interval"}
        raise ValueError("No action context annotation covers this frame; supply start_type and previous_action explicitly.")

    def compute_frame(
        self, frame_id, xT_grid, *, start_type=None, previous_action=None,
        bcos_value=None, ppcf_kwargs=None, ball_drive_kwargs=None, bcos_kwargs=None,
    ):
        """Compute all core outputs in dependency order for one frame.

        Any usable tracking frame can be evaluated independently of the
        possession sampling. Resolve missing context from frame annotations
        or a containing action interval; explicit arguments override it.
        Accepts raw or standardized categories. Supply bcos_value to reuse
        an external BCOS sum; otherwise an optional BCOS instance computes
        it. Keyword dictionaries are passed to their component methods.
        Always recomputes this frame; returns all four core result mappings.
        """
        import numpy as np
        from PRESS.utils.match import standardize_action_context

        self.get_frame(frame_id)
        if start_type is None or previous_action is None:
            context = self.get_frame_context(frame_id)
            if start_type is None:
                start_type = context["start_type"]
            if previous_action is None:
                previous_action = context["previous_action"]
        choice_start, choice_previous = start_type, previous_action
        start_type = standardize_action_context(start_type)
        previous_action = standardize_action_context(previous_action, previous=True)
        if start_type is None or previous_action is None:
            raise ValueError("Unsupported action context; unknown categories were excluded from model training.")
        grid = np.asarray(xT_grid, dtype=float)
        if grid.ndim != 2 or 0 in grid.shape or not np.isfinite(grid).all() or (grid < 0).any():
            raise ValueError("xT_grid must be a finite nonnegative (n_x, n_y) array.")
        if bcos_value is not None and (not np.isfinite(bcos_value) or bcos_value < 0):
            raise ValueError("bcos_value must be a finite nonnegative scalar.")
        self.compute_ppcf(frame_id, **(ppcf_kwargs or {}))
        self.compute_ball_drive_loss(frame_id, **(ball_drive_kwargs or {}))
        self.compute_transition_probabilities(frame_id, grid, start_type_raw=choice_start,
                                              previous_action_raw=choice_previous)
        if bcos_value is None:
            from PRESS.extras import BCOS
            bcos = BCOS(self.tracking_data)
            field = bcos.compute_bcos(frame_id, **(bcos_kwargs or {}))
            bcos_value = float(np.asarray(field["Ball carrier open space"]).sum())
        self.compute_action_selection_probability(frame_id, grid, start_type, previous_action, bcos_value)
        self.compute_hazard(frame_id)
        return {key: self.get_result(key, frame_id) for key in self.results}

    def compute_possession(self, possession_id, xT_grid, *, frame_ids=None,
                           include_snapshots=True, **frame_kwargs):
        """Evaluate a possession at a chosen sampling of tracking frames.

        frame_ids=None selects canonical starts plus evaluation-only snapshots
        by default. include_snapshots=False restricts this to canonical rows.
        Explicit nonempty unique integer frame_ids choose another sampling;
        all must be usable frames inside the selected possession. Returns
        rows sorted by frame_id. Each row calls the same compute_frame API;
        sampling does not change its physics horizon or predict the outcome.
        checkpoint_kind/is_canonical distinguish additional evaluations;
        their overlapping forecasts must not be multiplied into survival.
        """
        import numpy as np
        import pandas as pd
        self.get_possession(possession_id)
        kinds = {}
        if self.actions_data is not None:
            checkpoint_table = self.get_checkpoints(possession_id, eligible_only=False, include_snapshots=True)
            kinds = dict(zip(checkpoint_table.frame_id, checkpoint_table.checkpoint_kind))
        if frame_ids is None:
            frames = self.get_checkpoints(possession_id, include_snapshots=include_snapshots).frame_id.to_numpy()
            if not len(frames):
                raise ValueError("This possession has no eligible default checkpoints; supply frame_ids for another sampling.")
        else:
            frames = np.asarray(frame_ids)
            if (frames.ndim != 1 or not frames.size
                    or not np.issubdtype(frames.dtype, np.number)
                    or np.issubdtype(frames.dtype, np.complexfloating)
                    or not np.isfinite(frames).all() or np.any(frames != np.floor(frames))):
                raise ValueError("frame_ids must be a nonempty one-dimensional sequence of integer frame IDs.")
            if len(np.unique(frames)) != len(frames):
                raise ValueError("frame_ids must not contain duplicate frames.")
            valid = self.get_frames(possession_id, valid_only=True).frame
            if not np.isin(frames, valid).all():
                raise ValueError("All frame_ids must be usable tracking frames inside this possession.")
        frames = np.sort(frames)
        rows = []
        for frame in frames:
            result = self.compute_frame(int(frame), xT_grid, **frame_kwargs)
            rows.append({"possession_id": possession_id, "frame_id": int(frame),
                         "checkpoint_kind": kinds.get(frame, "custom"),
                         "is_canonical": kinds.get(frame) == "canonical",
                         "p_pass": result["ACTION_SELECTION"]["Pass probability"],
                         **result["HAZARD"]})
        return pd.DataFrame(rows)

    def _invalidate(self, frame_id, *components):
        """Discard dependent results when an upstream component is recomputed."""
        for component in components:
            self.results[component].pop(frame_id, None)


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

        ``component`` accepts the result keys ``PPCF``, ``PI``,
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
    # PITCH CONTROL FIELD
    # ============================================================

    def _choice_model(self):
        """Resolve an explicitly supplied or installed final choice refit."""
        if self.pass_selection_model is None:
            from pathlib import Path
            from PRESS.model.pass_selection import PassSelectionModel
            path = Path(__file__).resolve().parents[1]/'utils/data/pass_selection_model.json'
            if path.is_file():
                self.pass_selection_model = PassSelectionModel.from_json(path)
        return self.pass_selection_model

    def compute_ppcf(self,frame_id,r_max=150,step=1.0,physics_kwargs=None,model=None):
        """Store raw control and isotonic-calibrated pass success on a pitch grid.

        Uses the frozen trajectory-MLE PassRiskModel, exact target distances
        and shared-survival exponential integration. model overrides the
        instance model; otherwise the installed full-data refit is loaded.
        Loading performs no training. Physics overrides are sensitivities,
        recorded as outside the saved calibration when settings differ.
        """
        import numpy as np
        from PRESS.model.pass_risk import PassRiskModel
        from PRESS.utils.grid import generate_circular_grid
        fitted=model if model is not None else self.pass_risk_model
        if fitted is None:fitted=PassRiskModel.pretrained()
        if not isinstance(fitted,PassRiskModel):fitted=PassRiskModel.from_json(fitted)
        state=self.get_state(frame_id)
        if self._choice_model() is not None:
            if step != 1.0:
                raise ValueError('The final choice model requires the native one-metre pitch grid.')
            length,width=state['pitch_length'],state['pitch_width']
            if abs(length-round(length))>1e-8 or abs(width-round(width))>1e-8:
                raise ValueError('Final choice grid requires integer pitch dimensions.')
            xx,yy=np.meshgrid(np.arange(round(length))+.5-length/2,
                              np.arange(round(width))+.5-width/2,indexing='ij')
            grid=np.column_stack([xx.ravel(),yy.ravel()]).astype(np.float32)
            if np.any(np.linalg.norm(grid-np.asarray(state['x_bc']),axis=1)>r_max):
                raise ValueError('r_max must cover the complete pitch for the final choice model.')
        else:
            grid=np.asarray(generate_circular_grid(r_max,step))+np.asarray(state['x_bc'])
        bounds=np.array([[-state['pitch_length']/2,state['pitch_length']/2],
                         [-state['pitch_width']/2,state['pitch_width']/2]])
        targets=grid[np.all((grid>=bounds[:,0]) & (grid<=bounds[:,1]),axis=1)]
        positions=np.vstack([state['r_atts'],np.asarray(state['x_bc'])[None,:],state['r_defs']])
        velocities=np.vstack([state['v_atts'],np.asarray(state['v_bc'])[None,:],state['v_defs']])
        carrier=len(state['r_atts']);attackers=np.arange(carrier+1)
        result=fitted.predict_field(positions,velocities,state['x_bc'],targets,attackers,carrier,
                                   physics_kwargs=physics_kwargs)
        self.results['PPCF'][frame_id]={
            'Pitch control field':result['control'],
            'Attacking pitch control field':result['raw'],
            'Pass probability':result['calibrated'],
            'attacker_idx':attackers,'ball_carrier_idx':carrier,
            'grid':targets,'ball flight time':result['flight_time'],
            'params':dict(model_id=fitted.model_id,lambda_control=fitted.lambda_control,s=fitted.s,
                          physics=result['physics'],calibration='isotonic',
                          calibration_matches_physics=result['calibration_matches_physics'],r_max=r_max,step=step)}
        self._invalidate(frame_id,'ACTION_SELECTION','HAZARD')


    def compute_transition_probabilities(
        self,
        frame_id,
        xT_grid,
        *, start_type_raw=None, previous_action_raw=None,
        match_clock_min=None, score_difference_prefix=None,
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

        final_model = self._choice_model()
        if final_model is not None:
            import numpy as np
            # Keep provider-level labels here; action selection separately
            # standardizes its own context. Never infer score from future events.
            frame = self.get_frame(frame_id).iloc[0]
            for column, value in [('start_type', start_type_raw), ('previous_action', previous_action_raw)]:
                if value is None and column in frame:
                    if column == 'start_type': start_type_raw = frame[column]
                    else: previous_action_raw = frame[column]
            clock = frame.get('match_clock_min', np.nan) if match_clock_min is None else match_clock_min
            score = frame.get('score_difference_prefix', np.nan) if score_difference_prefix is None else score_difference_prefix
            matrix = final_model.build_features(x_bc=state['x_bc'], v_bc=state['v_bc'],
                r_atts=state['r_atts'], r_defs=state['r_defs'], targets=ppcf['grid'],
                attacking_side=state['attacking_side'],
                target_pitch_control=ppcf['Attacking pitch control field'],
                ball_drive_prob=self.results['PI'][frame_id]['Ball drive loss probability'],
                start_type_raw=start_type_raw, previous_action_raw=previous_action_raw,
                match_clock_min=clock, score_difference_prefix=score,
                pitch_length=state['pitch_length'], pitch_width=state['pitch_width'])
            membership = None
            if final_model.family == 'cnl':
                control = np.asarray(ppcf['Pitch control field'])[ppcf['attacker_idx']]
                control = np.delete(control, ppcf['ball_carrier_idx'], axis=0).T
                valid = np.linalg.norm(state['r_atts'], axis=1)<500
                control[:,~valid] = 0.
                total=control.sum(axis=1,keepdims=True)
                if not valid.any(): raise ValueError('CNL needs at least one valid teammate.')
                with np.errstate(divide='ignore',invalid='ignore'):
                    membership=np.log(control)-np.log(total)
                membership=np.where(total==0,np.where(valid,-np.log(valid.sum()),-np.inf),membership)
            probabilities=final_model.predict_features(matrix,log_membership=membership)
            ppcf['Transition probabilities']=probabilities
            ppcf['transition_model_id']=final_model.model_id
            self._invalidate(frame_id,'ACTION_SELECTION','HAZARD')
            return probabilities

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
            pitch_length=state["pitch_length"],
            pitch_width=state["pitch_width"],
        )

        transition_probabilities = predict_transition_probabilities(X)

        self.results["PPCF"][frame_id][
            "Transition probabilities"
        ] = transition_probabilities

        self._invalidate(frame_id, "ACTION_SELECTION", "HAZARD")

        return transition_probabilities


    # ============================================================
    # PRESSING INTENSITY
    # ============================================================

    def compute_ball_drive_loss(
        self,
        frame_id,
        t_press=1.0,
        physics_kwargs=None,
        model=None,
    ):
        """
        Compute and store loss probabilities for one arbitrary frame.

        model optionally supplies an exported BallDriveModel or JSON path;
        otherwise the constructor's ball_drive_model is used. With a fitted
        model, physics_kwargs are explicit physics-only overrides (a_linear,
        a_turn, v_max, dt, pitch_length/width, preserve_initial_overspeed).
        Its saved pitch convention is used by default, matching the benchmark.
        With no fitted model the historical log-linear parameter API remains.
        """

        from PRESS.model.solver_intensity import (
            integrate_loss_prob,
        )

        from PRESS.model.ball_drive import BallDriveModel
        fitted = self.ball_drive_model if model is None else model
        if fitted is not None:
            if not isinstance(fitted, BallDriveModel):
                fitted = BallDriveModel.from_json(fitted)
            state = self.get_state(frame_id)
            result = fitted.predict(state['r_defs'], state['v_defs'], state['x_bc'], state['v_bc'],
                                    horizon=t_press, physics_kwargs=physics_kwargs)
            self.results['PI'][frame_id] = {
                'Individual tackle probabilities': result['cause_probabilities'],
                'Ball drive loss probability': result['loss_probability'],
                'Survival probability': result['survival_probability'],
                'Survival curve': result['survival_curve'],
                'Times': result['times'],
                'params': {'t_press': t_press, 'physics_kwargs': result['physics'],
                           'model_id': fitted.model_id},
            }
            self.results['PPCF'].get(frame_id, {}).pop('Transition probabilities', None)
            self._invalidate(frame_id, 'ACTION_SELECTION', 'HAZARD')
            return

        if physics_kwargs is None:

            physics_kwargs = {

                "v_max": 7.0,

                "a_max": 7.0,

                "lambda_0": 0.36,

                "beta_v": 0.09,

                "beta_B": -0.02,

                "s": 0.43,
            }

        physics_kwargs = dict(physics_kwargs)
        physics_kwargs.setdefault("integration_method", "exponential")
        physics_kwargs.setdefault("dt", 0.04)

        state = self.get_state(
            frame_id,
        )

        p_loss = integrate_loss_prob(
            state["r_defs"],
            state["v_defs"],
            state["x_bc"],
            state["v_bc"],
            t_press,
            **(dict(physics_kwargs) | {"pitch_length": state["pitch_length"],
                                      "pitch_width": state["pitch_width"]}),
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

        self.results["PPCF"].get(frame_id, {}).pop("Transition probabilities", None)
        self._invalidate(frame_id, "ACTION_SELECTION", "HAZARD")


    # ============================================================
    # ACTION SELECTION
    # ============================================================

    def compute_action_selection_probability(
        self,
        frame_id,
        xT_grid,
        start_type,
        previous_action,
        bcos_value,
    ):
        """
        Compute P(pass) vs P(ball_drive) for the ball carrier at
        frame_id (trained XGBoost action selection model, single
        sigmoid over the instantaneous state -- not a per-destination
        choice set like compute_transition_probabilities).

        Requires compute_ppcf(), compute_ball_drive_loss(), and
        compute_transition_probabilities() to have already been run for
        this frame_id. ``bcos_value`` is supplied by the optional
        ``PRESS.extras.BCOS`` component because BCOS remains a trained
        action-selection feature but is not part of the core PRESS class.

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
            bcos=bcos_value,
            ball_drive_loss=ball_drive_loss,
            xT_grid=xT_grid,
            start_type=start_type,
            previous_action=previous_action,
            pitch_length=state["pitch_length"],
            pitch_width=state["pitch_width"],
        )

        pass_probability = predict_action_selection_probability(X)

        self.results["ACTION_SELECTION"][frame_id] = {
            "Pass probability": pass_probability,
            "Ball drive probability": 1.0 - pass_probability,
        }

        self._invalidate(frame_id, "HAZARD")

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
