"""Component orchestration; all numerical models live in PRESS.model."""

import jax.numpy as jnp


class ComputationMixin:
    """Internal implementation of the public PRESS methods."""

    def _action_model(self):
        """Resolve and retain the explicit or bundled final action model."""
        if self.action_selection_model is None:
            from PRESS.model import ActionSelectionModel

            self.action_selection_model = ActionSelectionModel.pretrained()
        return self.action_selection_model

    def compute_frame(
        self,
        frame_id,
        xT_grid=None,
        *,
        start_type=None,
        previous_action=None,
        bcos_value=None,
        ppcf_kwargs=None,
        ball_drive_kwargs=None,
        bcos_kwargs=None,
        pass_selection_kwargs=None,
        match_clock_min=None,
        score_difference_prefix=None,
        completed_canonical_drives=None,
        include_residual=False,
        time_since_start=None,
    ):
        """Compute all core outputs in dependency order for one frame.

        Any usable tracking frame can be evaluated independently of the
        possession sampling. Resolve missing context from frame annotations
        or a containing action interval; explicit arguments override it.
        The fitted model accepts raw categories with missing/unseen buckets.
        Supply bcos_value to reuse an external BCOS area; otherwise BCOS
        computes it. Keyword dictionaries pass to their component methods.
        Always recomputes this frame; returns four core result mappings. With
        include_residual=True, also computes residual risk and xLoss using
        observed/supplied elapsed time. Omitted xT_grid uses the fitted surface.
        """
        import numpy as np

        self.get_frame(frame_id)
        context = (
            self.get_frame_context(frame_id)
            if start_type is None or previous_action is None
            else {}
        )
        if start_type is None:
            start_type = context.get('start_type')
        if previous_action is None:
            previous_action = context.get('previous_action')
        choice_start, choice_previous = (start_type, previous_action)
        grid = self._resolve_xt_grid(xT_grid)
        if include_residual:
            if time_since_start is None:
                time_since_start = self.get_elapsed_time(frame_id)
            try:
                time_since_start = float(time_since_start)
            except (TypeError, ValueError):
                raise ValueError('time_since_start must be finite and nonnegative seconds.') from None
            if not np.isfinite(time_since_start) or time_since_start < 0:
                raise ValueError('time_since_start must be finite and nonnegative seconds.')
        if bcos_value is not None and (not np.isfinite(bcos_value) or bcos_value < 0):
            raise ValueError('bcos_value must be a finite nonnegative scalar.')
        self.compute_ppcf(frame_id, **ppcf_kwargs or {})
        self.compute_ball_drive_risk(frame_id, **ball_drive_kwargs or {})
        choice_context = dict(pass_selection_kwargs or {})
        choice_context.setdefault('start_type_raw', choice_start)
        choice_context.setdefault('previous_action_raw', choice_previous)
        if match_clock_min is not None:
            choice_context['match_clock_min'] = match_clock_min
        if score_difference_prefix is not None:
            choice_context['score_difference_prefix'] = score_difference_prefix
        self.compute_pass_selection(frame_id, grid, **choice_context)
        if bcos_value is None:
            field = self.compute_bcos(frame_id, **bcos_kwargs or {})
            bcos_value = float(field['BCOS area'])
        self.compute_action_selection_probability(
            frame_id,
            grid,
            start_type,
            previous_action,
            bcos_value,
            match_clock_min=match_clock_min,
            score_difference_prefix=score_difference_prefix,
            completed_canonical_drives=completed_canonical_drives,
        )
        self.compute_hazard(frame_id)
        components = ['PPCF', 'BALL_DRIVE_RISK', 'ACTION_SELECTION', 'HAZARD']
        if include_residual:
            self.compute_residual_risk(frame_id, time_since_start=time_since_start)
            self.compute_xloss(frame_id)
            components += ['RESIDUAL_RISK', 'XLOSS']
        return {key: self.get_result(key, frame_id) for key in components}

    def compute_possession(
        self,
        possession_id,
        xT_grid=None,
        *,
        frame_ids=None,
        include_snapshots=True,
        **frame_kwargs,
    ):
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
            checkpoint_table = self.get_checkpoints(
                possession_id,
                eligible_only=False,
                include_snapshots=True,
            )
            kinds = dict(zip(checkpoint_table.frame_id, checkpoint_table.checkpoint_kind))
        if frame_ids is None:
            frames = self.get_checkpoints(
                possession_id,
                include_snapshots=include_snapshots,
            ).frame_id.to_numpy()
            if not len(frames):
                raise ValueError(
                    'This possession has no eligible default checkpoints; supply '
                    'frame_ids for another sampling.'
                )
        else:
            frames = np.asarray(frame_ids)
            if (
                frames.ndim != 1
                or not frames.size
                or (not np.issubdtype(frames.dtype, np.number))
                or np.issubdtype(frames.dtype, np.complexfloating)
                or (not np.isfinite(frames).all())
                or np.any(frames != np.floor(frames))
            ):
                raise ValueError('frame_ids must be a nonempty one-dimensional sequence of integer frame IDs.')
            if len(np.unique(frames)) != len(frames):
                raise ValueError('frame_ids must not contain duplicate frames.')
            valid = self.get_frames(possession_id, valid_only=True).frame
            if not np.isin(frames, valid).all():
                raise ValueError('All frame_ids must be usable tracking frames inside this possession.')
        frames = np.sort(frames)
        rows = []
        for frame in frames:
            result = self.compute_frame(int(frame), xT_grid, **frame_kwargs)
            rows.append({
                'possession_id': possession_id,
                'frame_id': int(frame),
                'checkpoint_kind': kinds.get(frame, 'custom'),
                'is_canonical': kinds.get(frame) == 'canonical',
                'p_pass': result['ACTION_SELECTION']['Pass probability'],
                **result['HAZARD'],
                **({'xLoss': result['XLOSS']['xLoss']} if 'XLOSS' in result else {}),
            })
        return pd.DataFrame(rows)

    def _choice_model(self):
        """Resolve and retain the explicit or bundled final choice model."""
        if self.pass_selection_model is None:
            from PRESS.model import PassSelectionModel

            self.pass_selection_model = PassSelectionModel.pretrained()
        return self.pass_selection_model

    def compute_ppcf(self, frame_id, r_max=150, step=1.0, physics_kwargs=None, model=None):
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

        fitted = model if model is not None else self.pass_risk_model
        if fitted is None:
            fitted = PassRiskModel.pretrained()
        if not isinstance(fitted, PassRiskModel):
            fitted = PassRiskModel.from_json(fitted)
        state = self.get_state(frame_id)
        if step == 1.0:
            length, width = (state['pitch_length'], state['pitch_width'])
            # Keep complete 1m cells centred on the pitch. Integer dimensions
            # reproduce the benchmark exactly; fractional margins remain outside
            # these full cells and are flagged as a different inference grid.
            nx, ny = max(1, int(np.floor(length))), max(1, int(np.floor(width)))
            xx, yy = np.meshgrid(
                np.arange(nx) + 0.5 - nx / 2,
                np.arange(ny) + 0.5 - ny / 2,
                indexing='ij',
            )
            grid = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
            grid = grid[np.linalg.norm(grid - np.asarray(state['x_bc']), axis=1) <= r_max]
        else:
            grid = np.asarray(generate_circular_grid(r_max, step)) + np.asarray(state['x_bc'])
        bounds = np.array([
            [-state['pitch_length'] / 2, state['pitch_length'] / 2],
            [-state['pitch_width'] / 2, state['pitch_width'] / 2],
        ])
        targets = grid[np.all((grid >= bounds[:, 0]) & (grid <= bounds[:, 1]), axis=1)]
        if not len(targets):
            raise ValueError('The requested pitch grid has no candidate cells.')
        positions = np.vstack([
            state['r_atts'],
            np.asarray(state['x_bc'])[None, :],
            state['r_defs'],
        ])
        velocities = np.vstack([
            state['v_atts'],
            np.asarray(state['v_bc'])[None, :],
            state['v_defs'],
        ])
        carrier = len(state['r_atts'])
        attackers = np.arange(carrier + 1)
        result = fitted.predict_field(
            positions,
            velocities,
            state['x_bc'],
            targets,
            attackers,
            carrier,
            physics_kwargs=physics_kwargs,
        )
        self.results['PPCF'][frame_id] = {
            'Pitch control field': result['control'],
            'Attacking pitch control field': result['raw'],
            'Pass probability': result['calibrated'],
            'attacker_idx': attackers,
            'ball_carrier_idx': carrier,
            'grid': targets,
            'ball flight time': result['flight_time'],
            'params': dict(
                model_id=fitted.model_id,
                lambda_control=fitted.lambda_control,
                s=fitted.s,
                physics=result['physics'],
                calibration='isotonic',
                calibration_matches_physics=result['calibration_matches_physics'],
                r_max=r_max,
                step=step,
                choice_grid_matches_benchmark=(step == 1.0 and state['pitch_length'].is_integer() and
                    state['pitch_width'].is_integer() and
                    len(targets) == round(state['pitch_length']) * round(state['pitch_width'])),
            ),
        }
        self._invalidate(frame_id, 'ACTION_SELECTION', 'HAZARD')

    def compute_pass_selection(
        self,
        frame_id,
        xT_grid=None,
        *,
        start_type_raw=None,
        previous_action_raw=None,
        match_clock_min=None,
        score_difference_prefix=None,
    ):
        """
        Compute the destination CHOICE probability distribution
        (softmax over the trained XGBoost pass selection model) for the
        ball carrier at frame_id.

        Requires compute_ppcf() AND compute_ball_drive_risk() to have
        already been run for this frame_id.

        Stores in self.results["PPCF"][frame_id]["Pass selection probabilities"].
        """
        import numpy as np

        if frame_id not in self.results['PPCF']:
            raise ValueError('Compute PPCF before computing pass selection probabilities')
        if (
            frame_id not in self.results['BALL_DRIVE_RISK']
            or 'Ball drive risk' not in self.results['BALL_DRIVE_RISK'][frame_id]
        ):
            raise ValueError(
                'Compute ball drive loss before computing pass selection probabilities '
                '(ball_drive_prob is one of the model features).'
            )

        state = self.get_state(frame_id)
        ppcf = self.results['PPCF'][frame_id]
        final_model = self._choice_model()
        frame = self.get_frame(frame_id).iloc[0]
        raw_context = self._raw_action_context(frame_id)
        context = {
            'start_type_raw': raw_context.get('start_type') if start_type_raw is None else start_type_raw,
            'previous_action_raw': (
                raw_context.get('previous_action')
                if previous_action_raw is None else previous_action_raw
            ),
        }
        if self.results['BALL_DRIVE_RISK'][frame_id]['params']['t_press'] != 1.0:
            raise ValueError(
                'Pass selection requires the one-second ball-drive loss feature; '
                'recompute it with t_press=1.0.'
            )
        clock = (
            frame.get('match_clock_min', np.nan)
            if match_clock_min is None
            else match_clock_min
        )
        score = (
            frame.get('score_difference_prefix', np.nan)
            if score_difference_prefix is None
            else score_difference_prefix
        )
        matrix = final_model.build_features(
            x_bc=state['x_bc'],
            v_bc=state['v_bc'],
            r_atts=state['r_atts'],
            r_defs=state['r_defs'],
            targets=ppcf['grid'],
            attacking_side=state['attacking_side'],
            target_pitch_control=ppcf['Attacking pitch control field'],
            ball_drive_prob=self.results['BALL_DRIVE_RISK'][frame_id]['Ball drive risk'],
            start_type_raw=context['start_type_raw'],
            previous_action_raw=context['previous_action_raw'],
            match_clock_min=clock,
            score_difference_prefix=score,
            pitch_length=state['pitch_length'],
            pitch_width=state['pitch_width'],
        )
        membership = None
        if final_model.family == 'cnl':
            control = np.asarray(ppcf['Pitch control field'])[ppcf['attacker_idx']]
            control = np.delete(control, ppcf['ball_carrier_idx'], axis=0).T
            valid = np.linalg.norm(state['r_atts'], axis=1) < 500
            control[:, ~valid] = 0.0
            total = control.sum(axis=1, keepdims=True)
            if not valid.any():
                raise ValueError('CNL needs at least one valid teammate.')
            with np.errstate(divide='ignore', invalid='ignore'):
                membership = np.log(control) - np.log(total)
            membership = np.where(
                total == 0,
                np.where(valid, -np.log(valid.sum()), -np.inf),
                membership,
            )
        probabilities = final_model.predict_features(matrix, log_membership=membership)
        ppcf['Pass selection probabilities'] = probabilities
        ppcf['pass_selection_model_id'] = final_model.model_id
        ppcf['pass_selection_context'] = dict(
            context,
            match_clock_min=clock,
            score_difference_prefix=score,
        )
        self._invalidate(frame_id, 'ACTION_SELECTION', 'HAZARD')
        return probabilities

    def compute_ball_drive_risk(self, frame_id, t_press=1.0, physics_kwargs=None, model=None):
        """
        Compute and store loss probabilities for one arbitrary frame.

        model optionally supplies an exported BallDriveModel or JSON path;
        otherwise the constructor's ball_drive_model or bundled final spline
        refit is used. physics_kwargs accepts physics-only overrides (a_linear,
        a_turn, v_max, dt, pitch_length/width, preserve_initial_overspeed).
        Native frame dimensions are supplied to match choice-model features;
        explicit pitch overrides remain available for sensitivity analysis.
        """
        from PRESS.model.ball_drive import BallDriveModel

        fitted = self.ball_drive_model if model is None else model
        if fitted is None:
            fitted = BallDriveModel.pretrained()
        elif not isinstance(fitted, BallDriveModel):
            fitted = BallDriveModel.from_json(fitted)
        state = self.get_state(frame_id)
        result = fitted.predict(
            state['r_defs'],
            state['v_defs'],
            state['x_bc'],
            state['v_bc'],
            horizon=t_press,
            physics_kwargs=dict(
                pitch_length=state['pitch_length'],
                pitch_width=state['pitch_width'],
            ) | dict(physics_kwargs or {}),
        )
        self.results['BALL_DRIVE_RISK'][frame_id] = {
            'Individual tackle probabilities': result['cause_probabilities'],
            'Ball drive risk': result['loss_probability'],
            'Survival probability': result['survival_probability'],
            'Survival curve': result['survival_curve'],
            'Times': result['times'],
            'params': {
                't_press': t_press,
                'physics_kwargs': result['physics'],
                'model_id': fitted.model_id,
            },
        }
        self.results['PPCF'].get(frame_id, {}).pop('Pass selection probabilities', None)
        self._invalidate(frame_id, 'ACTION_SELECTION', 'HAZARD')

    def compute_action_selection_probability(
        self,
        frame_id,
        xT_grid=None,
        start_type=None,
        previous_action=None,
        bcos_value=None,
        *,
        match_clock_min=None,
        score_difference_prefix=None,
        completed_canonical_drives=None,
    ):
        """
        Compute P(pass) vs P(ball_drive) for the ball carrier at
        frame_id (trained XGBoost action selection model, single
        sigmoid over the instantaneous state -- not a per-destination
        choice set like compute_pass_selection).

        Requires compute_ppcf(), compute_ball_drive_risk(), and
        compute_pass_selection() to have already been run for
        this frame_id. ``bcos_value`` is supplied by the optional
        ``PRESS.extras.BCOS`` component because BCOS remains a trained
        action-selection feature but is not part of the core PRESS class.

        The revised model uses raw start/previous categories, e.g.
        "pass_reception" and "ball_drive", preserving unknown/missing.
        Optional clock, prefix score and completed canonical carries can be
        supplied explicitly; otherwise resolve them from stored context.

        The feature builder receives the stored one-second loss probability
        and converts it once to drive_success_1s = 1 - ball_drive_loss.
        Pass selection keeps the loss probability directly; residual risk
        reuses the converted action record. Stored ball drive risk is unchanged.

        Stores in
        self.results["ACTION_SELECTION"][frame_id]["Pass probability"].
        """
        if frame_id not in self.results['PPCF']:
            raise ValueError('Compute PPCF before computing action selection probability.')
        if (
            frame_id not in self.results['PPCF']
            or 'Pass selection probabilities' not in self.results['PPCF'][frame_id]
        ):
            raise ValueError(
                'Compute pass selection probabilities before computing action selection '
                'probability (rppcf / xT_*_pass / *_pressure_line_index_pass all '
                'depend on it).'
            )
        if (
            frame_id not in self.results['BALL_DRIVE_RISK']
            or 'Ball drive risk' not in self.results['BALL_DRIVE_RISK'][frame_id]
        ):
            raise ValueError('Compute ball drive loss before computing action selection probability.')

        state = self.get_state(frame_id)
        ppcf = self.results['PPCF'][frame_id]
        ball_drive_loss = self.results['BALL_DRIVE_RISK'][frame_id]['Ball drive risk']
        if bcos_value is None:
            cached_bcos = self.results['BCOS'].get(frame_id)
            if cached_bcos is None:
                raise ValueError('Supply bcos_value or compute BCOS before action selection.')
            bcos_value = float(cached_bcos['BCOS area'])
        final = self._action_model()
        import numpy as np
        if self.results['BALL_DRIVE_RISK'][frame_id]['params']['t_press'] != 1.0:
            raise ValueError('Action selection requires one-second carry loss.')
        frame = self.get_frame(frame_id).iloc[0]
        prefix = ppcf.get('pass_selection_context', {})
        context = self._raw_action_context(frame_id)
        raw_start = start_type if start_type is not None else context['start_type']
        raw_previous = (
            previous_action
            if previous_action is not None
            else context['previous_action']
        )
        clock = prefix.get(
            'match_clock_min',
            frame.get('match_clock_min', np.nan),
        ) if match_clock_min is None else match_clock_min
        score = prefix.get(
            'score_difference_prefix',
            frame.get('score_difference_prefix', np.nan),
        ) if score_difference_prefix is None else score_difference_prefix
        count = (
            context['completed_canonical_drives']
            if completed_canonical_drives is None
            else completed_canonical_drives
        )
        feature_record = final.build_feature_record(
            x_bc=state['x_bc'],
            v_bc=state['v_bc'],
            r_defs=state['r_defs'],
            r_atts=state['r_atts'],
            attacking_side=state['attacking_side'],
            targets=ppcf['grid'],
            pass_selection_probabilities=ppcf['Pass selection probabilities'],
            pass_success_probability=ppcf['Pass probability'],
            bcos=bcos_value,
            ball_drive_loss=ball_drive_loss,
            start_type_raw=raw_start,
            previous_action_raw=raw_previous,
            match_clock_min=clock,
            score_difference_prefix=score,
            completed_canonical_drives=count,
            pitch_length=state['pitch_length'],
            pitch_width=state['pitch_width'],
        )
        import pandas as pd
        X = final.encode(pd.DataFrame([feature_record]))
        pass_probability = float(final.predict_features(X)[0])
        self.results['ACTION_SELECTION'][frame_id] = {
            'Pass probability': pass_probability,
            'Ball drive probability': 1.0 - pass_probability,
            'Features': feature_record,
            'model_id': final.model_id,
        }
        self._invalidate(frame_id, 'HAZARD')
        return pass_probability

    def compute_hazard(self, frame_id):
        """Compose native next-action loss before the pass/carry choice.

        h = (1-P(pass))*drive_loss + P(pass)*sum(choice*(1-pass_success)).
        Uses the stored calibrated pass completion and native action policy;
        there is no post-hoc calibration on h. Requires all four component
        outputs for this frame. Returns a float and stores branch quantities.
        This is not the terminal individual-possession outcome (see xLoss).
        """
        if (
            frame_id not in self.results['BALL_DRIVE_RISK']
            or 'Ball drive risk' not in self.results['BALL_DRIVE_RISK'][frame_id]
        ):
            raise ValueError('Compute ball drive loss before computing hazard.')
        if (
            frame_id not in self.results['PPCF']
            or 'Pass selection probabilities' not in self.results['PPCF'][frame_id]
        ):
            raise ValueError(
                'Compute pass selection probabilities before computing hazard (rppcf '
                'depends on it).'
            )
        if frame_id not in self.results['ACTION_SELECTION']:
            raise ValueError(
                'Compute action selection probability before computing hazard '
                '(P(pass) is the mixing weight).'
            )
        from PRESS.utils.hazard import compute_hazard_raw

        ppcf = self.results['PPCF'][frame_id]
        ball_drive_loss = self.results['BALL_DRIVE_RISK'][frame_id]['Ball drive risk']
        action_selection_pass_probability = self.results['ACTION_SELECTION'][frame_id]['Pass probability']
        hazard_h, drive_risk, pass_failure_expected, rppcf = compute_hazard_raw(
            ball_drive_loss=ball_drive_loss,
            pass_selection_probabilities=ppcf['Pass selection probabilities'],
            pass_success_probability=ppcf['Pass probability'],
            action_selection_pass_probability=action_selection_pass_probability,
        )
        self.results['HAZARD'][frame_id] = {
            'Hazard': hazard_h,
            'Drive risk': drive_risk,
            'Pass failure expected': pass_failure_expected,
            'rppcf': rppcf,
        }
        return hazard_h

    def _resolve_xt_grid(self, supplied):
        """Resolve a finite xT grid; fitted feature builders retain their saved surface."""
        import numpy as np

        if supplied is None:
            supplied = self._action_model().xt_grid
        grid = np.asarray(supplied, dtype=float)
        if grid.ndim != 2 or 0 in grid.shape or (not np.isfinite(grid).all()) or (grid < 0).any():
            raise ValueError('xT_grid must be a finite nonnegative (n_x, n_y) array.')
        return grid

    def compute_bcos(self, frame_id, r_max=40.0, step=1.0, physics_args=None):
        """Compute/store the optional BCOS map and area for this frame.

        Uses the installed ball-drive dispersion unless an explicit model or
        physical override was supplied. Default area is based on 1m cells.
        Recomputing invalidates action choice, h, residual risk and xLoss.
        """
        from PRESS.extras import BCOS

        bcos = BCOS(self.tracking_data, ball_drive_model=self.ball_drive_model)
        result = bcos.compute_bcos(frame_id, r_max=r_max, step=step, physics_args=physics_args)
        self.results['BCOS'][frame_id] = result
        self._invalidate(frame_id, 'ACTION_SELECTION', 'HAZARD')
        return self.get_bcos_result(frame_id)

    def compute_residual_risk(self, frame_id, *, time_since_start=None, model=None):
        """Compute terminal individual loss conditional on a successful 1s carry.

        Requires the current frame's action-selection output. It reuses that
        state's feature record and the matching action model to compute
        P(carry). Elapsed time is obtained from annotations at self.fps, or
        supplied explicitly in seconds; it is never replaced with zero.
        No future state, remaining time or observed action outcome is used.
        Returns native P(final loss | current state, successful carry).
        """
        import numpy as np
        from PRESS.model import ResidualRiskModel

        if frame_id not in self.results['ACTION_SELECTION']:
            raise ValueError('Compute action selection before residual risk.')
        action = self.results['ACTION_SELECTION'][frame_id]
        record = action.get('Features')
        if record is None:
            raise ValueError('Residual risk requires a current-model action feature record.')
        elapsed = self.get_elapsed_time(frame_id) if time_since_start is None else time_since_start
        if not np.isscalar(elapsed) or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('time_since_start must be finite and nonnegative seconds.')
        fitted = model if model is not None else self.residual_risk_model
        fitted = (
            ResidualRiskModel.pretrained()
            if fitted is None
            else self._load_model(fitted, ResidualRiskModel)
        )
        action_model = self._action_model()
        if action.get('model_id') != action_model.model_id:
            raise ValueError('Stored action features use a different action-selection model.')
        features = fitted.features_from_action_record(
            record,
            time_since_start=float(elapsed),
            action_model=action_model,
        )
        value = float(fitted.predict_features(features)[0])
        self.results['RESIDUAL_RISK'][frame_id] = {
            'Residual loss probability': value,
            'params': {
                'time_since_start': float(elapsed),
                'model_id': fitted.model_id,
                'action_model_id': action_model.model_id,
            },
        }
        self._invalidate(frame_id, 'XLOSS')
        return value

    def compute_xloss(self, frame_id, *, time_since_start=None, model=None):
        """Compose terminal individual-possession loss, including residual risk.

        Requires pass risk, destination choice and action choice. If missing,
        residual risk is computed from the current stored features; explicit
        time/model overrides recompute it. A completed pass terminates the
        individual possession without a loss. Returns xLoss as a float.
        """
        from PRESS.extras.residual_risk import compose_individual_xloss

        hazard = self.compute_hazard(frame_id)
        if (
            frame_id not in self.results['RESIDUAL_RISK']
            or time_since_start is not None
            or model is not None
        ):
            self.compute_residual_risk(frame_id, time_since_start=time_since_start, model=model)
        residual = self.results['RESIDUAL_RISK'][frame_id]
        action = self.results['ACTION_SELECTION'][frame_id]
        local = self.results['HAZARD'][frame_id]
        value = compose_individual_xloss(
            p_pass=action['Pass probability'],
            drive_loss=local['Drive risk'],
            pass_failure=local['Pass failure expected'],
            residual_loss=residual['Residual loss probability'],
        )
        self.results['XLOSS'][frame_id] = {
            'xLoss': value,
            'Hazard': hazard,
            'Residual loss probability': residual['Residual loss probability'],
            'params': dict(residual['params']),
        }
        return value

    def compute_player_possession(
        self,
        player_possession_id,
        xT_grid=None,
        *,
        frame_ids=None,
        include_snapshots=True,
        **frame_kwargs,
    ):
        """Evaluate an individual interval using the same frame-level model.

        Default sampling uses its annotated canonical checkpoints plus optional
        snapshots. Explicit frame_ids can select any usable state inside the
        interval; they do not change the one-second action definition.
        """
        import numpy as np
        import pandas as pd

        interval = self.get_player_possession(player_possession_id)
        valid = self.get_player_frames(player_possession_id, valid_only=True).frame
        checkpoints = None
        if self.actions_data is not None:
            checkpoints = self.get_checkpoints(eligible_only=False, include_snapshots=True)
            if 'possession_id' in checkpoints:
                checkpoints = checkpoints.loc[checkpoints.possession_id.eq(player_possession_id)]
            else:
                checkpoints = checkpoints.loc[checkpoints.frame_id.between(interval.frame_start, interval.frame_end)]
        if frame_ids is None:
            if checkpoints is None:
                raise ValueError('Supply actions_data or explicit frame_ids to sample this possession.')
            selected = checkpoints[checkpoints.eligible]
            if not include_snapshots:
                selected = selected[selected.is_canonical]
            frame_ids = selected.frame_id.to_numpy()
        frames = np.asarray(frame_ids)
        if (
            frames.ndim != 1
            or not frames.size
            or (not np.issubdtype(frames.dtype, np.number))
            or np.issubdtype(frames.dtype, np.complexfloating)
            or (not np.isfinite(frames).all())
            or (frames != np.floor(frames)).any()
            or (len(np.unique(frames)) != len(frames))
            or (not np.isin(frames, valid).all())
        ):
            raise ValueError(
                'frame_ids must be unique integer usable frames inside the '
                'individual possession.'
            )
        kinds = {} if checkpoints is None else dict(zip(
            checkpoints.frame_id,
            checkpoints.checkpoint_kind,
        ))
        rows = []
        for frame in np.sort(frames):
            result = self.compute_frame(int(frame), xT_grid, **frame_kwargs)
            row = {
                'player_possession_id': player_possession_id,
                'frame_id': int(frame),
                'checkpoint_kind': kinds.get(frame, 'custom'),
                'is_canonical': kinds.get(frame) == 'canonical',
                'p_pass': result['ACTION_SELECTION']['Pass probability'],
                **result['HAZARD'],
            }
            if 'XLOSS' in result:
                row['xLoss'] = result['XLOSS']['xLoss']
                row['residual_loss'] = result['RESIDUAL_RISK']['Residual loss probability']
            rows.append(row)
        return pd.DataFrame(rows)
