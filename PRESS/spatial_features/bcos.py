"""Final BCOS fields and the compatible effective-hazard formulation."""

from __future__ import annotations

import numpy as np


class BCOS:
    """Compute origin-exposed transport, protected waiting or the frozen feature."""

    def __init__(self, tracking_data, *, ball_drive_model=None):
        self.tracking_data = tracking_data
        self.ball_drive_model = ball_drive_model
        self.results = {}

    def compute_bcos(self, frame_id, r_max=40.0, step=1.0, physics_args=None, *, formulation="transport", state=None):
        """Compute accessibility with waiting interception by default.

        transport_reservoir protects the origin until route entry;
        effective_hazard retains the feature used by the fitted classifiers.
        Optional state is the same frame's PRESS.get_state() output; the
        match facade uses it to avoid extracting the same tracking row again.
        """
        if state is not None:
            observed = state.get('frame_df')
            if observed is None or len(observed) != 1 or observed.frame.iloc[0] != frame_id:
                raise ValueError('state must be get_state() output for the same frame_id.')
        if formulation == "effective_hazard":
            result = self._compute_effective_hazard(frame_id,r_max,step,physics_args,state=state)
            result['params']['formulation'] = formulation
        elif formulation in {"transport", "transport_reservoir"}:
            from PRESS.model.bcos import BCOSModel
            from PRESS.utils.state import get_pv_players
            state = get_pv_players(frame_id,self.tracking_data) if state is None else state
            configuration = dict(physics_args or {})
            time = configuration.pop('time', np.inf)
            if formulation == "transport_reservoir":
                from PRESS.model.bcos_reservoir import ReservoirBCOSModel
                model = ReservoirBCOSModel.from_ball_drive(self.ball_drive_model,**configuration)
            else:
                model = BCOSModel.from_ball_drive(self.ball_drive_model,**configuration)
            result = model.predict_field(state['x_bc'],state['v_bc'],state['r_defs'],state['v_defs'],
                r_max=r_max,step=step,pitch_length=state['pitch_length'],pitch_width=state['pitch_width'],time=time)
        else:
            raise ValueError("formulation must be 'transport', 'transport_reservoir' or 'effective_hazard'.")
        self.results[frame_id] = result
        return result

    def _compute_effective_hazard(self, frame_id, r_max=40.0, step=1.0, physics_args=None, *, state=None):
        """Return field and area, using 1m targets and 0.03125m paths by default.

        The probability array is 'Ball carrier open space'; 'BCOS area' is
        its sum multiplied by step squared. Use that area as the action feature.
        Defaults use the installed carry model's dispersion, shared TTI and
        exponential survival update. Explicit physics_args may override scales.
        """
        from PRESS.model.ball_drive import BallDriveModel
        from PRESS.model.bcos_effective import compute_bcos_state
        from PRESS.utils.state import get_pv_players

        state = get_pv_players(frame_id, self.tracking_data) if state is None else state
        model = self.ball_drive_model or BallDriveModel.pretrained()
        if isinstance(model,dict):
            model = BallDriveModel.from_parameters(model)
        elif not isinstance(model, BallDriveModel):
            model = BallDriveModel.from_json(model)
        a = model.to_dict()
        p = a['physics']
        args = dict(physics_args or {})
        cfg = [
            args.pop('s', float(np.exp(a['theta'][3]))),
            args.pop('v_bc_max', 6.6),
            args.pop('v_def_max', p['v_max']),
            args.pop('a_bc_max', 7.0),
            args.pop('a_def_max', p['a_linear']),
            args.pop('a_turn_bc', 7.0),
            args.pop('a_turn_def', p['a_turn']),
        ]
        if 'tol' in args:
            args.setdefault('stop_eps', args.pop('tol'))
        max_points = args.pop('MAX_POINTS', None)
        if (
            max_points is not None
            and max_points < int(np.ceil(r_max / args.get('dx', 0.03125))) + 2
        ):
            raise ValueError('MAX_POINTS cannot truncate the selected BCOS path.')
        result = compute_bcos_state(
            np.vstack([state['r_atts'], state['x_bc'], state['r_defs']]),
            np.vstack([state['v_atts'], state['v_bc'], state['v_defs']]),
            r_max=r_max,
            step=step,
            pitch_length=state['pitch_length'],
            pitch_width=state['pitch_width'],
            configuration=cfg,
            **args,
        )
        # Expose the resolved physical settings by name in both formulations.
        # Retain the numerical kernel's ordered configuration for compatibility.
        result['params'].update(zip(
            ['s', 'v_bc_max', 'v_def_max', 'a_bc_max', 'a_def_max', 'a_turn_bc', 'a_turn_def'],
            cfg,
        ))
        self.results[frame_id] = result
        return result

    def get_result(self, frame_id):
        """Return stored probability field, area and integration metadata."""
        if frame_id not in self.results:
            raise KeyError(f'No BCOS result stored for frame {frame_id!r}.')
        return self.results[frame_id]
