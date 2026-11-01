"""Optional BCOS field, aligned with the final action-choice feature."""
from __future__ import annotations
import numpy as np

class BCOS:
    """Compute BCOS with the verified conditional-hazard integration."""
    def __init__(self,tracking_data,*,ball_drive_model=None):
        self.tracking_data=tracking_data;self.ball_drive_model=ball_drive_model;self.results={}

    def compute_bcos(self,frame_id,r_max=40.,step=1.,physics_args=None):
        """Return field and area, using 1m targets and 0.03125m paths by default.

        The probability array is 'Ball carrier open space'; 'BCOS area' is
        its sum multiplied by step squared. Use that area as the action feature.
        Defaults use the installed carry model's dispersion, shared TTI and
        exponential survival update. Explicit physics_args may override scales.
        """
        from PRESS.model.ball_drive import BallDriveModel
        from PRESS.model.bcos_fine import compute_bcos_state
        from PRESS.utils.state import get_pv_players
        state=get_pv_players(frame_id,self.tracking_data)
        model=self.ball_drive_model or BallDriveModel.pretrained()
        if not isinstance(model,BallDriveModel):model=BallDriveModel.from_json(model)
        a=model.to_dict();p=a['physics'];args=dict(physics_args or {})
        cfg=[args.pop('s',float(np.exp(a['theta'][3]))),args.pop('v_bc_max',6.6),
             args.pop('v_def_max',p['v_max']),args.pop('a_bc_max',7.),args.pop('a_def_max',p['a_linear']),
             args.pop('a_turn_bc',7.),args.pop('a_turn_def',p['a_turn'])]
        if 'tol' in args:args.setdefault('stop_eps',args.pop('tol'))
        max_points=args.pop('MAX_POINTS',None)
        if max_points is not None and max_points<int(np.ceil(r_max/args.get('dx',.03125)))+2:
            raise ValueError('MAX_POINTS cannot truncate the selected BCOS path.')
        result=compute_bcos_state(np.vstack([state['r_atts'],state['x_bc'],state['r_defs']]),
            np.vstack([state['v_atts'],state['v_bc'],state['v_defs']]),r_max=r_max,step=step,
            pitch_length=state['pitch_length'],pitch_width=state['pitch_width'],configuration=cfg,**args)
        self.results[frame_id]=result;return result

    def get_result(self,frame_id):
        """Return stored probability field, area and integration metadata."""
        if frame_id not in self.results:raise KeyError(f'No BCOS result stored for frame {frame_id!r}.')
        return self.results[frame_id]
