"""State-only pass-option fields shared by choice and downstream models."""
from pathlib import Path
import numpy as np
from PRESS.model import BallDriveModel, PassRiskModel, PassSelectionModel
from PRESS.model.pass_selection import build_choice_features, choice_feature_names
from PRESS.utils.state import compute_dynamic_pressure_lines
from pipeline.library.common import read_json
from pipeline.library.expected_threat import fit as fit_xt
from pipeline.library.states import folds


def upstream(paths, fold=None):
    if fold is None:
        bd = paths.task('ball_drive', 'production') / 'ball_drive_model.json'
        pp = paths.task('pass_probability', 'production') / 'pass_risk_model.json'
        matches = sorted((p.name for p in (paths.base / 'matches').iterdir() if p.is_dir()))
    else:
        bd = paths.task('ball_drive', 'training') / f'fold_{fold}' / 'six_fit_model.json'
        pp = paths.task('pass_probability', 'training') / f'fold_{fold}' / 'six_fit_model.json'
        matches = folds()[str(fold)]['fit_matches']
    drive, passing = BallDriveModel.from_json(bd), PassRiskModel.from_json(pp)
    if set(drive.to_dict()['training']['match_ids']) != set(matches):
        raise ValueError('Ball-drive model does not match the requested fit split')
    if set(passing.to_dict()['fit']['fit_matches']) != set(matches):
        raise ValueError('Pass-risk model does not match the requested fit split')
    return drive, passing, fit_xt(paths.base, matches)


def targets_for(state):
    length, width = (state['pitch_length'], state['pitch_width'])
    xx, yy = np.meshgrid(np.arange(-length / 2 + 0.5, length / 2, 1.0),
        np.arange(-width / 2 + 0.5, width / 2, 1.0),
        indexing='ij')
    return np.column_stack([xx.ravel(), yy.ravel()])


def options(state, drive, passing, xt, vocabulary, *, start_type, previous_action, clock, score):
    targets = targets_for(state)
    loss = float(drive.predict(r_defs=state['r_defs'], v_defs=state['v_defs'], x_bc=state['x_bc'], v_bc=state['v_bc'], physics_kwargs=dict(pitch_length=state['pitch_length'], pitch_width=state['pitch_width']))['loss_probability'])
    field = passing.predict_field(state['p_pos'], state['p_vel'], state['x_bc'], targets, np.arange(10), 10)
    direction = 1 if state['attacking_side'] == 'left_to_right' else -1
    lines = (compute_dynamic_pressure_lines(state['r_defs'], state['attacking_side']),
        compute_dynamic_pressure_lines(state['r_atts'], 'right_to_left' if direction == 1 else 'left_to_right'))
    matrix = build_choice_features(x_bc=state['x_bc'],
        v_bc=state['v_bc'],
        r_atts=state['r_atts'],
        r_defs=state['r_defs'],
        targets=targets,
        attacking_side=state['attacking_side'],
        target_pitch_control=field['raw'],
        ball_drive_prob=loss,
        xt_grid=xt,
        vocabulary=vocabulary,
        start_type_raw=start_type,
        previous_action_raw=previous_action,
        match_clock_min=clock,
        score_difference_prefix=score,
        pitch_length=state['pitch_length'],
        pitch_width=state['pitch_width'],
        pressure_lines=lines)
    from pipeline.library.pass_selection.membership import log_membership
    control = np.asarray(field['control'][:10]).T
    log_alpha, _ = log_membership(control, np.linalg.norm(state['p_pos'][:10], axis=-1) < 500)
    names = choice_feature_names(vocabulary)
    return dict(matrix=matrix,
        names=names,
        targets=targets,
        loss=loss,
        completion=field['calibrated'],
        raw_control=field['raw'],
        log_alpha=log_alpha,
        lines=lines)
