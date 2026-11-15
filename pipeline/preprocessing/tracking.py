"""Native tracking, offline kinematics, state annotation and canonical segmentation.

Task-specific outcome filters are applied by the corresponding task library."""
import numpy as np
import pandas as pd
FPS = 10
POLICY = 'one_second_actions_with_exposure_tails_v2'
SCHEMA_VERSION = 'press_base_v4'
TRAINING_COHORT_POLICY = 'outfield_carriers_no_shot_v1'


def annotate_training_cohort(table, players, *, canonical_players=False):
    """Flag carrier eligibility while retaining all intervals for inference.

    Only the carrier role is filtered: goalkeepers remain in tracking and in
    defensive/receiving states. Unknown roles fail.
    """
    result = table.copy()
    key = 'player_possession' if canonical_players else 'player_id'
    roles = players.set_index(key).is_gk
    result['carrier_is_goalkeeper'] = result.player_id.map(roles)
    if result.carrier_is_goalkeeper.isna().any():
        raise ValueError('Training carrier has no verified goalkeeper/outfield role.')
    result['carrier_is_goalkeeper'] = result.carrier_is_goalkeeper.astype(bool)
    shot = result.end_type.eq('shot')
    keeper = result.carrier_is_goalkeeper
    result['training_cohort_eligible'] = ~shot & ~keeper
    result['training_cohort_exclusion_reason'] = np.select([shot & keeper, shot, keeper],
        ['shot_ending;goalkeeper_carrier', 'shot_ending', 'goalkeeper_carrier'],
        default='')
    result['training_cohort_policy'] = TRAINING_COHORT_POLICY
    return result


from PRESS.io.kinematics import differentiate


from PRESS.io.skillcorner import add_kinematics, load_raw_tracking, annotate_tracking


def annotate_state_quality(tracking):
    """Audit usable carrier/state without removing extrapolated observations."""
    result = tracking.copy()
    carrier_ok = np.zeros(len(result), dtype=bool)
    counts = {side: np.zeros(len(result), dtype=int) for side in ('home', 'away')}
    for column in result.columns:
        if not (column.startswith(('home_', 'away_')) and column.endswith('_x')):
            continue
        label = column[:-2]
        finite = np.isfinite(result[[f'{label}_{axis}' for axis in ('x', 'y', 'vx', 'vy')]].to_numpy(dtype=float)).all(axis=1)
        counts[label.split('_')[0]] += finite
        carrier_ok |= result.player_possession.eq(label).to_numpy() & finite
    for side, count in counts.items():
        result[f'n_{side}_usable'] = count
    result['state_eligible'] = carrier_ok & result.attacking_side.isin(['left_to_right', 'right_to_left']) & (counts['home'] > 0) & (counts['away'] > 0)
    result['state_exclusion_reason'] = np.select([result.player_possession.isna(), ~carrier_ok, ~result.attacking_side.isin(['left_to_right', 'right_to_left']), (counts['home'] == 0) | (counts['away'] == 0)],
        ['no_selected_carrier', 'missing_carrier_kinematics', 'unknown_direction', 'empty_team_state'],
        default='')
    return result


def segment_possessions(intervals, *, fps=FPS, horizon_frames=10):
    """Apply the package's shared canonical segmentation without changing policy."""
    from PRESS.io.segmentation import canonical_actions
    return canonical_actions(intervals,fps=fps,horizon_frames=horizon_frames)


def drive_observations(actions):
    """Observed exposure for rate training; distinct from canonical choice rows.

    Failure time is interval-observed at 10Hz. A zero-frame failure is kept
    in the first 0.1s bin, explicitly flagged rather than dropped.
    """
    rows = actions.loc[actions.action.eq('ball_drive') | actions.snapshot_reason.eq('short_control_tail')].copy()
    rows['action'] = 'ball_drive'
    rows['observed_duration'] = rows.duration
    rows['loss_observed'] = rows.outcome.eq(0)
    rows['target_status'] = np.select([rows.loss_observed, ~rows.is_canonical],
        ['failure_interval', 'censored'],
        default='survived')
    rows['observation_upper'] = np.where(rows.loss_observed & rows.duration.eq(0), 1 / FPS, rows.duration)
    rows['observation_lower'] = np.where(rows.loss_observed,
        np.maximum(rows.observation_upper - 1 / FPS, 0),
        rows.observation_upper)
    rows['zero_frame_failure'] = rows.loss_observed & rows.duration.eq(0)
    rows['horizon_label_observed'] = rows.loss_observed | rows.duration.ge(1.0)
    rows['stop_reason'] = np.where(rows.loss_observed,
        'possession_loss',
        np.where(rows.is_canonical, 'one_second_completed', rows.end_type))
    rows['drive_observation_id'] = np.arange(1, len(rows) + 1)
    return rows


def pass_curvature(row, tracking):
    """Maximum lateral deviation (% of endpoint distance), requiring full coverage."""
    start, end = (int(row.frame_end), int(row.frame_reception))
    trajectory = tracking.loc[start:end, ['ball_x', 'ball_y']]
    if len(trajectory) != end - start + 1 or not np.isfinite(trajectory.to_numpy()).all():
        return np.nan
    points = trajectory.to_numpy(dtype=float)
    line = np.array([row.x_target, row.y_target]) - points[0]
    length = np.linalg.norm(line)
    if length == 0:
        return np.nan
    cross = np.abs(line[0] * (points[:, 1] - points[0, 1]) - line[1] * (points[:, 0] - points[0, 0]))
    return float(cross.max() / length ** 2 * 100)
