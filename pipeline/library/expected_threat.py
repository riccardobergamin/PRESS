"""Fixed 10×6 xT, estimated exclusively on the requested fit matches."""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def prepare_skillcorner_events(df, pitch_length=105, pitch_width=68, n_x=10, n_y=6):
    """
    Extract carries, successful passes, shots and possession
    losses from SkillCorner dynamic events.

    Coordinate convention:
        x: -pitch_length/2 -> +pitch_length/2
        y: -pitch_width/2  -> +pitch_width/2

    No attacking-direction normalization is performed.
    """
    df = df.copy()
    carries = df[(df['event_type'] == 'player_possession') & df['is_player_possession_start_matched'].fillna(False) & df['is_player_possession_end_matched'].fillna(False)].copy()
    carries = carries[['team_id', 'x_start', 'y_start', 'x_end', 'y_end']].copy()
    passes = df[(df['end_type'] == 'pass') & df['is_player_possession_end_matched'].fillna(False) & df['is_pass_reception_matched'].fillna(False) & (df['pass_outcome'] == 'successful')].copy()
    passes = passes[['team_id', 'x_end', 'y_end', 'player_targeted_x_reception', 'player_targeted_y_reception']].copy()
    passes = passes.rename(columns={'x_end': 'x_start', 'y_end': 'y_start', 'player_targeted_x_reception': 'x_end', 'player_targeted_y_reception': 'y_end'})
    shots = df[(df['end_type'] == 'shot') & df['is_player_possession_end_matched'].fillna(False)].copy()
    shots = shots[['team_id', 'x_end', 'y_end', 'game_interruption_after']].copy()
    shots = shots.rename(columns={'x_end': 'x_shot', 'y_end': 'y_shot'})
    shots['goal'] = shots['game_interruption_after'] == 'goal_for'
    losses = df[(df['end_type'].isin(['possession_loss', 'clearance']) | (df['end_type'] == 'pass') & (df['pass_outcome'] != 'successful')) & df['is_player_possession_end_matched'].fillna(False)].copy()
    losses = losses[['team_id', 'x_end', 'y_end']].copy()
    losses = losses.rename(columns={'x_end': 'x_loss', 'y_end': 'y_loss'})
    carries = carries.dropna(subset=['x_start', 'y_start', 'x_end', 'y_end'])
    passes = passes.dropna(subset=['x_start', 'y_start', 'x_end', 'y_end'])
    shots = shots.dropna(subset=['x_shot', 'y_shot'])
    losses = losses.dropna(subset=['x_loss', 'y_loss'])

    def add_transition_bins(data, x_start_col, y_start_col, x_end_col, y_end_col):
        data = data.copy()
        data['x_bin_start'] = np.floor((data[x_start_col] + pitch_length / 2) / pitch_length * n_x).astype(int)
        data['y_bin_start'] = np.floor((data[y_start_col] + pitch_width / 2) / pitch_width * n_y).astype(int)
        data['x_bin_end'] = np.floor((data[x_end_col] + pitch_length / 2) / pitch_length * n_x).astype(int)
        data['y_bin_end'] = np.floor((data[y_end_col] + pitch_width / 2) / pitch_width * n_y).astype(int)
        valid = data['x_bin_start'].between(0,
            n_x - 1) & data['y_bin_start'].between(0,
            n_y - 1) & data['x_bin_end'].between(0,
            n_x - 1) & data['y_bin_end'].between(0,
            n_y - 1)
        return data.loc[valid].copy()
    carries = add_transition_bins(carries, 'x_start', 'y_start', 'x_end', 'y_end')
    passes = add_transition_bins(passes, 'x_start', 'y_start', 'x_end', 'y_end')
    shots['x_bin'] = np.floor((shots['x_shot'] + pitch_length / 2) / pitch_length * n_x).astype(int)
    shots['y_bin'] = np.floor((shots['y_shot'] + pitch_width / 2) / pitch_width * n_y).astype(int)
    shots = shots[shots['x_bin'].between(0, n_x - 1) & shots['y_bin'].between(0, n_y - 1)].copy()
    losses['x_bin'] = np.floor((losses['x_loss'] + pitch_length / 2) / pitch_length * n_x).astype(int)
    losses['y_bin'] = np.floor((losses['y_loss'] + pitch_width / 2) / pitch_width * n_y).astype(int)
    losses = losses[losses['x_bin'].between(0, n_x - 1) & losses['y_bin'].between(0, n_y - 1)].copy()
    return (carries, passes, shots, losses)


def cell(x, y, nx, ny):
    return np.clip(np.floor((np.asarray(x) + 52.5) / 105 * nx).astype(int),
        0,
        nx - 1) * ny + np.clip(np.floor((np.asarray(y) + 34) / 68 * ny).astype(int),
        0,
        ny - 1)


def rebin(frame, kind, nx, ny):
    result = frame.copy()
    if kind in {'passes', 'carries'}:
        result['origin'] = cell(frame.x_start, frame.y_start, nx, ny)
        result['target'] = cell(frame.x_end, frame.y_end, nx, ny)
    else:
        suffix = 'shot' if kind == 'shots' else 'loss'
        result['origin'] = cell(frame['x_' + suffix], frame['y_' + suffix], nx, ny)
    return result


def transition(frame, nx, ny, bandwidth):
    counts = np.zeros((nx * ny, nx, ny))
    np.add.at(counts, (frame.origin, frame.target // ny, frame.target % ny), 1.0)
    unseen = counts.sum(axis=(1, 2)) == 0
    if bandwidth:
        counts = gaussian_filter(counts,
            sigma=(0, bandwidth / (105 / nx), bandwidth / (68 / ny)),
            mode='nearest')
    counts += 1e-08
    result = counts.reshape(nx * ny, nx * ny)
    result /= result.sum(axis=1, keepdims=True)
    result[unseen] = 1 / (nx * ny)
    return result


def goal_probability(frame, nx, ny, bandwidth):
    shots = np.bincount(frame.origin, minlength=nx * ny).reshape(nx, ny).astype(float)
    goals = np.bincount(frame.loc[frame.goal, 'origin'], minlength=nx * ny).reshape(nx, ny).astype(float)
    if bandwidth:
        sigma = (bandwidth / (105 / nx), bandwidth / (68 / ny))
        shots = gaussian_filter(shots, sigma=sigma, mode='nearest')
        goals = gaussian_filter(goals, sigma=sigma, mode='nearest')
    return (goals / (shots + 1e-08)).ravel()


def xt_model(tables, nx, ny, sigmas):
    counts = {k: np.bincount(v.origin, minlength=nx * ny).astype(float) for k, v in tables.items()}
    total = sum(counts.values())
    probabilities = {k: np.divide(v, total, out=np.zeros_like(total), where=total > 0) for k,
        v in counts.items()}
    tp = transition(tables['passes'], nx, ny, sigmas['passes'])
    tc = transition(tables['carries'], nx, ny, sigmas['carries'])
    pg = goal_probability(tables['shots'], nx, ny, sigmas['shots'])
    operator = probabilities['passes'][:, None] * tp + probabilities['carries'][:, None] * tc
    reward = probabilities['shots'] * pg
    value = np.zeros(nx * ny)
    for iteration in range(10000):
        new = reward + operator @ value
        delta = np.max(abs(new - value))
        value = new
        if delta < 1e-12:
            break
    assert delta < 1e-12 and np.isfinite(value).all() and (value.min() >= 0) and (value.max() <= 1)
    return dict(xt=value.reshape(nx, ny),
        p_goal=pg.reshape(nx, ny),
        p_pass=probabilities['passes'].reshape(nx, ny),
        p_carry=probabilities['carries'].reshape(nx, ny),
        p_shot=probabilities['shots'].reshape(nx, ny),
        p_loss=probabilities['losses'].reshape(nx, ny),
        T_pass=tp,
        T_carry=tc,
        iterations=iteration + 1,
        bellman_residual=float(np.max(abs(value - reward - operator @ value))))


def prepare(base, matches):
    from pathlib import Path
    destination = Path(base) / 'xt'
    destination.mkdir(parents=True, exist_ok=False)
    tables = {name: [] for name in ['carries', 'passes', 'shots', 'losses']}
    for match in matches:
        events = pd.read_parquet(Path(base) / 'matches' / match / 'events.parquet')
        for name, frame in zip(tables, prepare_skillcorner_events(events)):
            frame['match_id'] = str(match)
            tables[name].append(frame)
    for name, parts in tables.items():
        pd.concat(parts, ignore_index=True).to_parquet(destination / (name + '.parquet'), index=False)


def fit(base, matches):
    from pathlib import Path
    tables = {}
    for name in ['carries', 'passes', 'shots', 'losses']:
        frame = pd.read_parquet(Path(base) / 'xt' / (name + '.parquet'))
        frame = frame.loc[frame.match_id.astype(str).isin(matches)]
        tables[name] = rebin(frame, name, 10, 6)
    return xt_model(tables, 10, 6, dict(passes=8.0, carries=5.0, shots=5.0))['xt']
