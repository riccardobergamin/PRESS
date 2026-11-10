"""10×6 xT: event counts -> component smoothing -> Bellman iteration.

The grid is fixed. The three smoothing bandwidths are selected again inside
the requested fitting matches, as in the accepted six-match and production
fits. They are not fixed to the all-ten-match production values.

Reading order after event preparation:
    rebin -> transition / goal_probability -> xt_model
    component_scores -> select_bandwidth -> fit_on_matches -> fit
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


BANDWIDTHS_M = [0., 3., 5., 8., 12., 18.]


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
    """T[i,j]: destination given one event type and one fixed origin i.

    `frame` contains either completed passes or within-possession displacements.
    Smooth destination counts only; the zero sigma leaves origins separate.
    Each row sums to one. This matrix contains no extra success probability.
    """
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
    """P(goal | shot,cell): smooth goal and shot counts, then take their ratio."""
    shots = np.bincount(frame.origin, minlength=nx * ny).reshape(nx, ny).astype(float)
    goals = np.bincount(frame.loc[frame.goal, 'origin'], minlength=nx * ny).reshape(nx, ny).astype(float)
    if bandwidth:
        sigma = (bandwidth / (105 / nx), bandwidth / (68 / ny))
        shots = gaussian_filter(shots, sigma=sigma, mode='nearest')
        goals = gaussian_filter(goals, sigma=sigma, mode='nearest')
    return (goals / (shots + 1e-08)).ravel()


def xt_model(tables, nx, ny, sigmas):
    """p_a[i]=n_a[i]/n_total[i]; xT=q+M@xT.

    p_pass counts completed passes, not successes divided by attempted passes.
    The carry table contains start-to-end displacements regardless of the
    terminal outcome. These event frequencies p_a are NOT smoothed.
    Only destination counts (T) and goal/shot counts use Gaussian smoothing.
    """
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


# Smoothing selection: use only the matches available for this xT fit.

def component_scores(validation, fitting, component, nx, ny, bandwidth):
    """Held-out score for one component estimated on `fitting` events.

    Pass/carry: -log(T[i,j] / cell_area), a destination-density score.
    Shots: Bernoulli CE of the goal outcome. Neither scores the full xT target.
    """
    if component == 'shots':
        p_goal = goal_probability(fitting, nx, ny, bandwidth)
        probability = np.clip(p_goal[validation.origin], 1e-6, 1 - 1e-6)
        goal = validation.goal.to_numpy(float)
        return -(goal * np.log(probability) + (1 - goal) * np.log1p(-probability))

    T = transition(fitting, nx, ny, bandwidth)
    probability = T[validation.origin, validation.target]
    cell_area = (105 / nx) * (68 / ny)
    return -np.log(probability / cell_area)


def select_bandwidth(events, component, fitting_matches, nx=10, ny=6):
    """Leave one fitting match out; choose the largest h within one paired SE.

    Returns the chosen bandwidth, per-match scores and the candidate comparison.
    The jackknife below removes match score groups; it does not refit a model.
    """
    records = []
    for held_match in sorted(fitting_matches):
        fitting = events.loc[events.match_id.isin(fitting_matches)
                             & events.match_id.ne(held_match)]
        validation = events.loc[events.match_id.eq(held_match)]
        if validation.empty:
            continue
        for bandwidth in BANDWIDTHS_M:
            losses = component_scores(validation, fitting, component, nx, ny, bandwidth)
            records.append(dict(match_id=held_match, bandwidth=bandwidth,
                                n=len(losses), nll=float(losses.mean())))

    scores = pd.DataFrame(records)

    # Pool held-out match scores with event counts as weights.
    pooled = scores.groupby('bandwidth', sort=True).apply(
        lambda group: np.average(group.nll, weights=group.n), include_groups=False,
    )
    best = float(pooled.idxmin())
    reference = scores.loc[scores.bandwidth.eq(best)].set_index('match_id')

    comparisons = []
    for bandwidth in BANDWIDTHS_M:
        candidate = scores.loc[scores.bandwidth.eq(bandwidth)]
        candidate = candidate.set_index('match_id').reindex(reference.index)
        weights = candidate.n.to_numpy(float)
        differences = (candidate.nll - reference.nll).to_numpy(float)

        # Delta = sum_m n_m*d_m / sum_m n_m.
        weighted_total = np.sum(weights * differences)
        delta = np.average(differences, weights=weights)

        # Delete-one-match Delta and its jackknife standard error.
        leave_one_out = (weighted_total - weights * differences) / (weights.sum() - weights)
        n_matches = len(weights)
        se = np.sqrt((n_matches - 1) / n_matches
                     * np.sum((leave_one_out - leave_one_out.mean()) ** 2))
        comparisons.append(dict(
            bandwidth=bandwidth, pooled_nll=float(pooled.loc[bandwidth]),
            delta_from_best=float(delta), paired_se=float(se),
            admitted=bool(delta <= se + 1e-12),
        ))

    comparison = pd.DataFrame(comparisons)
    selected = float(comparison.loc[comparison.admitted, 'bandwidth'].max())
    comparison['selected'] = comparison.bandwidth.eq(selected)
    return selected, scores, comparison


def fit_on_matches(event_tables, fitting_matches, nx=10, ny=6):
    """Select smoothing and fit xT on exactly the requested matches.

    Input: four event DataFrames with match_id and continuous coordinates.
    Output: all fitted components plus the actual smoothing choices and scores.
    Six-match feature fits and the ten-match production fit use the same steps.
    """
    fitting_matches = sorted(str(match) for match in fitting_matches)
    tables = {}
    for name, events in event_tables.items():
        events = events.copy()
        events['match_id'] = events.match_id.astype(str)
        events = events.loc[events.match_id.isin(fitting_matches)]
        tables[name] = rebin(events, name, nx, ny)

    bandwidths, tuning_scores, comparisons = {}, {}, {}
    for component in ['passes', 'carries', 'shots']:
        bandwidth, scores, comparison = select_bandwidth(
            tables[component], component, fitting_matches, nx, ny,
        )
        bandwidths[component] = bandwidth
        tuning_scores[component] = scores
        comparisons[component] = comparison

    fitted = xt_model(tables, nx, ny, bandwidths)
    fitted.update(
        bandwidth_m=bandwidths, fitting_matches=fitting_matches,
        tuning_scores=tuning_scores, bandwidth_comparisons=comparisons,
    )
    return fitted


def prepare(base, matches):
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
    """Load the four event tables and return the fitting-only 10×6 xT surface.

    Downstream callers keep receiving the same (10,6) array. The component fit
    remains inspectable through fit_on_matches; no installed model is replaced.
    """
    event_tables = {
        name: pd.read_parquet(Path(base) / 'xt' / (name + '.parquet'))
        for name in ['carries', 'passes', 'shots', 'losses']
    }
    return fit_on_matches(event_tables, matches)['xt']
