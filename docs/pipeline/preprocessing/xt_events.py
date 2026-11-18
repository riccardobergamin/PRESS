"""Original xT event cohort; not the stricter physical-pass cohort."""
import numpy as np


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

