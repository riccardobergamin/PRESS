import numpy as np
import pandas as pd
import jax.numpy as jnp
from .possession_selection import is_matched, pass_receipt_link_reason


def extract_pass_features(possession, tracking_data):
    """
    Extract ball carrier and receiver states for completed passes.

    The function identifies passes with a valid target receiver,
    retrieves the ball carrier position and velocity at the pass
    frame, and retrieves the receiver position at the reception frame.

    Parameters
    ----------
    possession : pandas.DataFrame
        Possession-level event data containing pass information.

    tracking_data : pandas.DataFrame
        Tracking data containing player positions and velocities.

    Returns
    -------
    pandas.DataFrame
        Pass dataframe enriched with:
            passer_x
            passer_y
            passer_vx
            passer_vy
            receiver_x
            receiver_y
    """
    passes = possession[(possession['end_type'] == 'pass') & possession['target_pass'].notna()].copy()
    passer_x = []
    passer_y = []
    passer_vx = []
    passer_vy = []
    receiver_x = []
    receiver_y = []
    for _, row in passes.iterrows():
        passer = row['player_id']
        receiver = row['target_pass']
        pass_frame = tracking_data[tracking_data['frame'] == row['frame_end']]
        receipt_frame = tracking_data[tracking_data['frame'] == row['frame_receipt']]
        if pass_frame.empty or receipt_frame.empty:
            passer_x.append(np.nan)
            passer_y.append(np.nan)
            passer_vx.append(np.nan)
            passer_vy.append(np.nan)
            receiver_x.append(np.nan)
            receiver_y.append(np.nan)
            continue
        passer_state = pass_frame.iloc[0]
        receiver_state = receipt_frame.iloc[0]
        passer_x.append(passer_state[f'{passer}_x'])
        passer_y.append(passer_state[f'{passer}_y'])
        passer_vx.append(passer_state[f'{passer}_vx'])
        passer_vy.append(passer_state[f'{passer}_vy'])
        receiver_x.append(receiver_state[f'{receiver}_x'])
        receiver_y.append(receiver_state[f'{receiver}_y'])
    passes['passer_x'] = passer_x
    passes['passer_y'] = passer_y
    passes['passer_vx'] = passer_vx
    passes['passer_vy'] = passer_vy
    passes['receiver_x'] = receiver_x
    passes['receiver_y'] = receiver_y
    return passes


def prepare_transition_input(passes, successful_only=True):
    """
    Convert pass dataframe into arrays required by
    the transition model.

    Parameters
    ----------
    passes : pandas.DataFrame
        Pass dataframe containing player states.

    successful_only : bool
        If True, keep only successful passes.

    Returns
    -------
    x_bc : jax.Array, shape (N,2)
        Ball carrier positions.

    v_bc : jax.Array, shape (N,2)
        Ball carrier velocity vectors.

    r_target : jax.Array, shape (N,2)
        Receiver target positions.
    """
    if successful_only:
        passes = passes[passes['pass_outcome'] == 'successful']
    passes = passes.dropna(subset=['passer_x', 'passer_y', 'passer_vx', 'passer_vy', 'receiver_x', 'receiver_y'])
    x_bc = jnp.array(passes[['passer_x', 'passer_y']].values)
    v_bc = jnp.array(passes[['passer_vx', 'passer_vy']].values)
    r_target = jnp.array(passes[['receiver_x', 'receiver_y']].values)
    return (x_bc, v_bc, r_target)


def build_pass_dataframe(x_bc, v_bc, r_target, match_id):
    """
    Build a dataframe containing geometric features
    describing each pass.

    Parameters
    ----------
    x_bc : array-like, shape (N,2)
        Ball carrier positions.

    v_bc : array-like, shape (N,2)
        Ball carrier velocity vectors.

    r_target : array-like, shape (N,2)
        Receiver positions.

    match_id : scalar or array-like
        Match identifier.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing:

            match_id
                Match identifier.

            distance
                Euclidean distance between ball carrier
                and receiver.

            speed
                Ball carrier speed magnitude.

            theta
                Signed angle between velocity direction
                and target direction in radians.
                Range: [-pi, pi].
    """
    x_bc = np.asarray(x_bc)
    v_bc = np.asarray(v_bc)
    r_target = np.asarray(r_target)
    delta = r_target - x_bc
    distance = np.linalg.norm(delta, axis=1)
    speed = np.linalg.norm(v_bc, axis=1)
    dot = np.sum(v_bc * delta, axis=1)
    cross = v_bc[:, 0] * delta[:, 1] - v_bc[:, 1] * delta[:, 0]
    theta = np.arctan2(cross, dot)
    return pd.DataFrame({'match_id': match_id, 'distance': distance, 'speed': speed, 'theta': theta})


def to_absolute_coordinates(x, y, attacking_side):
    """
    Convert SkillCorner attacking-perspective coordinates
    to absolute tracking coordinates.

    SkillCorner:
        left_to_right  -> unchanged
        right_to_left  -> rotate 180 degrees around (0, 0)
    """
    if attacking_side == 'left_to_right':
        return (x, y)
    if attacking_side == 'right_to_left':
        return (-x, -y)
    raise ValueError(f'Unknown attacking_side: {attacking_side}')


def reconstruct_pass_target(x_end, y_end, pass_distance, pass_angle, attacking_side):
    """
    Reconstruct pass reception position.

    x_end and y_end are expressed in SkillCorner
    attacking-perspective coordinates.

    pass_angle is the signed angle, in degrees,
    relative to the direction of attack.

    The target is reconstructed in SkillCorner
    coordinates and then converted to absolute
    tracking coordinates.
    """
    angle = np.deg2rad(pass_angle)
    x_target = x_end + pass_distance * np.cos(angle)
    y_target = y_end + pass_distance * np.sin(angle)
    return to_absolute_coordinates(x_target, y_target, attacking_side)


def extract_passes(events, fps=10):
    """
    Extract valid pass events.

    A pass is kept if:

        - event_type == "player_possession"
        - end_type == "pass"
        - is_player_possession_end_matched == True
        - the immediate next player_possession has its start matched
        - the passer's possession start and the next possession's end
          are not required to be matched

    Successful passes
    -----------------
    If reception is matched:
        use player_targeted_x_reception /
        player_targeted_y_reception.

    Otherwise:
        reconstruct the target using
        x_end, y_end, pass_distance and pass_angle.

    Unsuccessful passes
    -------------------
    The first subsequent player_possession must start with a
    valid interception/recovery type.

    The target is the observed interception/recovery location from
    the next possession's x_start / y_start, converted to tracking
    coordinates. It does not identify the passer's intended destination.
    The subsequent start must be explicitly matched.

    frame_reception is always the frame_start of the first
    subsequent player_possession.
    """
    possessions = events[events['event_type'].eq('player_possession')].copy()
    possessions = possessions.sort_values([column for column in ('match_id', 'period', 'frame_start') if column in possessions.columns],
        kind='stable').reset_index(drop=True)
    passes = []
    valid_interception_types = {'pass_interception',
        'free_kick_interception',
        'corner_interception',
        'goal_kick_interception',
        'recovery'}
    pass_indices = possessions.index[possessions['end_type'].eq('pass')]
    for idx in pass_indices:
        row = possessions.loc[idx]
        x_receiver = y_receiver = np.nan
        if not is_matched(row['is_player_possession_end_matched']):
            continue
        subsequent_events = possessions.loc[possessions.index > idx]
        if subsequent_events.empty:
            continue
        next_event = subsequent_events.iloc[0]
        if any((pd.isna(row[column]) or pd.isna(next_event[column]) or row[column] != next_event[column] for column in ('match_id', 'period') if column in possessions.columns)):
            continue
        if not is_matched(next_event['is_player_possession_start_matched']):
            continue
        if pass_receipt_link_reason(row, next_event):
            continue
        frame_end = int(row['frame_end'])
        frame_reception = int(next_event['frame_start'])
        gap_frames = frame_reception - frame_end
        outcome = str(row['pass_outcome']).lower()
        if outcome == 'successful':
            if is_matched(row['is_pass_reception_matched']):
                x_target = row['player_targeted_x_reception']
                y_target = row['player_targeted_y_reception']
                x_receiver = row['player_targeted_x_reception']
                y_receiver = row['player_targeted_y_reception']
                if pd.isna(x_target) or pd.isna(y_target):
                    continue
                x_target, y_target = to_absolute_coordinates(float(x_target),
                    float(y_target),
                    row['attacking_side'])
                x_receiver, y_receiver = (x_target, y_target)
            else:
                if pd.isna(row['x_end']) or pd.isna(row['y_end']) or pd.isna(row['pass_distance']) or pd.isna(row['pass_angle']):
                    continue
                x_target, y_target = reconstruct_pass_target(x_end=float(row['x_end']),
                    y_end=float(row['y_end']),
                    pass_distance=float(row['pass_distance']),
                    pass_angle=float(row['pass_angle']),
                    attacking_side=row['attacking_side'])
            passes.append({'frame_start': row['frame_start'], 'frame_end': row['frame_end'], 'frame_reception': frame_reception, 'pass_duration': gap_frames / fps, 'team_id': row['team_id'], 'player_id': row['player_id'], 'receiver_id': row['player_targeted_id'], 'attacking_side': row['attacking_side'], 'x_target': float(x_target), 'y_target': float(y_target), 'x_receiver': float(x_receiver), 'y_receiver': float(y_receiver), 'outcome': 1, 'interception_type': None})
            continue
        if outcome != 'unsuccessful':
            continue
        next_possession = next_event
        interception_type = next_possession['start_type']
        if interception_type not in valid_interception_types:
            continue
        x_target = next_possession['x_start']
        y_target = next_possession['y_start']
        if pd.isna(x_target) or pd.isna(y_target):
            continue
        x_target, y_target = to_absolute_coordinates(float(x_target),
            float(y_target),
            next_possession['attacking_side'])
        passes.append({'frame_start': row['frame_start'], 'frame_end': row['frame_end'], 'frame_reception': frame_reception, 'pass_duration': gap_frames / fps, 'team_id': row['team_id'], 'player_id': row['player_id'], 'receiver_id': row['player_targeted_id'], 'attacking_side': row['attacking_side'], 'x_target': float(x_target), 'y_target': float(y_target), 'x_receiver': float(x_receiver), 'y_receiver': float(y_receiver), 'outcome': 0, 'interception_type': interception_type})
    return pd.DataFrame(passes,
        columns=['frame_start', 'frame_end', 'frame_reception', 'pass_duration', 'team_id', 'player_id', 'receiver_id', 'attacking_side', 'x_target', 'y_target', 'x_receiver', 'y_receiver', 'outcome', 'interception_type'])
