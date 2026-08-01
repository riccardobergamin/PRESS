reticulate::use_condaenv("databallpy_env")
import os
import numpy as np
from databallpy import get_game
from floodlight.io.dfl import read_event_data_xml

def segment_possessions(game):
    
    events_df = game.event_data.dropna(subset=['original_event_id'])
    tracking_df = game.tracking_data
    
    possessions = []
    tracking_indexed = tracking_df.set_index('frame')

    # Triggers iniziali: Passaggi completati o Ball Claiming
    triggers = events_df[
        ((events_df['original_event'] == "Pass") & (events_df['is_successful'] == True)) | 
        (events_df['original_event'] == 'BallClaiming')
    ].sort_values('tracking_frame').reset_index(drop=True)

    for i, start_event in triggers.iterrows():
        t_event = int(start_event['tracking_frame'])
        current_player = start_event['player_id']
        
        # 1. Identifichiamo il limite della ricerca (prossimo evento del ricevitore)
        if start_event['databallpy_event'] == 'pass':
            receiver_id = start_event['receiver_player_id']
            
            # Cerchiamo il primo evento successivo in cui il ricevitore è protagonista
            receiver_next_event = events_df[
                (events_df['tracking_frame'] > t_event) & 
                (events_df['player_id'] == receiver_id)
            ].sort_values('tracking_frame').first_valid_index()

            if receiver_next_event is None:
                continue
            
            limit_frame = int(events_df.loc[receiver_next_event, 'tracking_frame'])
            t_start = None

            # Ricerca del primo tocco nel tracking tra t_event e il suo prossimo evento
            for t_search in range(t_event + 1, limit_frame + 1):
                if t_search in tracking_indexed.index:
                    pos_row = tracking_indexed.loc[t_search, 'player_possession']
                    pos_values = [pos_row] if isinstance(pos_row, str) else (pos_row.values if hasattr(pos_row, 'values') else [])

                    if receiver_id in pos_values:
                        t_start = t_search
                        break
            
            # FALLBACK: Se non trovo possesso nel tracking, prendo il frame precedente all'evento finale
            if t_start is None:
                t_start = limit_frame - 1
            
            current_player = receiver_id
        else:
            # Se è BallClaiming, il possesso inizia all'evento stesso
            t_start = t_event

        # 2. Definizione FINE POSSESSO (resta coerente con la tua logica originale)
        future_events = events_df[
            (events_df['tracking_frame'] >= t_start) &
            # Cerchiamo l'evento che chiude il possesso (pass, tiro, fallo, ecc.)
            (
                (events_df['databallpy_event'].isin(['pass', 'shot'])) |
                (events_df['original_event'].isin(['fouled', 'ballClaimed', 'ballcontactSucceeded'])) |
                (events_df['is_defensive_clearance'] == True)
            )
        ].sort_values('tracking_frame')

        # Filtriamo per assicurarci che l'evento di chiusura sia del giocatore attuale o una perdita
        if future_events.empty:
            continue

        outcome_event = future_events.iloc[0]

        # Logica di salvataggio (Outcome)
        res = {
            'player': current_player,
            'start_frame': t_start,
            'end_frame': outcome_event['tracking_frame'],
            'outcome': outcome_event['databallpy_event'].upper() if outcome_event['databallpy_event'] in ['pass', 'shot'] else outcome_event['original_event'],
            'pass_result': outcome_event['result'] if outcome_event['databallpy_event'] == "pass" else 'N/A'
        }
        
        # Correzione label speciali
        if outcome_event['is_defensive_clearance']: res['outcome'] = 'LOST_BY_CLEARANCE'
        if outcome_event['original_event'] == 'fouled': res['outcome'] = 'FOULED_SUBITO'
        if outcome_event['original_event'] in ['ballClaimed', 'ballcontactSucceeded'] and outcome_event['player_id'] != current_player:
            res['outcome'] = 'LOST_BY_TACKLE'

        possessions.append(res)

    return pd.DataFrame(possessions)





# Define local directory paths and data file names
matches = {
    "J03WMX": "1. FC Köln vs. FC Bayern München",
    "J03WN1": "VfL Bochum 1848 vs. Bayer 04 Leverkusen"
}

path = "IDSEE/"
output_base = "output/"

match_id = "J03WMX"

pos_file = f"DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-{match_id}.xml"
info_file = f"DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-{match_id}.xml"
event_file = f"DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-{match_id}.xml"



game = get_game(
        tracking_data_provider="sportec",
        tracking_metadata_loc=path + info_file,
        tracking_data_loc=path + pos_file,
        event_data_loc=path + event_file,
        event_metadata_loc=path + info_file,
        event_data_provider="sportec"
)


# Aggregation lists to harmonize Sportec's native event tags
pass_list = ['Pass', 'Cross']
shot_list = ['SuccessfulShot', 'SavedShot', 'BlockedShot', 'ShotWide', 'ShotWoodWork']
tackle_list = ['ballClaimed', 'ballcontactSucceeded', 'BallClaiming', 'fouled']

# Set up logical conditions for numpy conditional mapping
conditions = [
    game.event_data['original_event'].isin(pass_list),
    game.event_data['original_event'].isin(shot_list),
    game.event_data['original_event'].isin(tackle_list)
]

# Target standardized labels aligned with databallpy conventions
choices = ['pass', 'shot', 'tackle']

# Apply mapping, falling back to pre-existing databallpy labels if unmatched
game.event_data['databallpy_event'] = np.select(
    conditions,
    choices,
    default=game.event_data['databallpy_event']
)

# Align discrete manual event timestamps with continuous, automatic tracking frames.
game.synchronise_tracking_and_event_data(
    n_batches="smart",
    verbose=True,
    offset=1.
)

# Appends possession columns to the tracking dataframe, identifying who controls the ball.
game.tracking_data.add_individual_player_possession()

player_ids = list(set([col.replace('_x', '').replace('_y', '') for col in game.tracking_data.columns if 'home' in col or 'away' in col]))


for player in player_ids:
    game.tracking_data.add_velocity(
        column_ids=player, 
        max_velocity=50.,
        filter_type="moving_average",
        window_length=12
    )

game.tracking_data.frame = game.tracking_data.frame - game.tracking_data.frame.iloc[0]

game.event_data['original_event_id'] = game.event_data['original_event_id'].astype(str).str.strip()

# add defensive cleareance

import pandas as pd
from kloppy import sportec

events, _, _ = read_event_data_xml(
        f"{path}{event_file}", f"{path}{info_file}"
  )
  
all_events_list = []
for half in ['firstHalf', 'secondHalf']:
  for team in ['Home', 'Away']:
    df_temp = events[half][team].events.copy()
    df_temp['half'] = half
    df_temp['team_side'] = team
    all_events_list.append(df_temp)

df_all_events = pd.concat(all_events_list, ignore_index=True)

OtherBallAction = df_all_events[df_all_events.eID == "OtherBallAction"]

def is_clearance(qualifier):
  if isinstance(qualifier, dict):
    return qualifier.get('DefensiveClearance') == 'true'
  if isinstance(qualifier, list):
    return any(
                isinstance(q, dict) and q.get('DefensiveClearance') == 'true'
                for q in qualifier
            )
    return False
      
OtherBallAction['is_defensive_clearance'] = OtherBallAction['qualifier'].apply(is_clearance)

game.event_data['datetime'] = pd.to_datetime(game.event_data['datetime'], format='mixed')

game.event_data = game.event_data.merge(
        OtherBallAction[['timestamp', 'is_defensive_clearance']],
        left_on='datetime',
        right_on='timestamp',
        how='left'
)

game.event_data['is_defensive_clearance'] = game.event_data['is_defensive_clearance'].fillna(False)
game.event_data = game.event_data.drop(columns=['timestamp'])


# add receiver
passes_kloppy = sportec.load_event(
    meta_data=path + info_file,
    event_data=path + event_file
).to_df().dropna(subset=['event_id'])

game.event_data['original_event_id'] = game.event_data['original_event_id'].astype(str).str.strip()
passes_kloppy['event_id'] = passes_kloppy['event_id'].astype(str).str.strip()

game.event_data = pd.merge(
        game.event_data.drop(columns=['event_id']),
        passes_kloppy.dropna(subset=['event_id'])[["event_id", "receiver_player_id", "result"]],
        left_on='original_event_id',
        right_on='event_id',
        how='left'
)

df_possession = segment_possessions(game)
    
df_possession.to_csv(os.path.join(match_dir, "possession.csv"), index=False)




