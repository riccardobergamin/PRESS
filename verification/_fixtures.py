"""Deterministic artificial inputs used only by isolated unit tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_test_match():
    """Return fresh tracking, team intervals, actions, and an (n_x, n_y) xT grid.

    All values are synthetic. Tracking has 22 players at 10 Hz, two team
    possessions and a carrier change inside the first team possession.
    The xT grid is illustrative, not learned from SkillCorner events.
    """
    home = np.array([
        [-48, 0], [-30, -20], [-30, -7], [-30, 7], [-30, 20],
        [-10, -20], [-10, -7], [-10, 7], [-10, 20], [12, -10], [14, 12],
    ], dtype=float)
    away = np.array([
        [48, 0], [30, -20], [30, -7], [30, 7], [30, 20],
        [5, -20], [5, -7], [5, 7], [5, 20], [-16, -10], [-18, 12],
    ], dtype=float)
    rows = []
    for frame in range(40):
        carrier = "home_7" if frame < 10 else "home_8" if frame < 30 else "away_7"
        row = {"frame": frame, "timestamp": frame / 10, "period": 1,
               "player_possession": carrier,
               "attacking_side": "left_to_right" if frame < 30 else "right_to_left",
               "match_clock_min": frame / 600, "score_difference_prefix": 0}
        for side, positions in (("home", home), ("away", away)):
            for number, (x, y) in enumerate(positions, 1):
                vx = 0.4 if side == "home" else -0.2
                prefix = f"{side}_{number}"
                row.update({f"{prefix}_x": x + vx * frame / 10,
                            f"{prefix}_y": y, f"{prefix}_vx": vx,
                            f"{prefix}_vy": 0.0, f"{prefix}_v": abs(vx)})
        row["ball_x"] = row[f"{carrier}_x"]
        row["ball_y"] = row[f"{carrier}_y"]
        rows.append(row)
    possessions = pd.DataFrame([
        {"possession_id": 1, "team_id": "home", "frame_start": 0, "frame_end": 29, "lost": True},
        {"possession_id": 2, "team_id": "away", "frame_start": 30, "frame_end": 39, "lost": False},
    ])
    actions = pd.DataFrame([
        {"frame_start": f, "frame_end": min(f + 9, 39), "action": "ball_drive",
         "start_type": "pass_reception" if f < 30 else "recovery",
         "previous_action": "pass_reception" if f in (0, 10) else "ball_drive",
         "time_since_start": 1.0 if f == 20 else 0.0}
        for f in (0, 10, 20, 30)
    ])
    xt = np.repeat(np.linspace(0.001, 0.12, 16)[:, None], 12, axis=1)
    return pd.DataFrame(rows), possessions, actions, xt


def write_skillcorner_match(directory, *, pitch_length=105., pitch_width=68.):
    """Small original-format JSON/JSONL/CSV fixture for loader unit checks."""
    import json
    from pathlib import Path
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    match_id=directory.name
    tracking,*_=load_test_match()
    identities={f'{side}_{i}':offset+i for side,offset in [('home',100),('away',200)] for i in range(1,12)}
    players=[dict(id=value,team_id=1 if alias.startswith('home') else 2,
                  number=int(alias.split('_')[1]),short_name=alias,
                  player_role={'acronym':'GK' if alias.endswith('_1') else 'CM'})
             for alias,value in identities.items()]
    metadata=dict(players=players,home_team={'id':1},away_team={'id':2},
                  pitch_length=pitch_length,pitch_width=pitch_width)
    (directory/f'{match_id}_match.json').write_text(json.dumps(metadata))
    with (directory/f'{match_id}_tracking_extrapolated.jsonl').open('w') as stream:
        for row in tracking.itertuples():
            value=dict(frame=row.frame,timestamp=str(pd.to_timedelta(row.frame/10,unit='s')),
                period=row.period,ball_data=dict(x=row.ball_x,y=row.ball_y,z=0,is_detected=True),
                possession=dict(player_id=identities[row.player_possession],group='home' if row.frame<30 else 'away'),
                player_data=[dict(player_id=identifier,x=getattr(row,alias+'_x'),
                                 y=getattr(row,alias+'_y'),is_detected=True)
                             for alias,identifier in identities.items()])
            stream.write(json.dumps(value)+'\n')
    rows=[]
    specs=[(0,9,'home_7','pass','successful','home_8','pass_reception'),
           (10,29,'home_8','pass','unsuccessful',None,'pass_reception'),
           (30,39,'away_7','possession_loss',None,None,'pass_interception')]
    for i,(start,end,actor,ending,outcome,receiver,origin) in enumerate(specs):
        rows.append(dict(event_id=str(i+1),match_id=match_id,event_type='player_possession',
            player_id=identities[actor],team_id=1 if actor.startswith('home') else 2,
            frame_start=start,frame_end=end,time_start=f'00:00:{start/10:04.1f}',
            time_end=f'00:00:{end/10:04.1f}',period=1,start_type=origin,end_type=ending,
            pass_outcome=outcome,player_targeted_id=identities.get(receiver),
            game_interruption_before=None,game_interruption_after=None,
            attacking_side='left_to_right' if actor.startswith('home') else 'right_to_left',
            is_player_possession_start_matched=True,is_player_possession_end_matched=True,
            is_previous_pass_matched=True,team_score=0,opponent_team_score=0))
    pd.DataFrame(rows).to_csv(directory/f'{match_id}_dynamic_events.csv',index=False)
    return directory
