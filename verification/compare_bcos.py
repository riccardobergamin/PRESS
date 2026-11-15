"""Compare BCOS formulations on reproducibly stratified real match frames.

Usage: python verification/compare_bcos.py MATCH_DIR [...] --output OUTPUT_DIR
Each directory contains <match_id>_tracking.parquet. No model is fitted.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from PRESS.spatial_features import BCOS
from PRESS.utils.state import get_pv_players


def compare(match_dirs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for directory in map(Path, match_dirs):
        match_id = directory.name
        tracking = pd.read_parquet(directory / f'{match_id}_tracking.parquet')
        candidates = []
        # Fixed temporal subsample limits selection cost, not model resolution.
        for frame in tracking.loc[tracking.player_possession.notna(), 'frame'].iloc[::100]:
            try:
                state = get_pv_players(int(frame), tracking)
                nearest = np.linalg.norm(state['r_defs'] - state['x_bc'], axis=1).min()
                speed = np.linalg.norm(state['v_bc'])
                border = min(state['pitch_length']/2-abs(state['x_bc'][0]),
                             state['pitch_width']/2-abs(state['x_bc'][1]))
                candidates.append(dict(frame=int(frame), speed=speed, nearest=nearest,
                                       border=border, x=state['x_bc'][0], team=state['team_id'],
                                       period=state['frame_df'].iloc[0].get('period', None),
                                       attack_x=state['x_bc'][0]*(1 if state['attacking_side']=='left_to_right' else -1),
                                       velocity_attack=state['v_bc'][0]*(1 if state['attacking_side']=='left_to_right' else -1)))
            except ValueError:
                continue
        pool = pd.DataFrame(candidates)
        # Different speeds, proximity to defenders and pitch boundary. Within
        # each stratum select a fixed interior rank, avoiding extreme outliers.
        strata = [('slow', pool.speed.lt(1)), ('fast', pool.speed.gt(5)),
                  ('close_pressure', pool.nearest.lt(2)),
                  ('open', pool.nearest.gt(6)), ('boundary', pool.border.lt(4)),
                  ('central', pool.border.gt(20)),
                  ('against_attack',pool.velocity_attack.lt(-2))]
        for period in sorted(pool.period.dropna().unique()):
            for team in ['home','away']:
                strata.append((f'period_{period}_{team}',pool.period.eq(period)&pool.team.eq(team)))
        selected = set()
        engine = BCOS(tracking)
        for label, mask in strata:
            choices = pool.loc[mask & ~pool.frame.isin(selected)]
            if choices.empty:
                continue
            case = choices.iloc[len(choices)//2]
            frame = int(case.frame)
            selected.add(frame)
            start = perf_counter()
            old = engine.compute_bcos(frame, formulation='effective_hazard')
            middle = perf_counter()
            new = engine.compute_bcos(frame, formulation='transport')
            end = perf_counter()
            np.testing.assert_array_equal(old['grid'], new['grid'])
            p, q = old['Ball carrier open space'], new['Ball carrier open space']
            record = dict(match_id=match_id, stratum=label, **case.to_dict(),
                old_area=old['BCOS area'], new_area=new['BCOS area'],
                area_difference=new['BCOS area']-old['BCOS area'],
                area_relative_change=(new['BCOS area']/old['BCOS area']-1),
                mean_absolute_probability_difference=float(np.mean(abs(q-p))),
                max_absolute_probability_difference=float(np.max(abs(q-p))),
                old_seconds=middle-start, new_seconds=end-middle,
                new_mass_error=new['max_mass_error'], n_targets=len(p))
            records.append(record)
            pd.DataFrame(records).to_csv(output/'comparison.csv', index=False)
            print(json.dumps(record), flush=True)
    return pd.DataFrame(records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matches', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare(args.matches, args.output)
