"""Re-evaluate fixed real frames with protected and interceptable waiting.

Inputs: the existing case table and original prepared tracking tables. No
frames are reselected and no model is trained. All outputs are diagnostics.
The preliminary effective-hazard areas are retained as recorded references;
the previous transport surfaces are recomputed and checked against the case
table before comparing the active implementation.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from PRESS.model import BCOSModel, ReservoirBCOSModel
from PRESS.utils.state import get_pv_players


def compare(cases_csv, match_root, output_dir):
    cases = pd.read_csv(cases_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, refinements = [], []
    protected = ReservoirBCOSModel.from_ball_drive()
    waiting = BCOSModel.from_ball_drive()
    levels = [BCOSModel.from_ball_drive(arrival_order=n, path_order=n*4//3)
              for n in [24, 48, 96]]
    for match_id, group in cases.groupby('match_id', sort=False):
        tracking_file = Path(match_root)/str(match_id)/(str(match_id)+'_tracking.parquet')
        tracking = pd.read_parquet(tracking_file)
        for row in group.itertuples():
            state = get_pv_players(int(row.frame), tracking)
            arguments = [state[key] for key in ['x_bc', 'v_bc', 'r_defs', 'v_defs']]
            start = perf_counter()
            old = protected.predict_field(*arguments, pitch_length=state['pitch_length'],
                                           pitch_width=state['pitch_width'])
            middle = perf_counter()
            new = waiting.predict_field(*arguments, pitch_length=state['pitch_length'],
                                         pitch_width=state['pitch_width'])
            end = perf_counter()
            np.testing.assert_array_equal(old['grid'], new['grid'])
            np.testing.assert_allclose(old['BCOS area'], row.new_area, atol=2e-7, rtol=0,
                err_msg='The protected-reservoir reference or input state changed.')
            p, q = old['Ball carrier open space'], new['Ball carrier open space']
            assert np.max(q-p) < 2e-12, 'Waiting interception increased accessibility.'
            record = dict(match_id=str(match_id), frame=int(row.frame), stratum=row.stratum,
                speed=float(row.speed), nearest_defender=float(row.nearest),
                border_distance=float(row.border), team=row.team, period=int(row.period),
                effective_hazard_reference_area=float(row.old_area),
                protected_area=old['BCOS area'], waiting_area=new['BCOS area'],
                difference=new['BCOS area']-old['BCOS area'],
                relative_difference=new['BCOS area']/old['BCOS area']-1,
                mean_absolute_probability_difference=float(np.mean(abs(q-p))),
                max_absolute_probability_difference=float(np.max(abs(q-p))),
                max_probability_increase=float(np.max(q-p)),
                protected_reference_error=old['BCOS area']-row.new_area,
                mass_error=new['max_mass_error'], n_targets=len(p),
                protected_seconds=middle-start, waiting_seconds=end-middle)
            records.append(record)

            # Same distances/directions as the earlier convergence check.
            angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
            targets = np.array([state['x_bc']+distance*np.array([np.cos(a), np.sin(a)])
                for distance in [.05, .2, 1, 5, 15, 30, 40] for a in angles])
            targets = targets[(abs(targets[:,0]) <= state['pitch_length']/2)
                              & (abs(targets[:,1]) <= state['pitch_width']/2)]
            refined = [model.predict(*arguments, targets) for model in levels]
            refinement = dict(match_id=str(match_id), frame=int(row.frame),
                n_targets=len(targets),
                max_24_48=float(abs(refined[0]['Ball carrier open space']
                                   -refined[1]['Ball carrier open space']).max()),
                max_48_96=float(abs(refined[1]['Ball carrier open space']
                                   -refined[2]['Ball carrier open space']).max()),
                max_cause_48_96=float(abs(refined[1]['Defender interception probabilities']
                                         -refined[2]['Defender interception probabilities']).max()),
                mass_error=max(item['max_mass_error'] for item in refined))
            refinements.append(refinement)
            pd.DataFrame(records).to_csv(output_dir/'comparison.csv', index=False)
            pd.DataFrame(refinements).to_csv(output_dir/'convergence.csv', index=False)
            print(json.dumps(record), flush=True)
            print('refinement', json.dumps(refinement), flush=True)
    return pd.DataFrame(records), pd.DataFrame(refinements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases_csv', type=Path)
    parser.add_argument('match_root', type=Path)
    parser.add_argument('output_dir', type=Path)
    args = parser.parse_args()
    compare(args.cases_csv, args.match_root, args.output_dir)
