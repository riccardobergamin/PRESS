"""Numerical refinement on real data; no model fitting."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from PRESS.model import BCOSModel
from PRESS.utils.state import get_pv_players
import argparse
p=argparse.ArgumentParser(description='Refine BCOS quadrature on selected real frames.')
p.add_argument('cases_csv',type=Path)
p.add_argument('match_root',type=Path)
p.add_argument('output_csv',type=Path)
args=p.parse_args()
cases=pd.read_csv(args.cases_csv)
records=[]
for mid,group in cases.groupby('match_id'):
 t=pd.read_parquet(args.match_root/str(mid)/(str(mid)+'_tracking.parquet'))
 for row in group.itertuples():
  state=get_pv_players(int(row.frame),t)
  angle=np.linspace(0,2*np.pi,8,endpoint=False)
  points=np.array([state['x_bc']+d*np.array([np.cos(a),np.sin(a)]) for d in [.05,.2,1,5,15,30,40] for a in angle])
  points=points[(abs(points[:,0])<=52.5)&(abs(points[:,1])<=34)]
  args=[state[k] for k in ['x_bc','v_bc','r_defs','v_defs']]+[points]
  levels=[BCOSModel.from_ball_drive(arrival_order=n,path_order=n*4//3).predict(*args) for n in [24,48,96]]
  record=dict(match_id=str(mid),frame=int(row.frame),n_targets=len(points),
   max_24_48=float(abs(levels[0]['Ball carrier open space']-levels[1]['Ball carrier open space']).max()),
   max_48_96=float(abs(levels[1]['Ball carrier open space']-levels[2]['Ball carrier open space']).max()),
   max_cause_48_96=float(abs(levels[1]['Defender interception probabilities']-levels[2]['Defender interception probabilities']).max()),
   mass_error=max(x['max_mass_error'] for x in levels))
  records.append(record)
  print(json.dumps(record),flush=True)
pd.DataFrame(records).to_csv(args.output_csv,index=False)
