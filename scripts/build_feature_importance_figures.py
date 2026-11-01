"""Rebuild frozen-model split-importance plots from public aggregate tables."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
LABELS={
    'pass_distance':'Pass distance','pass_angle':'Pass angle vs carrier motion','passer_speed':'Carrier speed',
    'target_pitch_control':'Raw target pitch control','target_xt_team':'Own xT at target',
    'target_xt_opp':'Opposition xT at target','ball_drive_prob':'One-second carry-loss risk',
    'xt_team_bc':'Own xT at carrier','xt_opp_bc':'Opposition xT at carrier',
    'team_pressure_line_index':'Own line at target','opp_pressure_line_index':'Opposition line at target',
    'opponents_ahead_target':'Opponents beyond target','teammates_ahead_target':'Teammates beyond target',
    'opponents_ahead_bc':'Opponents beyond carrier','teammates_ahead_bc':'Teammates beyond carrier',
    'bcos':'BCOS area','drive_success_1s':'One-second carry success','carrier_speed':'Carrier speed',
    'carrier_x_attack':'Carrier longitudinal position','carrier_y_attack':'Carrier lateral position',
    'carrier_velocity_attack_angle':'Carrier velocity angle','completed_canonical_drives':'Completed full carries',
    'time_since_start':'Elapsed individual-possession time','action_carry_probability':'P(carry)',
    'match_clock_min':'Match minute','score_difference_prefix':'Known score difference',
    'start_type_raw':'Granular start type','previous_action_raw':'Granular previous action',
    'team_line_pass_success':'Own line · pass success','team_line_pass_failure':'Own line · pass failure',
    'opp_line_pass_success':'Opposition line · pass success','opp_line_pass_failure':'Opposition line · pass failure',
    'teammates_ahead_pass_success':'Teammates ahead · pass success','teammates_ahead_pass_failure':'Teammates ahead · pass failure',
    'opponents_ahead_pass_success':'Opponents ahead · pass success','opponents_ahead_pass_failure':'Opponents ahead · pass failure',
    'pass_success_mean':'Expected pass success','xt_team_pass_success':'Own xT · pass success',
    'xt_opp_pass_failure':'Opposition xT · pass failure','xt_team_drive_1s':'Own xT after1s carry',
    'xt_opp_drive_1s':'Opposition xT after1s carry','team_line_distance_drive_1s':'Own line distance after1s',
    'opp_line_distance_drive_1s':'Opposition line distance after1s',
    'team_line_index_drive_1s':'Own line after1s','opp_line_index_drive_1s':'Opposition line after1s'}

def main():
    d=pd.read_csv(ROOT/'data/benchmarks/feature_importance/importance_summary.csv')
    titles={'pass_selection':'Pass destination choice','action_selection':'Pass versus carry choice','residual_risk':'Conditional residual loss'}
    for task,title in titles.items():
        g=d[d.task.eq(task)].sort_values('rank').head(12).iloc[::-1]
        y=np.arange(len(g));fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
        ax.barh(y,100*g.gain_share,color='#16758a',alpha=.65,height=.65)
        x=100*g.fold_mean_share.to_numpy();err=np.vstack([x-100*g.fold_min_share.to_numpy(),100*g.fold_max_share.to_numpy()-x])
        ax.errorbar(x,y,xerr=err,fmt='o',color='#bf6f25',ecolor='#bf6f25',capsize=3,ms=4,lw=1.3)
        ax.set_yticks(y,[LABELS.get(n,n.replace('_',' ')) for n in g.feature])
        ax.set_xlabel('Share of total training split gain (%)')
        ax.set_title(title+' · feature importance',fontsize=13,pad=12)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.legend(handles=[Patch(facecolor='#16758a',alpha=.65,label='Production refit'),
            Line2D([0],[0],marker='o',color='#bf6f25',label='Outer folds: mean and min–max')],fontsize=8,loc='lower right')
        for ext in ['png','svg']:fig.savefig(ROOT/f'figures/feature_importance_{task}.{ext}',dpi=180)
        plt.close(fig)
    print('THREE_MODEL_IMPORTANCE_FIGURES_REBUILT')

if __name__=='__main__':main()
