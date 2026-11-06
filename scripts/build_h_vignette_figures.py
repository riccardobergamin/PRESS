"""Rebuild hazard case figures from exported observed-match OOF summaries."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PRESS.utils.possession import hazard_band_thresholds
ROOT=Path(__file__).resolve().parents[1]
COLORS=['#dceede','#fff0b8','#f9cf9e','#edb3b5']
LABELS=['None','Low','Medium','High']

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=170)
    plt.close(fig)

def main():
    t=hazard_band_thresholds();edges=[0.,t[8],t[4],t[2],1.]
    cases=pd.read_csv(ROOT/'data/examples/hazard/cases.csv',dtype={'match_id':str});summary=pd.read_csv(ROOT/'data/examples/hazard/case_summary.csv')
    fig,axes=plt.subplots(3,1,figsize=(10,11),sharey=True)
    for ax,(case,g) in zip(axes,cases.groupby('case',sort=True)):
        g=g.sort_values('frame_start');s=summary[summary.case.eq(case)].iloc[0]
        for lo,hi,color in zip(edges[:-1],edges[1:],COLORS):ax.axhspan(lo,hi,color=color,alpha=.7)
        x=g.time_since_start
        ax.plot(x,g.h,'o-',color='#182d40',lw=2,label='h',zorder=4)
        ax.plot(x,g.carry_contribution,'--',color='#316f9d',lw=1.5,label='P(carry) × carry loss')
        ax.plot(x,g.pass_contribution,':',color='#b35f17',lw=2,label='P(pass) × mean pass loss')
        release=g[g.action.eq('pass')];ax.scatter(release.time_since_start,release.h,marker='s',s=80,facecolors='white',edgecolors='#182d40',zorder=5,label='Observed pass release')
        outcome='completed pass' if not s.terminal_individual_loss else 'failed pass'
        ax.set(title=f"Case {case} · {s.player_name} · {s.pattern}\n{s.start_timestamp}–{s.end_timestamp} · {outcome}",ylabel='Next-action loss probability',ylim=(0,.45),xlabel='Seconds since start of individual possession')
        ax.grid(axis='x',alpha=.2);ax.legend(loc='upper left',fontsize=8,ncol=2)
    fig.suptitle('Three context-checked pressure situations\nFrozen matching-fold predictions; cases selected for illustration',fontsize=12,y=.995)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'h_real_case_trajectories')
if __name__=='__main__':main()
