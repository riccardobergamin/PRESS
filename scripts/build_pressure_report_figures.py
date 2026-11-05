"""Rebuild pressure-band and sample-report figures from frozen public aggregates."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/examples/pressure_report'
BANDS=['None','Low','Medium','High'];COLORS=['#b9dbc2','#f6dc88','#eca865','#cf737c']
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(ROOT/'figures'/f'{name}.{ext}',dpi=170)
 plt.close(fig)
def build_band_validation():
 b=pd.read_csv(DATA/'h_band_bootstrap_terminal_coverage_20261105.csv',keep_default_na=False).set_index('band').loc[BANDS];x=np.arange(4)
 assert b.n.sum()==14919 and b.failures.sum()==1465
 assert np.allclose(b.observed_failure, b.failures/b.n, rtol=0, atol=1e-14)
 fig,ax=plt.subplots(figsize=(8.2,5),layout='constrained')
 for offset,column,lower,upper,label,color in [
  (-.1,'mean_h','mean_h_lower95','mean_h_upper95','Mean predicted h','#15354b'),
  (.1,'observed_failure','observed_lower95','observed_upper95','Observed current-action failures','#c37e31')]:
  y=100*b[column].to_numpy(float);lo=100*b[lower].to_numpy(float);hi=100*b[upper].to_numpy(float)
  ax.errorbar(x+offset,y,yerr=[y-lo,hi-y],fmt='o',color=color,capsize=4,lw=1.4,ms=5,label=label)
 for i,(_,r) in enumerate(b.iterrows()):ax.text(i,100*max(r.mean_h_upper95,r.observed_upper95)+1.7,f'n={int(r.n):,}',ha='center',fontsize=9)
 ax.set(xticks=x,xticklabels=['Minimal (None)','Low','Medium','High'],ylabel='Current-action failure (%)',ylim=(0,47),title='Fixed absolute bands · held-out development predictions')
 ax.legend(fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);save(fig,'h_band_validation')

def build_pattern_shares():
 b=pd.read_csv(DATA/'pattern_counts.csv');fig,axes=plt.subplots(1,2,figsize=(14,4.5),sharey=True,layout='constrained');y=np.arange(len(b))[::-1]
 for ax,share,lower,upper,denominator,title in [
  (axes[0],'cohort_share','cohort_lower95','cohort_upper95','cohort_possessions','All terminal-covered possessions'),
  (axes[1],'share','lower95','upper95','eligible_possessions','Pattern-specific eligible possessions')]:
  xmax=100*b[upper].max()*1.75
  for yy,(_,r),color in zip(y,b.iterrows(),['#2a7881','#c37e31','#5c769b']):
   ax.plot([100*r[lower],100*r[upper]],[yy,yy],color=color,lw=2)
   ax.scatter([100*r[share]],[yy],s=55,color=color,zorder=3)
   ax.text(xmax*.98,yy,f"{int(r.positive)}/{int(r[denominator]):,} · {100*r[share]:.2f}%",va='center',ha='right',fontsize=9)
  ax.set(yticks=y,yticklabels=b.pattern,xlabel='Share of possessions (%)',xlim=(0,xmax),ylim=(-.6,2.6),title=title)
  ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 fig.suptitle('Pressure patterns · 95% Wilson intervals',fontsize=12);save(fig,'vignette_08_typologies')

def build_team_exposure():
 team=pd.read_csv(DATA/'team_summary.csv').sort_values('team_name');peak=pd.read_csv(DATA/'team_peak_bands.csv',keep_default_na=False);yy=np.arange(len(team))[::-1]
 fig,(left,mid,right)=plt.subplots(1,3,figsize=(15.5,7.5),sharey=True,gridspec_kw={'width_ratios':[1.15,1,1]});offset=np.zeros(len(team))
 for band,color in zip(BANDS,COLORS):
  count=peak[peak.peak_band.eq(band)].set_index('team_name').n_possessions.reindex(team.team_name).to_numpy();share=100*count/team.n_possessions.to_numpy();left.barh(yy,share,left=offset,color=color,height=.65,label=band)
  for y,start,width in zip(yy,offset,share):
   if width>=10:left.text(start+width/2,y,f'{width:.0f}%',ha='center',va='center',fontsize=8)
  offset+=share
 left.set_yticks(yy,[f'{r.team_name} ({r.n_matches} '+('match)' if r.n_matches==1 else 'matches)') for r in team.itertuples()]);left.set(xlim=(0,100),xlabel='Possessions (%)',title='Maximum observed band')
 for ax,col,lo,hi,title in [(mid,'pressure_exposure_pct','exposure_lower95','exposure_upper95','Medium/High exposure'),(right,'retention_pct','retention_lower95','retention_upper95','Retention given exposure')]:
  ax.hlines(yy,team[lo],team[hi],color='#72949f',lw=1.4);ax.scatter(team[col],yy,s=30,color='#216b81',zorder=3)
  for y,r in zip(yy,team.itertuples()):
   k,n=(r.n_pressure_possessions,r.n_possessions) if ax is mid else (r.n_pressure_retained,r.n_pressure_possessions)
   ax.text(104,y,f'{k}/{n}',va='center',ha='right',fontsize=8)
  ax.set(xlim=(0,106),xlabel='Share of possessions (%)',title=title);ax.tick_params(axis='y',labelleft=False)
 for ax in [left,mid,right]:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 fig.suptitle('Ten-match A-League sample · one count per individual possession\nExposure and retention: 95% Wilson intervals; match coverage is unequal',fontsize=12,y=.99)
 fig.legend(handles=[Patch(facecolor=c,label=b) for b,c in zip(BANDS,COLORS)],loc='lower center',ncol=4,frameon=False,fontsize=9);fig.tight_layout(rect=(0,.055,1,.94));save(fig,'vignette_09_league_report')

def score_forest(table,entity,name,title):
 table=table.sort_values(['team_name','player_name'] if entity=='player' else ['team_name']).copy();n=len(table);fig,ax=plt.subplots(figsize=(12,max(5.5,.31*n+1.6)),layout='constrained');yy=np.arange(n)[::-1]
 lo=float(table.eo_per100_lower95.min());hi=float(table.eo_per100_upper95.max());span=max(hi-lo,10);xleft=min(lo,0)-.04*span;xright=max(hi,0)+.50*span;labelx=max(hi,0)+.47*span
 ax.axvline(0,color='0.55',lw=1,ls='--')
 ax.hlines(yy,table.eo_per100_lower95,table.eo_per100_upper95,color='#7799a8',lw=1.2)
 one=table.n_matches.eq(1)
 for mask,marker,color,label in [(one,'o','#8aa4af','One represented match'),(~one,'s','#216b81','Multiple represented matches')]:
  ax.scatter(table.loc[mask,'eo_per100'],yy[mask],s=30,marker=marker,color=color,label=label,zorder=3)
 for y,r in zip(yy,table.itertuples()):ax.text(labelx,y,f'{r.n_pressure_actions} / {r.n_pressure_possessions}',ha='right',va='center',fontsize=8)
 names=[f'{r.player_name} · {r.team_name} ({r.n_matches})' for r in table.itertuples()] if entity=='player' else [f'{r.team_name} ({r.n_matches})' for r in table.itertuples()]
 ax.set(yticks=yy,yticklabels=names,xlim=(xleft,xright),ylim=(-1,n),xlabel='E−O per 100 elevated-risk actions · positive = fewer failures than predicted',title=title)
 ax.text(labelx,n-.2,'Actions / possessions',ha='right',va='center',fontsize=9,fontweight='bold');ax.tick_params(axis='y',labelsize=8.5);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper center',bbox_to_anchor=(.58,-.08 if n<20 else -.035),fontsize=8,ncol=2,frameon=False);save(fig,name)

def build_score_profiles():
 players=pd.read_csv(DATA/'player_summary.csv');shown=players[players.n_pressure_possessions.ge(20)]
 assert len(shown)==40 and shown.eo_per100_lower95.notna().all()
 score_forest(shown,'player','h_player_eo_profiles','Player profiles · ≥20 exposed possessions\n95% possession-bootstrap intervals; number of represented matches in parentheses')
 teams=pd.read_csv(DATA/'team_summary.csv');score_forest(teams,'team','h_team_eo_profiles','Team profiles · fixed held-out predictions\n95% possession-bootstrap intervals; number of represented matches in parentheses')

def main():
 build_band_validation();build_pattern_shares();build_team_exposure();build_score_profiles()
 print('Rebuilt fixed-band validation, pattern shares, team exposure and E−O player/team profiles.')
if __name__=='__main__':main()
