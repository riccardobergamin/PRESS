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
def main():
 b=pd.read_csv(DATA/'band_validation.csv',keep_default_na=False).set_index('band').loc[BANDS];x=np.arange(4)
 fig,ax=plt.subplots(figsize=(8.2,5),layout='constrained')
 y=100*b.observed_loss.to_numpy(float);lo=100*b.cluster_lower95.to_numpy(float);hi=100*b.cluster_upper95.to_numpy(float)
 ax.bar(x,y,color=COLORS,width=.62,label='Observed current-action failures')
 ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='none',ecolor='#414141',capsize=4,lw=1.4,label='95% whole-match bootstrap interval')
 ax.plot(x,100*b.mean_h.to_numpy(float),'D--',color='#15354b',ms=6,lw=1.4,label='Mean predicted h')
 for i,(_,r) in enumerate(b.iterrows()):ax.text(i,100*r.cluster_upper95+1.7,f'n={int(r.n):,}',ha='center',fontsize=9)
 ax.set(xticks=x,xticklabels=['Minimal (None)','Low','Medium','High'],ylabel='Current-action failure (%)',ylim=(0,47),title='Fixed absolute bands · held-out development predictions')
 ax.legend(fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);save(fig,'h_band_validation')
 patterns=pd.read_csv(DATA/'pattern_counts.csv');fig,ax=plt.subplots(figsize=(8.5,4.2),layout='constrained');yy=np.arange(len(patterns))[::-1]
 ax.barh(yy,patterns.positive,color=['#2a7881','#c37e31','#5c769b'],height=.55)
 ax.set_yticks(yy,['Pressure escape','Drawing pressure','Sustained successful carrying'])
 for y,r in zip(yy,patterns.itertuples()):ax.text(r.positive+5,y,f'{r.positive}  (eligible={r.eligible_possessions}; unknown={r.unknown})',va='center',fontsize=9)
 ax.set(xlim=(0,500),xlabel='Individual possessions with the observed pattern',title='Three reporting patterns · definitions and denominators differ')
 ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);save(fig,'vignette_08_typologies')
 team=pd.read_csv(DATA/'team_summary.csv').sort_values('team_name');peak=pd.read_csv(DATA/'team_peak_bands.csv',keep_default_na=False)
 fig,(left,right)=plt.subplots(1,2,figsize=(13,7),sharey=True,gridspec_kw={'width_ratios':[1.25,1]});yy=np.arange(len(team))[::-1];offset=np.zeros(len(team))
 for band,col in zip(BANDS,COLORS):
  count=peak[peak.peak_band.eq(band)].set_index('team_name').n_possessions.reindex(team.team_name).to_numpy();share=100*count/team.n_possessions.to_numpy()
  left.barh(yy,share,left=offset,color=col,height=.65,label=band)
  for y,start,width in zip(yy,offset,share):
   if width>=9:left.text(start+width/2,y,f'{width:.0f}%',ha='center',va='center',fontsize=8)
  offset+=share
 left.set_yticks(yy,[f'{r.team_name} ({r.n_matches} match'+('es)' if r.n_matches!=1 else ')') for r in team.itertuples()]);left.set(xlim=(0,100),xlabel='Possessions by highest observed canonical band (%)',title='Exposure · one individual possession, one count')
 p=team.retention_pct.to_numpy();lo=team.retention_lower95.to_numpy();hi=team.retention_upper95.to_numpy()
 right.errorbar(p,yy,xerr=[p-lo,hi-p],fmt='o',color='#216b81',ecolor='#72949f',capsize=3,ms=5)
 for y,r in zip(yy,team.itertuples()):right.text(99,y,f'{r.n_pressure_retained}/{r.n_pressure_possessions}',va='center',ha='right',fontsize=8)
 right.set(xlim=(35,102),xlabel='Terminal nonloss in pressure-exposed possessions (%)',title='Retention · Medium/High exposure');right.tick_params(axis='y',labelleft=False)
 for ax in [left,right]:ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 fig.suptitle('A-League 2024/25 · ten-match sample, twelve teams\nFrozen OOF summaries; observation coverage is unequal',fontsize=12,y=.99)
 fig.legend(handles=[Patch(facecolor=c,label=b) for b,c in zip(BANDS,COLORS)],loc='lower center',ncol=4,frameon=False,fontsize=9)
 fig.tight_layout(rect=(0,.055,1,.94));save(fig,'vignette_09_league_report')
 players=pd.read_csv(DATA/'player_summary.csv');shown=players[players.n_pressure_possessions.ge(20)].copy()
 fig,ax=plt.subplots(figsize=(9,5.8),layout='constrained')
 for r in shown.itertuples():ax.plot([r.n_pressure_possessions]*2,[r.retention_lower95,r.retention_upper95],color='#78929d',alpha=.25,lw=1)
 one=shown[shown.n_matches.eq(1)];more=shown[shown.n_matches.ge(2)]
 ax.scatter(one.n_pressure_possessions,one.retention_pct,marker='o',s=35,color='#8aa4af',label='One observed match')
 ax.scatter(more.n_pressure_possessions,more.retention_pct,marker='s',s=38,color='#216b81',label='Two or more observed matches')
 for pid,shift in [(14736,(-70,25)),(51045,(-90,-35))]:
  r=shown[shown.provider_player_id.eq(pid)].iloc[0];ax.scatter([r.n_pressure_possessions],[r.retention_pct],s=100,facecolors='none',edgecolors='#a46620',lw=1.8)
  ax.annotate(r.player_name+f'\n{r.n_pressure_retained}/{r.n_pressure_possessions}, {r.n_matches} matches',(r.n_pressure_possessions,r.retention_pct),xytext=shift,textcoords='offset points',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':2},arrowprops={'arrowstyle':'-','color':'#a46620'})
 ax.set(xlabel='Distinct pressure-exposed individual possessions',ylabel='Possessions ending without individual loss (%)',ylim=(0,103),xlim=(17,58),title='Player retention profiles · exposure and uncertainty visible')
 ax.legend(loc='lower right',fontsize=8);ax.grid(alpha=.15);save(fig,'h_player_retention_profiles')
 tr=pd.read_csv(DATA/'endpoint_transitions.csv',keep_default_na=False);matrix=tr.pivot(index='start_band',columns='end_band',values='n').reindex(index=BANDS,columns=BANDS).fillna(0)
 values=matrix.to_numpy();row_total=values.sum(axis=1);fraction=values/row_total[:,None]
 fig,ax=plt.subplots(figsize=(6.4,5.4),layout='constrained');im=ax.imshow(fraction,vmin=0,vmax=1,cmap='Blues')
 for i in range(4):
  for j in range(4):ax.text(j,i,f'{values[i,j]:.0f}\n{100*fraction[i,j]:.1f}%',ha='center',va='center',color='white' if fraction[i,j]>.5 else '#152e40',fontsize=9)
 ax.set(xticks=range(4),xticklabels=BANDS,yticks=range(4),yticklabels=BANDS,xlabel='Band at true individual-possession end',ylabel='Band at true individual-possession start',title='Observed endpoint transitions\n5,601 complete-endpoint possessions')
 fig.colorbar(im,ax=ax,label='Share within starting band');save(fig,'h_endpoint_transitions')
 print('Pressure report figures rebuilt from fixed aggregate tables.')
if __name__=='__main__':main()
