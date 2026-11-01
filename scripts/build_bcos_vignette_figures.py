"""Rebuild descriptive BCOS figures from public single-match aggregates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/examples/bcos'

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/{name}.{ext}',dpi=180)
    plt.close(fig)

def main():
    meta=json.loads((DATA/'metadata.json').read_text());nx,ny=meta['grid'];L=meta['pitch_length'];W=meta['pitch_width']
    cells=pd.read_csv(DATA/'spatial_cells.csv')
    norm=Normalize(0,float(cells.loc[cells.displayed,'mean_bcos_m2'].max()))
    cmap=plt.get_cmap('viridis');teams=['home','away']
    fig,axes=plt.subplots(1,2,figsize=(12,5.3),layout='constrained')
    for ax,team in zip(axes,teams):
        pitch=Pitch(pitch_type='skillcorner',pitch_length=L,pitch_width=W,
            pitch_color='#eef1f5',line_color='#3d4858',linewidth=1,line_zorder=4)
        pitch.draw(ax=ax)
        for r in cells[cells.team.eq(team)].itertuples():
            x=-L/2+r.ix*L/nx;y=-W/2+r.iy*W/ny
            color=cmap(norm(r.mean_bcos_m2)) if r.displayed else '#dfe4ea'
            ax.add_patch(Rectangle((x,y),L/nx,W/ny,facecolor=color,edgecolor='white',lw=.45,zorder=1,
                hatch=None if r.displayed else '///'))
            if r.displayed:
                ax.text(x+L/nx/2,y+W/ny/2,str(r.n_possessions),ha='center',va='center',fontsize=8,
                    color='white' if norm(r.mean_bcos_m2)<.55 else '#17212b',zorder=5)
        ax.set_title(meta['teams'][team]+' · attack →',fontsize=12,pad=8)
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,label='Mean BCOS at the carrier position (m²)',shrink=.76,pad=.02)
    fig.suptitle('Brisbane FC vs Perth Glory · descriptive BCOS by carrier location',fontsize=14)
    fig.supxlabel('Labels: distinct individual possessions · hatched: fewer than 5 · no interpolation',fontsize=9)
    save(fig,'bcos_match_spatial')

    temporal=pd.read_csv(DATA/'temporal_thirds.csv')
    fig,axes=plt.subplots(3,2,figsize=(12,8.2),sharey='row',layout='constrained')
    colors={'home':'#16758a','away':'#bd5f28'}
    for i,third in enumerate(['Defensive','Central','Attacking']):
        for j,period in enumerate([1,2]):
            ax=axes[i,j]
            for team in teams:
                g=temporal[temporal.team.eq(team)&temporal.third.eq(third)&temporal.period.eq(period)].set_index('time_bin').reindex(range(5))
                g['centre_minute']=(np.arange(5)+.5)*meta['half_durations_minutes'][str(period)]/5
                x=g.centre_minute.to_numpy();y=g.mean_bcos_m2.where(g.displayed.fillna(False).astype(bool)).to_numpy()
                ax.plot(x,y,'o-',color=colors[team],lw=1.8,ms=4,label=meta['teams'][team])
            ax.set_xlim(0,meta['half_durations_minutes'][str(period)])
            ax.set_ylim(bottom=0)
            for goal in meta.get('goals',[]):
                if int(goal['period'])==period:
                    ax.axvline(goal['period_minute'],ls=':',lw=1.4,color=colors[goal['team']],
                        label=goal['team_name']+' goal ('+goal['clock']+')' if i==0 else '_nolegend_')
            ax.grid(alpha=.18);ax.set_title(f'{third} third · half {period}',fontsize=11)
            if j==0:ax.set_ylabel('Mean BCOS (m²)')
            if i==2:ax.set_xlabel('Minutes elapsed in this half')
            if i==0 and j==1:ax.legend(fontsize=9)
    fig.suptitle('BCOS through the match, conditional on the carrier’s third',fontsize=14)
    fig.supxlabel('Five equal windows per half · checkpoint means · gaps: fewer than 5 individual possessions',fontsize=9)
    save(fig,'bcos_match_temporal')

    support=pd.read_csv(DATA/'grid_support.csv');boot=pd.read_csv(DATA/'grid_bootstrap_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,3.9),layout='constrained')
    for team in teams:
        g=support[support.team.eq(team)].sort_values('n_cells')
        axes[0].plot(g.n_cells,100*g.share_observations_supported,'o-',color=colors[team],label=meta['teams'][team])
        b=boot[boot.team.eq(team)&boot.cluster_unit.eq('team_possession_id')].copy();b['n_cells']=b.nx*b.ny;b=b.sort_values('n_cells')
        axes[1].plot(b.n_cells,100*b.median_relative_standard_error,'o-',color=colors[team])
    axes[0].set(ylabel='Checkpoints in supported cells (%)',xlabel='Number of display cells',ylim=(0,105))
    axes[1].set(ylabel='Median bootstrap SE / cell mean (%)',xlabel='Number of display cells')
    axes[0].legend(fontsize=8)
    for ax in axes:ax.axvline(nx*ny,color='.5',ls='--',lw=1);ax.grid(alpha=.15)
    fig.suptitle('Display resolution: sample support and aggregation stability',fontsize=12)
    save(fig,'bcos_grid_support')
    print('BCOS_DESCRIPTIVE_FIGURES_REBUILT')

if __name__=='__main__':main()
