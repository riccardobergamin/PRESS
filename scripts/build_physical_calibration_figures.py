"""Render physical-component calibration from audited public aggregate bins."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=ROOT/'data/benchmarks/calibration'
    bins=pd.read_csv(source/'pooled_bins.csv')
    metrics=pd.read_csv(source/'pooled_metrics.csv')
    assert not bins.duplicated(['task','model','bin']).any()
    for row in metrics.itertuples():
        selected=bins.loc[bins.task.eq(row.task)&bins.model.eq(row.model)]
        assert len(selected)==10 and int(selected.n.sum())==row.n
        assert int(selected.losses.sum())==row.losses
        ece=np.sum(selected.n/row.n*abs(selected.observed_loss-selected.predicted_loss))
        assert abs(ece-row.ECE_quantile)<1e-12,(row.task,row.model,ece,row.ECE_quantile)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    labels={'Precedente':'Previous','Aggiornato':'Updated'}
    colors={'Precedente':'#336EAC','Aggiornato':'#DA6D2B'}
    def panel(ax,task,title,limit):
        meta=metrics.loc[metrics.task.eq(task)].iloc[0]
        ax.plot([0,limit],[0,limit],'--',color='#777777',linewidth=1.2,label='Perfect calibration')
        for model in ['Precedente','Aggiornato']:
            selected=bins.loc[bins.task.eq(task)&bins.model.eq(model)].sort_values('predicted_loss')
            score=metrics.loc[metrics.task.eq(task)&metrics.model.eq(model),'ECE_quantile'].item()
            ax.plot(selected.predicted_loss,selected.observed_loss,'o-',color=colors[model],markersize=6,linewidth=1.8,label=f'{labels[model]} · ECE {score*100:.3f}%')
        ax.set(title=f'{title}\nn={int(meta.n):,} · losses={int(meta.losses):,}',xlabel='Predicted loss probability',ylabel='Observed loss frequency',xlim=(0,limit),ylim=(0,limit),aspect='equal')
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=9)
    def save(fig,name):
        fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180)
        fig.savefig(ROOT/'figures'/f'{name}.svg')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,5.9),constrained_layout=True)
    panel(ax,'ball_drive','Ball-drive risk · pooled out-of-fold calibration',.35)
    save(fig,'ball_drive_physical_pooled_calibration')
    fig,axes=plt.subplots(1,2,figsize=(12.8,6),constrained_layout=True)
    panel(axes[0],'pass_raw','Pass risk · raw',1.)
    panel(axes[1],'pass_isotonic','Pass risk · isotonic',1.)
    fig.suptitle('Pooled out-of-fold calibration · five folds, ten matches',fontsize=14)
    save(fig,'pass_risk_physical_pooled_calibration')
    print('Verified six curves / 60 bins against reported ECE; generated two PNG/SVG figures.')

if __name__=='__main__':main()
