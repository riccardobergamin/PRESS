"""Reproduce residual and composed-risk reliability plots from public bins."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
    bins=pd.read_csv(ROOT/'data/benchmarks/residual_risk/calibration_quantile10.csv')
    metrics=pd.read_csv(ROOT/'data/benchmarks/residual_risk/pooled_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharex=True,sharey=True,constrained_layout=True)
    for ax,variant,title in zip(axes,['base','with_action_choice'],['Base','Base + P(carry)']):
        ax.plot([0,1],[0,1],color='0.55',linestyle='--',linewidth=1,label='Perfect calibration')
        for method,label,color in [('native','Native','#16758a'),('temperature','Temperature','#d57b27')]:
            b=bins[bins.variant.eq(variant)&bins.method.eq(method)].sort_values('bin')
            m=metrics[metrics.variant.eq(variant)&metrics.method.eq(method)].iloc[0]
            assert len(b)==10 and b.n.sum()==m.n
            ax.plot(b.predicted,b.observed,'o-',color=color,label=f'{label}: ECE {100*m.ece_quantile10:.2f}%')
        ax.set(title=title,xlabel='Mean predicted terminal loss',xlim=(0,1),ylim=(0,1))
        ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper left');ax.set_aspect('equal')
    axes[0].set_ylabel('Observed terminal loss share')
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/residual_risk_pooled_calibration.{ext}',dpi=180)
    plt.close(fig)
    folder=ROOT/'data/benchmarks/composed_risk'
    if folder.exists():
        bins=pd.read_csv(folder/'calibration_quantile10.csv')
        metrics=pd.read_csv(folder/'pooled_metrics.csv')
        fig,axes=plt.subplots(1,2,figsize=(10,4.6),sharex=True,sharey=True,constrained_layout=True)
        for ax,model,title in zip(axes,['h','xloss_with_action_choice'],['Next-action hazard h','Terminal individual xLoss']):
            ax.plot([0,1],[0,1],color='0.55',linestyle='--',linewidth=1,label='Perfect calibration')
            b=bins[bins.cohort.eq('verified_endpoints')&bins.model.eq(model)].sort_values('bin')
            m=metrics[metrics.cohort.eq('verified_endpoints')&metrics.model.eq(model)].iloc[0]
            assert len(b)==10 and b.n.sum()==m.n
            ax.plot(b.predicted,b.observed,'o-',color='#16758a',label=f'Pooled OOF: ECE {100*m.ece_quantile10:.2f}%')
            ax.set(title=title,xlabel='Mean predicted loss',xlim=(0,1),ylim=(0,1))
            ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper left')
        axes[0].set_ylabel('Observed loss share for the stated target')
        for ext in ['png','svg']:fig.savefig(ROOT/f'figures/composed_risk_pooled_calibration.{ext}',dpi=180)
        plt.close(fig)
    print('RESIDUAL_POOLED_CALIBRATION_FIGURES_WRITTEN')

if __name__=='__main__':main()
