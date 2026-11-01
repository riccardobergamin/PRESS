"""Render published aggregate bins; no raw tracking, prediction or fitting."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]

def main():
    folder=ROOT/'data/benchmarks/action_selection'
    bins=pd.read_csv(folder/'calibration_deciles.csv');metrics=pd.read_csv(folder/'metrics.csv').set_index('model')
    for model in metrics.index:
        d=bins.loc[bins.model.eq(model)]
        assert len(d)==10 and d.n.sum()==16918
        assert abs(np.sum(d.n*abs(d.predicted-d.observed))/16918-metrics.loc[model,'ece_quantile10'])<1e-12
    plt.rcParams.update({'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.5),layout='constrained')
    ax=axes[0];ax.plot([0,1],[0,1],'--',color='.6',linewidth=1)
    for model,label,color in [('native_raw','Native (primary)','#126a80'),('temperature_diagnostic','Temperature diagnostic','#c97930')]:
        d=bins.loc[bins.model.eq(model)]
        ax.plot(d.predicted,d.observed,'o-',color=color,ms=4,label=f'{label}: ECE {100*metrics.loc[model,"ece_quantile10"]:.2f}%')
    ax.set(xlabel='Predicted probability of passing',ylabel='Observed pass frequency',xlim=(0,1),ylim=(0,1),title='Pooled held-out calibration · 16,918 decisions',aspect='equal')
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    d=bins.loc[bins.model.eq('native_raw')]
    axes[1].bar(d.bin,100*(d.observed-d.predicted),color='#126a80')
    axes[1].axhline(0,color='.5',linewidth=1)
    axes[1].set(xlabel='Equal-count probability bin (low → high)',ylabel='Observed minus predicted (percentage points)',title='Native probabilities · signed bin gaps',xticks=list(range(10)))
    for ax in [axes[0]]:
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
    for ext in ['png','svg']:fig.savefig(ROOT/f'figures/action_selection_pooled_calibration.{ext}',dpi=180)
    plt.close(fig);print('Rendered action-selection calibration from verified aggregate bins.')

if __name__=='__main__':main()
