"""Render frozen aggregate pass diagnostics; no fitting or raw tracking inputs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'data/benchmarks/pass_risk'
    bins=pd.read_csv(folder/'calibration_deciles.csv')
    curves=pd.read_csv(folder/'curvature_metrics.csv')
    case='02_vmax_5_5'
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2))
    axes[0].plot([0,1],[0,1],color='#999999',ls='--',lw=1,label='Perfect calibration')
    for method,label,color in [('trajectory_raw','Raw PPCF','#999999'),
                               ('trajectory_platt','Platt','#ba6c32'),
                               ('trajectory_isotonic','Isotonic (selected)','#245985')]:
        d=bins.loc[bins.case.eq(case)&bins.subset.eq('all')&bins.model.eq(method)]
        axes[0].plot(d.predicted_success,d.observed_success,'o-',color=color,label=label,ms=4)
    axes[0].set(xlabel='Mean predicted completion probability',ylabel='Observed completion frequency',
                xlim=(-.02,1.02),ylim=(-.02,1.02),title='Held-out calibration · 10 quantile bins')
    axes[0].legend(fontsize=8,loc='upper left')
    d=curves.loc[curves.case.eq(case)&curves.model.eq('trajectory_isotonic')].set_index('subset')
    labels=['curvature_lt10','curvature_ge10'];x=np.arange(2);w=.34
    axes[1].bar(x-w/2,d.loc[labels,'cross_entropy'],w,color='#245985',label='Selected model')
    axes[1].bar(x+w/2,d.loc[labels,'null_cross_entropy'],w,color='#c9ced3',label='Training-subgroup null')
    axes[1].set_xticks(x,['Curvature <10%\n(n=5,211)','Curvature ≥10%\n(n=1,495)'])
    axes[1].set(ylabel='Cross-entropy (natural log)',title='Same model across curvature groups',ylim=(0,.60))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(ROOT/f'figures/pass_benchmark_calibration.{suffix}',dpi=160)
    plt.close(fig)
    print('Rendered frozen OOF calibration and curvature diagnostics.')


if __name__=='__main__':main()
