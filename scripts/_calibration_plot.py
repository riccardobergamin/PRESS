"""Shared display style for frozen pooled calibration-bin summaries."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
AUDIT = []


def axes(title, xlabel='Predicted loss probability',
         ylabel='Observed loss frequency', *, limit=1., ax=None):
    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 6), layout='constrained')
    else:
        fig = ax.figure
    ax.plot([0, limit], [0, limit], '--', color='0.6', linewidth=1,
            label='Perfect calibration')
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, limit),
           ylim=(0, limit), aspect='equal')
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.15)
    return fig, ax


def points(ax, table, x, y, label, *, color='#126a80', marker='o', expected_n=None):
    assert len(table) == 10
    assert np.isfinite(table[[x, y]]).all().all()
    assert table[[x, y]].ge(0).all().all() and table[[x, y]].le(1).all().all()
    if expected_n is not None:
        assert table.n.sum() == expected_n
    ax.scatter(table[x], table[y], s=48, alpha=.75, color=color,
               marker=marker, label=label, edgecolors='none', linewidths=0)
    ece = float(np.average(abs(table[x]-table[y]), weights=table.n))
    AUDIT.append(dict(title=ax.get_title(), series=label, bins=len(table),
                      n=int(table.n.sum()), ece_from_bins=ece))
    return ece


def save(fig, *names):
    for ax in fig.axes:
        assert len(ax.lines) == 1, 'Only the perfect-calibration diagonal may be a line.'
        assert not ax.patches, 'Calibration figures must not contain bars.'
        assert all(len(c.get_offsets()) == 10 for c in ax.collections)
        ax.legend(fontsize=9, loc='upper left')
    for name in names:
        for suffix in ['png', 'svg']:
            fig.savefig(ROOT/'figures'/f'{name}.{suffix}', dpi=180)
    plt.close(fig)
