"""Rebuild descriptive BCOS maps and checkpoint series; no model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scripts._pitch_heatmap import draw_cell_map

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/examples/bcos'
COLORS = {'home': '#16758a', 'away': '#bd5f28'}


def save(fig, name):
    for ext in ['png', 'svg']:
        fig.savefig(ROOT / f'figures/{name}.{ext}', dpi=180)
    plt.close(fig)


def spatial_display(cells, nx, ny):
    """Fill only unobserved display cells from adjacent observed checkpoints."""
    observed = np.full((nx, ny), np.nan)
    counts = np.zeros((nx, ny), int)
    weights = np.zeros((nx, ny), int)
    for row in cells.itertuples():
        observed[row.ix, row.iy] = row.mean_bcos_m2
        counts[row.ix, row.iy] = row.n_possessions
        weights[row.ix, row.iy] = row.n_checkpoints
    shown = observed.copy()
    filled = ~np.isfinite(observed)
    for ix, iy in zip(*np.where(filled)):
        sl = (slice(max(0, ix - 1), min(nx, ix + 2)),
              slice(max(0, iy - 1), min(ny, iy + 2)))
        valid = np.isfinite(observed[sl])
        assert valid.any(), 'No observed adjacent cell available.'
        shown[ix, iy] = np.average(observed[sl][valid], weights=weights[sl][valid])
    return observed, shown, counts, filled


def temporal_display(g, duration):
    """Keep window means separate from linear display interpolation."""
    centres = np.arange(.5, duration, .5)
    means, counts = [], []
    for t in centres:
        selected = g[(g.period_minute >= max(0, t - 2.5)) &
                     (g.period_minute <= min(duration, t + 2.5))]
        means.append(selected.bcos_m2.mean() if len(selected) else np.nan)
        counts.append(len(selected))
    means = np.asarray(means)
    # Fill internal empty windows only; do not extrapolate beyond support.
    shown = pd.Series(means).interpolate(limit_area='inside').to_numpy()
    return centres, means, shown, np.asarray(counts)


def main():
    meta = json.loads((DATA / 'metadata.json').read_text())
    nx, ny = meta['grid']
    cells = pd.read_csv(DATA / 'spatial_cells.csv')
    norm = Normalize(0, float(cells.mean_bcos_m2.max()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout='constrained')
    spatial_records = []
    for ax, team in zip(axes, ['home', 'away']):
        observed, grid, counts, filled = spatial_display(cells[cells.team.eq(team)], nx, ny)
        mesh = draw_cell_map(ax, grid, norm=norm, cmap='viridis', counts=counts)
        for ix, iy in zip(*np.where(filled)):
            ax.add_patch(Rectangle((-52.5 + ix * 105 / nx, -34 + iy * 68 / ny),
                                   105 / nx, 68 / ny, facecolor='none',
                                   edgecolor=(1, 1, 1, .7), hatch='///',
                                   linewidth=.5, zorder=3))
        for ix, iy in np.ndindex(grid.shape):
            spatial_records.append(dict(team=team, ix=ix, iy=iy,
                observed_mean_bcos_m2=observed[ix, iy], display_bcos_m2=grid[ix, iy],
                n_possessions=int(counts[ix, iy]), estimated=bool(filled[ix, iy])))
        ax.set_title(meta['teams'][team] + ' · attack →', fontsize=12, pad=8)
    fig.colorbar(mesh, ax=axes, label='Mean BCOS at carrier position (m²)',
                 shrink=.76, pad=.02)
    fig.suptitle('Brisbane FC–Perth Glory · BCOS by carrier location', fontsize=14)
    fig.supxlabel('10 × 6 cells · labels: distinct possessions · hatched cells (0): neighbouring-cell estimate',
                  fontsize=9)
    save(fig, 'bcos_match_spatial')
    pd.DataFrame(spatial_records).to_csv(DATA / 'spatial_display_cells.csv', index=False)

    series = pd.read_csv(DATA / 'checkpoint_series.csv')
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharey='row', layout='constrained')
    temporal_records = []
    for i, third in enumerate(['Defensive', 'Central', 'Attacking']):
        for j, period in enumerate([1, 2]):
            ax = axes[i, j]
            duration = meta['half_durations_minutes'][str(period)]
            for team in ['home', 'away']:
                g = series[series.team.eq(team) & series.third.eq(third) &
                           series.period.eq(period)].sort_values('period_minute')
                ax.scatter(g.period_minute, g.bcos_m2, s=9, alpha=.23,
                           color=COLORS[team], edgecolors='none',
                           label=meta['teams'][team])
                centres, means, shown, counts = temporal_display(g, duration)
                ax.plot(centres, shown, color=COLORS[team], lw=1.5, ls='--', alpha=.8)
                ax.plot(centres, means, color=COLORS[team], lw=1.8)
                for t, mean, value, count in zip(centres, means, shown, counts):
                    temporal_records.append(dict(team=team, period=period, third=third,
                        period_minute=t, window_mean_bcos_m2=mean, display_bcos_m2=value,
                        n_checkpoints=int(count), interpolated=bool(not np.isfinite(mean) and np.isfinite(value))))
            ax.set(xlim=(0, duration), ylim=(0, None),
                   title=f'{third} third · half {period}')
            ax.grid(alpha=.15)
            if j == 0:
                ax.set_ylabel('BCOS (m²)')
            if i == 2:
                ax.set_xlabel('Minutes elapsed in this half')
            if i == 0 and j == 1:
                legend = ax.legend(fontsize=9)
                for handle in legend.legend_handles:
                    handle.set_alpha(1)
    fig.suptitle('BCOS at the observed checkpoints, by carrier third', fontsize=14)
    fig.supxlabel('Dots: checkpoints · solid: 5-minute means · dashed: interpolation across empty windows',
                  fontsize=9)
    save(fig, 'bcos_match_temporal')
    pd.DataFrame(temporal_records).to_csv(DATA / 'temporal_display.csv', index=False)
    print('BCOS_DESCRIPTIVE_FIGURES_REBUILT')


if __name__ == '__main__':
    main()
