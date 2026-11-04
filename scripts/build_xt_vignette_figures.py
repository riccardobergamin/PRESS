"""Plot the exact ten-match xT surface embedded in the deployed models."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
from scripts._pitch_heatmap import draw_cell_map

ROOT = Path(__file__).resolve().parents[1]


def plot_xt_surface(grid):
    """Return a conventional pitch heatmap without interpolating the cells."""
    grid = np.asarray(grid, float)
    fig, ax = plt.subplots(figsize=(10.5, 7), layout='constrained')
    mesh = draw_cell_map(ax, grid, norm=Normalize(0, float(grid.max())), cmap='magma')
    ax.set_title('Expected Threat · production fit on ten matches\n10 × 6 cells · attack →',
                 fontsize=13, pad=12)
    cb = fig.colorbar(mesh, ax=ax, shrink=.8, pad=.025,
                      format=PercentFormatter(1, decimals=0))
    cb.set_label('xT')
    return fig, ax


def main():
    artifact = json.loads((ROOT / 'PRESS/utils/data/pass_selection_model.json').read_text())
    grid = np.asarray(artifact['xt_grid'], float)
    assert grid.shape == (10, 6) and np.isfinite(grid).all()
    saved = np.load(ROOT / 'data/examples/expected_threat/xt_grid.npy')
    np.testing.assert_array_equal(grid, saved)
    fig, _ = plot_xt_surface(grid)
    for ext in ['png', 'svg']:
        fig.savefig(ROOT / f'figures/expected_threat_production.{ext}', dpi=180)
    plt.close(fig)
    print('XT_PRODUCTION_FIGURE_REBUILT')


if __name__ == '__main__':
    main()
