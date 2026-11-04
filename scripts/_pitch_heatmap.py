"""Shared, unsmoothed cell rendering for the descriptive xT and BCOS pages."""
import numpy as np
from mplsoccer import Pitch


def draw_cell_map(ax, values, *, norm, cmap, counts=None):
    """Draw an (n_x, n_y) array on the actual cell edges, attacking right."""
    values = np.asarray(values, float)
    nx, ny = values.shape
    pitch = Pitch(pitch_type='skillcorner', pitch_length=105, pitch_width=68,
                  pitch_color='#e9edf1', line_color='white', linewidth=1.1,
                  line_zorder=4)
    pitch.draw(ax=ax)
    mesh = ax.pcolormesh(np.linspace(-52.5, 52.5, nx + 1),
                         np.linspace(-34, 34, ny + 1), values.T,
                         cmap=cmap, norm=norm, shading='flat',
                         edgecolors=(1, 1, 1, .22), linewidth=.35, zorder=1)
    if counts is not None:
        for ix, iy in np.ndindex(values.shape):
            if np.isfinite(values[ix, iy]):
                ax.text(-52.5 + (ix + .5) * 105 / nx,
                        -34 + (iy + .5) * 68 / ny, str(int(counts[ix, iy])),
                        ha='center', va='center', fontsize=8, zorder=5,
                        color='white' if norm(values[ix, iy]) < .55 else '#17212b')
    return mesh
