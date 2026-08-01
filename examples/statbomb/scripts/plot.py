import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mplsoccer import Pitch

def plot_xt(xt_grid):
    # 1. Setup Pitch (StatsBomb coords: 120x80)
    pitch = Pitch(pitch_type='statsbomb', pitch_color="#22312b", line_color="white")
    fig, ax = pitch.draw(figsize=(13, 8))
    fig.set_facecolor('#1a1a1a')

    # 2. Coordinate della griglia originale (12x16)
    # xT.shape è (12, 16) -> 12 righe (y), 16 colonne (x)
    ny, nx = xt_grid.shape
    x_coords = np.linspace(0, 120, nx)
    y_coords = np.linspace(0, 80, ny)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # 3. Griglia ad alta risoluzione per lo smoothing
    xi = np.linspace(0, 120, 300)
    yi = np.linspace(0, 80, 300)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # 4. Interpolazione Cubic
    points = np.column_stack((xx.ravel(), yy.ravel()))
    values = xt_grid.ravel()
    
    xt_smooth = griddata(points, values, (xi_grid, yi_grid), method='cubic')

    # 5. Plot con contourf
    levels = np.linspace(0, np.nanmax(xt_smooth), 100)
    cntr = ax.contourf(xi_grid, yi_grid, xt_smooth, levels=levels, 
                       cmap="hot", alpha=0.6, zorder=1)

    # 6. Colorbar e rifiniture
    cbar = fig.colorbar(cntr, ax=ax, fraction=0.02, pad=0.04)
    cbar.outline.set_edgecolor('white')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    plt.show()
