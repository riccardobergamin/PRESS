"""Figure helpers for the real-match ball-drive vignette; no model fitting."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np
from scipy.stats import gamma, lognorm

ROOT = Path(__file__).resolve().parents[1]


def save(fig, name, root=ROOT):
    for ext in ('png', 'svg'):
        fig.savefig(Path(root)/'figures'/f'{name}.{ext}', dpi=170)
    plt.close(fig)


def build_parameter_figure(root=ROOT):
    """Compare the actual priors with frozen joint-Laplace marginals."""
    root = Path(root)
    summaries = json.loads((root/'data/examples/ball_drive/posterior_summary.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), layout='constrained')
    specs = [
        ('lambda_0', np.geomspace(.05, 8., 1600), lognorm(s=1., scale=1.),
         r'Baseline tackle rate $\lambda_0$ (s$^{-1}$)'),
        ('s', np.geomspace(.03, 2., 1600), gamma(a=2.5, scale=1/5),
         r'Arrival dispersion $s$ (s)'),
    ]
    for ax, (name, x, prior, label) in zip(axes, specs):
        item = summaries['parameters'][name]
        posterior = lognorm(s=np.sqrt(item['log_variance']), scale=np.exp(item['log_mean']))
        lo, hi = item['credible_interval_95']
        y = posterior.pdf(x)
        ax.plot(x, prior.pdf(x), '--', color='#758495', lw=1.7, label='Prior')
        ax.plot(x, y, color='#245f82', lw=2, label='Laplace posterior')
        ax.fill_between(x, 0, y, where=(x>=lo)&(x<=hi), color='#245f82', alpha=.17,
                        label='95% credible interval')
        ax.axvline(item['transformed_joint_map'], color='#ab6c22', lw=1.3,
                   label='Transformed joint MAP')
        ax.set(xscale='log', xlabel=label, ylabel='Density', ylim=(0, None))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle('Parameter uncertainty in the bundled physical model', fontsize=13)
    save(fig, 'vignette_01_drive_risk', root)


def finish_case_plot(press, case, phase, root=ROOT):
    """Add case labels to a figure already drawn by PRESS visualization.

    The source is the observed tracking row selected by the public frame getter.
    Cropping is shared between panels; no player/ball position is projected or
    changed in either observed frame. Only the start panel contains a forecast.
    """
    fig = plt.gcf()
    ax = fig.axes[0]
    frame = case['frame_id'] if phase == 'start' else case['recovery_frame']
    row = press.get_frame(frame).iloc[0]
    # A common close-up makes the physical encounter visible on the pitch.
    points = []
    for f in (case['frame_id'], case['recovery_frame']):
        observed = press.get_frame(f).iloc[0]
        for actor in case['highlighted_players']:
            points.append([observed[actor+'_x'], observed[actor+'_y']])
    points = np.asarray(points, float)
    lower, upper = points.min(axis=0), points.max(axis=0)
    ax.set_xlim(lower[0]-4.5, upper[0]+4.5)
    ax.set_ylim(lower[1]-4., upper[1]+3.)
    # Native plots draw 1.5-second velocity arrows over a full pitch. Keep
    # locally originating arrows only and use a 0.25-second drawing scale
    # in this close-up; distant players' arrows must not cross the encounter.
    x_min, x_max = sorted(ax.get_xlim())
    y_min, y_max = sorted(ax.get_ylim())
    for collection in list(ax.collections):
        if isinstance(collection, Quiver):
            visible = ((collection.X >= x_min) & (collection.X <= x_max)
                       & (collection.Y >= y_min) & (collection.Y <= y_max))
            if not visible.any():
                collection.remove()
            else:
                collection.set_UVC(collection.U/6, collection.V/6)
    # Native numeric annotations are replaced by names for these highlighted
    # actors; the native probability colors and colorbar are kept unchanged.
    if phase == 'start':
        for text in list(ax.texts):
            if ': ' in text.get_text():
                text.remove()
    current = row.player_possession
    ax.scatter([row[current+'_x']], [row[current+'_y']], marker='*', s=220,
               color='#ffe36b', edgecolor='black', linewidth=1.2, zorder=12)
    offsets = {'away_17': (12, 17), 'home_6': (13, -23), 'home_14': (-98, 20)}
    for actor, name in case['highlighted_players'].items():
        label = name
        if phase == 'start' and actor in case['cause_probabilities']:
            label += f"\nLoss contribution: {100*case['cause_probabilities'][actor]:.1f}%"
        elif actor == current:
            label += '\nIn possession'
        ax.annotate(label, (row[actor+'_x'], row[actor+'_y']),
                    xytext=offsets[actor], textcoords='offset points', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2),
                    arrowprops=dict(arrowstyle='-', color='#35414d', lw=.8), zorder=15)
    risk = 100*case['installed_probability']
    if phase == 'start':
        title = (f"Wellington Phoenix–Melbourne Victory · {case['clock']} · frame {frame}\n"
                 f"Initial state: Velupillay carrying · P(loss within 1 s) = {risk:.1f}%\n"
                 'Defender colors show their individual loss-probability contributions')
        filename = 'vignette_01_drive_pitch'
    else:
        title = (f"Same encounter · {case['recovery_clock']} · frame {frame}\n"
                 f"End state: Payne in possession · initial forecast = {risk:.1f}%\n"
                 'Observed loss after 0.3 s; opponent recovery 0.2 s later')
        filename = 'vignette_01_drive_end_pitch'
    ax.set_title(title, fontsize=10.5, pad=13, weight='normal')
    fig.subplots_adjust(top=.81, bottom=.15)
    save(fig, filename, root)
