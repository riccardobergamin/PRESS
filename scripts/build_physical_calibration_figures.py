"""Render final physical-component calibration from frozen public bins."""
from pathlib import Path
import pandas as pd
try:
    from ._calibration_plot import axes, points, save
except ImportError:
    from _calibration_plot import axes, points, save

ROOT = Path(__file__).resolve().parents[1]


def build_ball_drive():
    current = pd.read_csv(ROOT/'data/benchmarks/calibration/pooled_bins.csv')
    final = current[current.task.eq('ball_drive') & current.model.eq('Aggiornato')]
    comparison = pd.read_csv(ROOT/'data/benchmarks/ball_drive/filtered_calibration_deciles.csv')
    fig, ax = axes('Ball-drive risk · pooled held-out calibration')
    ece = points(ax, final, 'predicted_loss', 'observed_loss',
                 'PRESS physical', expected_n=9114)
    expected = pd.read_csv(ROOT/'data/benchmarks/calibration/pooled_metrics.csv')
    expected = expected[expected.task.eq('ball_drive') & expected.model.eq('Aggiornato')].iloc[0]
    assert abs(ece-expected.ECE_quantile) < 1e-12
    for name, label, color, marker in [
        ('CatBoost natural', 'CatBoost · natural', '#c97930', 's'),
        ('CatBoost sampled + prior', 'CatBoost · resampled + prior correction', '#668b47', '^'),
    ]:
        selected = comparison[comparison.model.eq(name)]
        points(ax, selected, 'predicted_loss', 'observed_loss', label,
               color=color, marker=marker, expected_n=9114)
    save(fig, 'ball_drive_physical_pooled_calibration', 'ball_drive_final_calibration')


def build_pass_probability():
    bins = pd.read_csv(ROOT/'data/benchmarks/pass_risk/calibration_deciles.csv')
    selected = bins[bins.case.eq('02_vmax_5_5') & bins.subset.eq('all')
                    & bins.model.eq('trajectory_isotonic')]
    fig, ax = axes('Pass probability · pooled held-out calibration',
                   'Predicted completion probability', 'Observed completion frequency')
    ece = points(ax, selected, 'predicted_success', 'observed_success',
                 'PRESS · isotonic', expected_n=6706)
    assert abs(ece-.001833118796781196) < 1e-12
    save(fig, 'pass_benchmark_calibration', 'pass_risk_physical_pooled_calibration')


def main():
    build_ball_drive()
    build_pass_probability()
    print('Rendered final physical models and the retained CatBoost comparison as points.')


if __name__ == '__main__':
    main()
