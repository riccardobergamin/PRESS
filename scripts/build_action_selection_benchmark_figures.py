"""Render native action-selection calibration from frozen held-out bins."""
from pathlib import Path
import pandas as pd
try:
    from ._calibration_plot import axes, points, save
except ImportError:
    from _calibration_plot import axes, points, save

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'data/benchmarks/action_selection'
    bins = pd.read_csv(folder/'calibration_deciles.csv')
    metrics = pd.read_csv(folder/'metrics.csv').set_index('model')
    selected = bins[bins.model.eq('native_raw')].sort_values('bin')
    expected_ece = metrics.loc['native_raw', 'ece_quantile10']
    fig, ax = axes('Action selection · pooled held-out calibration',
                   'Predicted probability of passing', 'Observed pass frequency')
    ece = points(ax, selected, 'predicted', 'observed',
                 f'Native XGBoost · ECE {100*expected_ece:.3f}%', expected_n=16918)
    assert abs(ece-expected_ece) < 1e-12
    save(fig, 'action_selection_pooled_calibration')
    print('Rendered 10 native-model calibration points; no connecting lines or bar plot.')


if __name__ == '__main__':
    main()
