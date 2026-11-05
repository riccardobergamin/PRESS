"""Render primary residual and composed-risk reliability plots from public bins."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from ._calibration_plot import axes, points, save
except ImportError:
    from _calibration_plot import axes, points, save

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'data/benchmarks/residual_risk'
    bins = pd.read_csv(folder/'calibration_quantile10.csv')
    selected = bins[bins.variant.eq('with_action_choice') & bins.method.eq('native')]
    metrics = pd.read_csv(folder/'pooled_metrics.csv')
    expected = metrics[metrics.variant.eq('with_action_choice') & metrics.method.eq('native')].iloc[0]
    fig, ax = axes('Residual risk · pooled held-out calibration',
                   'Predicted terminal loss probability', 'Observed terminal loss frequency')
    ece = points(ax, selected, 'predicted', 'observed', 'PRESS residual risk', expected_n=7820)
    assert abs(ece-expected.ece_quantile10) < 1e-12
    save(fig, 'residual_risk_pooled_calibration')

    folder = ROOT/'data/benchmarks/composed_risk'
    bins = pd.read_csv(folder/'calibration_quantile10.csv')
    metrics = pd.read_csv(folder/'pooled_metrics.csv')
    fig, panels = plt.subplots(1, 2, figsize=(11.4, 5.5), layout='constrained')
    for ax, model, title, xlabel, ylabel in [
        (panels[0], 'h', 'Next-action hazard h', 'Predicted action-failure probability', 'Observed action-failure frequency'),
        (panels[1], 'xloss_with_action_choice', 'Individual xLoss', 'Predicted terminal loss probability', 'Observed terminal loss frequency'),
    ]:
        axes(title, xlabel, ylabel, ax=ax)
        cohort = 'verified_endpoints'
        selected = bins[bins.cohort.eq(cohort) & bins.model.eq(model)]
        expected = metrics[metrics.cohort.eq(cohort) & metrics.model.eq(model)].iloc[0]
        assert int(expected.n) == 15085
        ece = points(ax, selected, 'predicted', 'observed', 'PRESS', expected_n=15085)
        assert abs(ece-expected.ece_quantile10) < 1e-12
    save(fig, 'composed_risk_pooled_calibration')
    print('Rendered only the primary residual-risk, h and xLoss series as points.')


if __name__ == '__main__':
    main()
