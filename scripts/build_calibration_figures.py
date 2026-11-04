"""Build the portfolio's point-only calibration figures from frozen bins."""
try:
    from ._calibration_plot import AUDIT
    from .build_physical_calibration_figures import build_ball_drive, build_pass_probability
    from .build_action_selection_benchmark_figures import main as build_action
    from .build_residual_risk_benchmark_figures import main as build_residual_and_composed
except ImportError:
    from _calibration_plot import AUDIT
    from build_physical_calibration_figures import build_ball_drive, build_pass_probability
    from build_action_selection_benchmark_figures import main as build_action
    from build_residual_risk_benchmark_figures import main as build_residual_and_composed


def main():
    AUDIT.clear()
    build_ball_drive()
    build_pass_probability()
    build_action()
    build_residual_and_composed()
    assert len(AUDIT) == 8
    print('Rendered six calibration panels / eight model series / 80 quantile-bin points.')
    return list(AUDIT)


if __name__ == '__main__':
    main()
