"""Render the selected isotonic model's frozen held-out calibration bins."""
try:
    from .build_physical_calibration_figures import build_pass_probability
except ImportError:
    from build_physical_calibration_figures import build_pass_probability


def main():
    build_pass_probability()
    print('Rendered selected pass-probability calibration as points.')


if __name__ == '__main__':
    main()
