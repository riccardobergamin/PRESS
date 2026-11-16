# Ball-drive risk

The deployed physical model and the CatBoost comparison share the same observed carry cohort. Read [TRAINING.md](TRAINING.md) for estimation, validation and commands.

`ball_drive.py` is a self-contained prediction implementation using the fitted spline artifact and PRESS movement routines. It evaluates the stored spline transformations and integrates competing defender risks without estimating coefficients or knot locations. The package's public inference interface is `BallDriveModel`.

Training code is organized as `physical/model.py`, `physical/training.py` and `physical/pipeline.py`. The [CatBoost directory](catboost/README.md) follows the same features/training/pipeline structure. Numerical inputs are specified in [the input guide](../inputs/README.md).
