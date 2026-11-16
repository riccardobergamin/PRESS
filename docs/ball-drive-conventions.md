# Ball-drive movement and boundary conventions

The ball-drive model predicts defender arrival after a fixed-position turning delay followed by accelerated straight-line motion. The carrier continues at its observed velocity over the forecast horizon. These assumptions are shared by training and inference.

| Configuration | Meaning | Distributed model setting |
|---|---|---|
| `v_max` | Defender maximum speed during acceleration | 7 m/s |
| `a_linear` | Defender translational acceleration | 7 m/s² |
| `a_turn` | Turning-delay scale | 7 m/s² |
| `preserve_initial_overspeed` | Preserve an observed speed above the nominal maximum without further acceleration | `True` |
| `boundary_mode` | Boundary covariate convention | `signed` |

Pass physical settings through `physics_kwargs` in the ball-drive model interface. Learned coefficients, arrival dispersion and spline bases belong to the model, supplied through `model`. Changing physical settings leaves the fitted model and the other components unchanged. See [solver parameters](../solvers.qmd#named-model-parameters).

The low-level `time_to_intercept` routine calls translational acceleration `a_max`. Its inputs are position, velocity and target, and its turning delay is `speed * theta / a_turn`; the default `a_turn=None` uses `a_max`. The delay is a time cost applied before straight-line motion.

```python
import jax.numpy as jnp
from PRESS.physics.times import time_to_intercept

tau = time_to_intercept(
    p=jnp.array([[0., 0.]]), v=jnp.array([[8., 0.]]),
    r_target=jnp.array([[16., 0.]]),
    v_max=7., a_max=7., a_turn=7.,
    preserve_initial_overspeed=True,
)
# Approximately 2 s when the observed 8 m/s speed is preserved.
```

For a centred rectangle of length L and width W, the signed boundary margin is
`min(x+L/2, L/2-x, y+W/2, W/2-y)`: positive inside and negative outside. The model also accepts `absolute`, which takes the minimum absolute distance to the boundary lines. The fitted spline uses the signed convention, which is the one validated by the published benchmark.

The fitted tackle intensity includes carrier speed, boundary distance and defender approach angle. Its coefficients and spline transformations are described in the [model vignette](../vignettes/01_ball_drive_risk/01_ball_drive_risk.ipynb). The [kinematic parameter notes](kinematic-parameter-conventions.md) explain the literature and descriptive tracking checks supporting the movement settings.
