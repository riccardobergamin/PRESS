# External data preparation

PRESS itself needs canonical tracking and synchronized event links. The
standalone IDSSE example keeps provider loading and optional DataBallPy
synchronization outside the installed inference package. Segmentation and
context reconstruction use `PRESS.io` and `PRESS(...)`.

1. Obtain the Sportec XML files for `J03WMX` from the official IDSSE dataset.
2. Create a separate Python 3.10 environment and install
   `databallpy==0.7.3` and `pyarrow`. This avoids changing PRESS dependencies.
3. Run:

```sh
python examples/preprocessing/databallpy_bridge.py DATA_DIR J03WMX OUTPUT_DIR/aligned
```

4. In the PRESS environment:

```python
from examples.preprocessing.idsse import prepare_idsse
from PRESS import PRESS
from PRESS.io import segment_possessions

tracking, events, players, metadata = prepare_idsse(DATA_DIR, "J03WMX", OUTPUT_DIR)
tracking, individuals, actions, report = segment_possessions(
    tracking, events, fps=metadata["fps"], match_id="J03WMX",
)
press = PRESS(
    tracking, player_possessions_data=individuals, actions_data=actions,
    fps=metadata["fps"], match_id="J03WMX", player_metadata=players,
    pitch_length=metadata["pitch_length"], pitch_width=metadata["pitch_width"],
)
```

`DATA_DIR` and `OUTPUT_DIR` are caller-supplied paths. Alignment is reused only
when explicitly supplied in that output directory. The original XML is never
modified. See `external-data.qmd` for required fields and uncertainty limits.
Data: Deutsche Fußball Liga (DFL), CC BY 4.0; Bassek et al. (2025),
https://doi.org/10.1038/s41597-025-04505-y; official repository:
https://github.com/spoho-datascience/idsse-data. DataBallPy is MIT-licensed;
cite Oonk, Grob and Kempe (2026), https://doi.org/10.21105/joss.10223.
