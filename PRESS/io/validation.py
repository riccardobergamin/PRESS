"""Normalize harmless omissions without inferring tracking or event values."""

import re

import numpy as np
import pandas as pd

from PRESS.core.metadata import MatchInfo


def prepare_tracking(table, *, fps, pitch_length, pitch_width, match_id):
    """Copy canonical data, derive optional speed norms and validate its schema."""
    from PRESS.utils.sanitizers import validate_tracking_data

    if not isinstance(table, pd.DataFrame):
        raise ValueError("tracking_data must be a pandas DataFrame.")
    if table.empty or table.columns.duplicated().any():
        raise ValueError("tracking_data must be nonempty with unique column names.")
    data = table.copy()
    for column in list(data.columns):
        if re.fullmatch(r"(home|away)_\d+_vx", column):
            player = column[:-3]
            if f"{player}_v" not in data and f"{player}_vy" in data:
                data[f"{player}_v"] = np.hypot(data[column], data[f"{player}_vy"])
    dimensions = []
    for name, explicit, default in [
        ("pitch_length", pitch_length, 105.0),
        ("pitch_width", pitch_width, 68.0),
    ]:
        if name in data:
            values = pd.to_numeric(data[name], errors="coerce")
            if not np.isfinite(values).all() or values.nunique() != 1:
                raise ValueError(f"{name} must be constant and finite within one match.")
            value = float(values.iloc[0])
            if explicit is not None and not np.isclose(explicit, value):
                raise ValueError(f"Explicit {name} conflicts with tracking dimensions.")
        else:
            value = default if explicit is None else explicit
            data[name] = value
        dimensions.append(value)
    if "match_id" in data:
        ids = data.match_id.dropna().astype(str).unique()
        if len(ids) > 1:
            raise ValueError("PRESS accepts one match at a time.")
        if len(ids):
            if match_id is not None and str(match_id) != ids[0]:
                raise ValueError("match_id conflicts with tracking data.")
            match_id = ids[0]
    info = MatchInfo(
        fps=fps,
        pitch_length=dimensions[0],
        pitch_width=dimensions[1],
        match_id=match_id,
    )
    validate_tracking_data(data)
    return data.sort_values("frame", kind="stable").reset_index(drop=True), info
