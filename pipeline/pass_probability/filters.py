"""Verified release-to-control links; only pass probability requires 0.5–10 s."""
import numpy as np
import pandas as pd
from pipeline.library.preprocessing.passes import extract_passes
from pipeline.library.preprocessing.tracking import annotate_training_cohort, pass_curvature


def verified_passes(events, tracking, players, *, minimum=None, maximum=None):
    rows = extract_passes(events)
    raw = events.loc[events.event_type.eq('player_possession')]
    keys = ['frame_start', 'frame_end', 'player_id']
    rows = rows.merge(raw[keys + ['event_id', 'period', 'end_type', 'start_type']],
        on=keys,
        validate='one_to_one')
    # Preserve the accepted numeric choice key; event_id remains the provider ID.
    rows['pass_id'] = rows.event_id.map(lambda value: int(str(value).replace('_', '')))
    rows['match_id'] = str(events.match_id.iloc[0])
    rows = annotate_training_cohort(rows, players)
    rows = rows.loc[rows.training_cohort_eligible & rows.pass_duration.gt(0)].copy()
    if minimum is not None:
        rows = rows.loc[rows.pass_duration.ge(minimum)]
    if maximum is not None:
        rows = rows.loc[rows.pass_duration.le(maximum)]
    index = tracking.set_index('frame')
    valid = []
    for p in rows.itertuples(index=False):
        path = index.reindex(np.arange(int(p.frame_end), int(p.frame_reception) + 1))
        valid.append(bool(index.state_eligible.get(p.frame_end, False) and path.period.eq(p.period).all() and np.isfinite(path[['ball_x', 'ball_y']].to_numpy()).all()))
    rows = rows.loc[valid].copy()
    rows['curvature_pct'] = [pass_curvature(p, index) for p in rows.itertuples(index=False)]
    return rows.sort_values('pass_id').reset_index(drop=True)
