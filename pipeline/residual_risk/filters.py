"""Successful full carries with a verified parent outcome; no future predictors."""
from pipeline.ball_drive.filters import parent_mask
from pipeline.pass_probability.filters import verified_passes
from pipeline.action_selection.filters import select as choices


def select(actions, events, tracking, players):
    rows=choices(actions)
    rows=rows.loc[rows.action.eq('ball_drive') & rows.outcome.eq(1) & rows.duration_frames.eq(10)]
    passed=verified_passes(events,tracking,players)
    valid=rows.end_type.ne('pass') | rows.possession_id.isin(passed.pass_id)
    return rows.loc[valid & parent_mask(rows,events)].copy()
