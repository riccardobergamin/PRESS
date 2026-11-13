"""Final ball-drive cohort: whole-parent verification and one-second labels."""
import pandas as pd


def verified_loss_parents(events):
    """A valid associated opponent OBE must overlap the carrier's possession."""
    parents = events.loc[events.event_type.eq('player_possession')]
    obe = events.loc[events.event_type.eq('on_ball_engagement')]
    valid = set()
    for p in parents.loc[parents.end_type.eq('possession_loss')].itertuples(index=False):
        e = obe.loc[obe.associated_player_possession_event_id.eq(p.event_id) & obe.player_in_possession_id.eq(p.player_id) & obe.team_id.ne(p.team_id) & obe.period.eq(p.period) & obe.frame_start.le(p.frame_end) & obe.frame_end.ge(p.frame_start)]
        if len(e):
            valid.add(p.event_id)
    return valid


def parent_mask(rows, events):
    """An unverified terminal direct loss excludes every drive of that parent."""
    return rows.end_type.ne('possession_loss') | rows.possession_id.isin(verified_loss_parents(events))


def select(rows, events):
    mask = rows.state_eligible & rows.training_cohort_eligible & rows.horizon_label_observed
    return rows.loc[mask & parent_mask(rows, events)].copy()
