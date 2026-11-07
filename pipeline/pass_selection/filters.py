"""Destination choice includes short passes, but excludes flights of 10 s or more."""
from pipeline.pass_probability.filters import verified_passes


def select(events, tracking, players):
    return verified_passes(events, tracking, players).query('pass_duration < 10').copy()
