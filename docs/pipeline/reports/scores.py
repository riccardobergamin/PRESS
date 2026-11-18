"""Pattern shares, exposure, E-O/100 and fixed-prediction uncertainty."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from pipeline.io import PARENT_KEYS
from pipeline.composed_risk.scores import THRESHOLDS, BAND_NAMES, SEED, REPLICATES


PATTERNS = [
    ("Pressure escape", "escape_eligible", "pressure_escape"),
    ("Drawing pressure", "drawing_eligible", "drawing_pressure"),
    ("Sustained successful carrying", "sustained_eligible", "sustained_successful_carrying"),
]


def wilson_interval(positive, total):
    """Descriptive 95% binomial interval on distinct possessions."""
    if total == 0:
        return np.nan, np.nan
    z = norm.ppf(.975)
    p, denominator = positive/total, 1+z*z/total
    centre = (p+z*z/(2*total))/denominator
    width = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return centre-width, centre+width


def pattern_shares(possessions):
    """Report BOTH cohort shares and shares in each pattern's eligible set."""
    rows = []
    for name, eligibility, occurrence in PATTERNS:
        eligible = possessions.loc[possessions[eligibility]]
        positive = int(eligible[occurrence].sum())
        lo, hi = wilson_interval(positive, len(eligible))
        cohort_lo, cohort_hi = wilson_interval(positive, len(possessions))
        rows.append(dict(pattern=name, eligible_possessions=len(eligible), positive=positive,
            share=positive/len(eligible) if len(eligible) else np.nan, lower95=lo, upper95=hi,
            cohort_possessions=len(possessions), cohort_share=positive/len(possessions),
            cohort_lower95=cohort_lo, cohort_upper95=cohort_hi))
    return pd.DataFrame(rows)


def bootstrap_entity(actions, entity_id, entity_kind, replicates=REPLICATES):
    """Possession blocks: draw all elevated actions of each sampled possession.

    Each entity gets a deterministic stream, independent of table sorting.
    No match bootstrap is used for the primary score: players have 1-4 matches.
    """
    blocks = actions.groupby(PARENT_KEYS, sort=True).agg(
        n=("h", "size"), expected=("h", "sum"), observed=("h_loss", "sum"), high=("high", "sum"))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, entity_kind, int(entity_id)]))
    draws = rng.integers(0, len(blocks), size=(replicates, len(blocks)))
    n = blocks.n.to_numpy()[draws].sum(axis=1)
    expected = blocks.expected.to_numpy()[draws].sum(axis=1)
    observed = blocks.observed.to_numpy()[draws].sum(axis=1)
    high = blocks.high.to_numpy()[draws].sum(axis=1)
    statistics = dict(eo_per100=100*(expected-observed)/n, mean_h=expected/n,
                      observed_failure=observed/n, high_action_share=high/n)
    return {name: np.percentile(values, [2.5, 97.5]) for name, values in statistics.items()}


def entity_score(states, possessions, entity_id, kind):
    """E-O=100*mean(h-y), restricted to states with h>=Medium.

    No exposure -> undefined score. Intervals require >=20 exposed possessions.
    Exposure/retention intervals are Wilson, not the score's block bootstrap.
    """
    actions = states.loc[states.h.ge(THRESHOLDS[1])]
    n, blocks = len(actions), len(actions[PARENT_KEYS].drop_duplicates())
    row = dict(n_actions_total=len(states), n_possessions=len(possessions),
        n_matches=int(possessions.match_id.nunique()), n_exposed_matches=int(actions.match_id.nunique()),
        n_pressure_actions=n, n_pressure_possessions=blocks,
        expected_failures=float(actions.h.sum()), observed_failures=int(actions.h_loss.sum()),
        mean_h=actions.h.mean(), observed_failure=actions.h_loss.mean(),
        high_action_share=actions.high.mean(), eo_per100=100*(actions.h-actions.h_loss).mean(),
        score_interval_reported=blocks >= 20)
    for name in ["eo_per100", "mean_h", "observed_failure", "high_action_share"]:
        row[name+"_lower95"], row[name+"_upper95"] = np.nan, np.nan
    if blocks >= 20:
        for name, (lo, hi) in bootstrap_entity(actions, entity_id, 0 if kind == "player" else 1).items():
            row[name+"_lower95"], row[name+"_upper95"] = float(lo), float(hi)
    exposed = possessions.loc[possessions.elevated]
    retained = int(exposed.retained.sum())
    lo, hi = wilson_interval(len(exposed), len(possessions))
    rlo, rhi = wilson_interval(retained, len(exposed))
    row.update(n_pressure_retained=retained,
        pressure_exposure_pct=100*len(exposed)/len(possessions), exposure_lower95=100*lo, exposure_upper95=100*hi,
        retention_pct=100*retained/len(exposed) if len(exposed) else np.nan,
        retention_lower95=100*rlo, retention_upper95=100*rhi,
        n_high_possessions=int(possessions.high.sum()))
    return row


def compute_reports(states, possessions, output):
    """Player/team tables, frequencies, endpoint transitions and sensitivity."""
    output = output / "scores"
    output.mkdir(parents=True, exist_ok=False)
    pattern_shares(possessions).to_csv(output / "pattern_counts.csv", index=False)
    for kind, identifier in [("player", "provider_player_id"), ("team", "team_id")]:
        rows, patterns = [], []
        for entity, parents in possessions.groupby(identifier, sort=True):
            group = states.loc[states[identifier].eq(entity)]
            identity = {identifier: entity, "team_name": "/".join(sorted(parents.team_name.unique()))}
            if kind == "player":
                identity["player_name"] = parents.player_name.iloc[0]
            rows.append(identity | entity_score(group, parents, entity, kind))
            shares = pattern_shares(parents)
            for key, value in identity.items():
                shares[key] = value
            patterns.append(shares)
        pd.DataFrame(rows).to_csv(output / f"{kind}_summary.csv", index=False)
        pd.concat(patterns, ignore_index=True).to_csv(output / f"{kind}_pattern_shares.csv", index=False)
    peak = possessions.groupby(["team_name", "peak_band"]).size()
    index = pd.MultiIndex.from_product([sorted(possessions.team_name.unique()), BAND_NAMES], names=["team_name", "peak_band"])
    peak.reindex(index, fill_value=0).rename("n_possessions").reset_index().to_csv(output / "team_peak_bands.csv", index=False)
    initial = possessions.loc[possessions.has_initial].copy()
    for column, value in [("start_band", "start_h"), ("end_band", "end_h")]:
        initial[column] = np.array(BAND_NAMES)[np.searchsorted(THRESHOLDS, initial[value], side="right")]
    initial.groupby(["start_band", "end_band"]).agg(n=("retained", "size"), retained=("retained", "sum")).reset_index().to_csv(output / "endpoint_transitions.csv", index=False)
    # Team match bootstrap is sensitivity only, never the primary interval.
    sensitivity = []
    for team, actions in states.loc[states.h.ge(THRESHOLDS[1])].groupby("team_id", sort=True):
        blocks = actions.groupby("match_id").agg(n=("h", "size"), expected=("h", "sum"), observed=("h_loss", "sum"))
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, int(team)]))
        draw = rng.integers(len(blocks), size=(REPLICATES, len(blocks)))
        score = 100*(blocks.expected.to_numpy()[draw].sum(1)-blocks.observed.to_numpy()[draw].sum(1))/blocks.n.to_numpy()[draw].sum(1)
        lo, hi = np.percentile(score, [2.5,97.5])
        sensitivity.append(dict(team_id=team, team_name=actions.team_name.iloc[0], n_matches=len(blocks),
            eo_per100=100*(actions.h-actions.h_loss).mean(), match_bootstrap_lower95=lo, match_bootstrap_upper95=hi))
    pd.DataFrame(sensitivity).to_csv(output / "team_match_bootstrap_sensitivity.csv", index=False)
