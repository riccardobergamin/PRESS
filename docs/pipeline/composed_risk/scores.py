"""Pooled CE/gain/ECE and match-block bootstrap of fixed absolute h bands."""

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.io import new_folder, read_table, write_json
from pipeline.binary_classification.training import binary_metrics


BAND_NAMES = ["None", "Low", "Medium", "High"]
# Exact half-life thresholds: 8, 4, 2 equivalent canonical actions.
# Their displayed values are 0.082996, 0.159104, 0.292893.
THRESHOLDS = 1 - 2**(-1 / np.array([8., 4., 2.]))
SEED = 20261030
REPLICATES = 2000


def compute_metrics(states, output):
    """Score both targets on ALL verified states, then stratify by fold/match.

    CE and ECE are recalculated after pooling, not averaged across folds.
    ECE uses ten equal-count rank bins, as in the accepted binary benchmark.
    h bands use a different cohort and are deliberately not applied here.
    """
    folder = new_folder(Path(output) / "scores")
    records, calibration_records = [], []
    for target, probability, null in [("h_loss", "h", "null_h"),
                                       ("terminal_loss", "xloss", "null_xloss")]:
        for grouping in ["pooled", "fold", "match_id"]:
            groups = [("all", states)] if grouping == "pooled" else states.groupby(grouping, sort=True)
            for group, table in groups:
                metrics, bins = binary_metrics(table[target], table[probability], table[null])
                records.append(dict(model=probability, grouping=grouping, group=group, **metrics))
                if grouping == "pooled":
                    bins["model"] = probability
                    calibration_records.append(bins)
    scores = pd.DataFrame(records)
    scores.loc[scores.grouping.eq("pooled")].to_csv(folder / "pooled_metrics.csv", index=False)
    scores.loc[scores.grouping.eq("fold")].to_csv(folder / "metrics_by_fold.csv", index=False)
    scores.loc[scores.grouping.eq("match_id")].to_csv(folder / "metrics_by_match.csv", index=False)
    pd.concat(calibration_records, ignore_index=True).to_csv(folder / "calibration.csv", index=False)
    return scores


def bootstrap_bands(states, output, seed=SEED, replicates=REPLICATES):
    """Draw ten entire matches with replacement; keep predictions/cutoffs fixed.

    Each replicate computes ratios of summed counts, not means of match rates.
    Empty-band ratios are NaN and omitted only for that band's percentiles;
    their count and nearly-empty counts (<50 states) are always reported.
    The same draws provide a paired interval for mean(h)-observed failures.
    """
    folder = new_folder(Path(output) / "bands")
    states = states.loc[states.band_eligible].copy()
    states["band"] = np.searchsorted(THRESHOLDS, states.h, side="right")
    # Preserve accepted encounter order, and save it with the draws for replay.
    matches = states.match_id.unique()
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(matches), size=(replicates, len(matches)), replace=True)
    multiplicities = np.stack([(draws == i).sum(axis=1) for i in range(len(matches))], axis=1)
    pd.DataFrame(multiplicities, columns=matches).to_csv(folder / "match_multiplicities.csv", index=False)
    summaries, replicated, by_match = [], [], []
    for band, name in enumerate(BAND_NAMES):
        rows = states.loc[states.band.eq(band)]
        totals = rows.groupby("match_id", sort=False).agg(
            n=("h", "size"), predicted=("h", "sum"), failures=("h_loss", "sum")).reindex(matches, fill_value=0)
        n = multiplicities @ totals.n.to_numpy()
        predicted = np.divide(multiplicities @ totals.predicted.to_numpy(), n,
                              out=np.full(replicates, np.nan), where=n > 0)
        observed = np.divide(multiplicities @ totals.failures.to_numpy(), n,
                             out=np.full(replicates, np.nan), where=n > 0)
        intervals = {}
        for key, values in [("mean_h", predicted), ("observed", observed), ("gap", predicted-observed)]:
            lo, hi = np.nanpercentile(values, [2.5, 97.5]) if np.isfinite(values).any() else [np.nan, np.nan]
            intervals[key + "_lower95"], intervals[key + "_upper95"] = lo, hi
        summaries.append(dict(band=name, n=len(rows), failures=int(rows.h_loss.sum()),
            mean_h=rows.h.mean(), observed_failure=rows.h_loss.mean(), gap=(rows.h-rows.h_loss).mean(),
            **intervals, bootstrap_count_min=int(n.min()), bootstrap_count_median=float(np.median(n)),
            bootstrap_count_max=int(n.max()), empty_replicates=int((n == 0).sum()),
            near_empty_replicates=int((n < 50).sum())))
        replicated.append(pd.DataFrame(dict(replicate=np.arange(replicates), band=name,
                                            n=n, mean_h=predicted, observed_failure=observed)))
        by_match.append(totals.reset_index().assign(band=name))
    summary = pd.DataFrame(summaries)
    summary.to_csv(folder / "summary.csv", index=False)
    pd.concat(by_match, ignore_index=True).to_csv(folder / "totals_by_match.csv", index=False)
    pd.concat(replicated, ignore_index=True).to_parquet(folder / "replicates.parquet", index=False)
    write_json(folder / "procedure.json", dict(seed=seed, replicates=replicates,
        thresholds=THRESHOLDS, percentile_bounds=[2.5,97.5], n_states=len(states),
        very_high_states=int(states.h.ge(.5).sum()), matches=matches,
        unit="whole match", fixed_predictions=True, model_uncertainty_included=False,
        empty_policy="Omit undefined ratios from that band's percentiles; report their count."))
    return summary
