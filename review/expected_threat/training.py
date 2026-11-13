"""xT smoothing selection, held-out component scores and final surface fits.

Read: component_scores -> paired_match_difference -> select_bandwidth
-> fit_on_matches -> validate_grid -> collect_validation.
There is no learned tree ensemble: fitting means empirical component estimates
and Bellman iteration; validation chooses their smoothing bandwidths.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import (PITCH_LENGTH, PITCH_WIDTH, EVENT_TYPES, bin_events,
                   transition_probabilities, goal_probabilities, fit_xt, spatial_cells)


BANDWIDTHS_M = [0., 3., 5., 8., 12., 18.]


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


# 1. Scores for the individual estimated components

def component_scores(validation, fitting, component, grid, bandwidth_m):
    """Evaluate an observed destination or goal outcome with fitting quantities.

    Pass/carry: -log(T_origin,target / cell_area), a piecewise-uniform spatial
    density score. Shot: Bernoulli CE of goal given shot, with clipping 1e-6.
    Transition scores are not Bernoulli losses and depend on the area unit.
    """
    nx, ny = grid
    if component == "shots":
        goal_grid = goal_probabilities(fitting, grid, bandwidth_m)
        probability = np.clip(goal_grid[validation.origin], 1e-6, 1. - 1e-6)
        goal = validation.goal.to_numpy(float)
        nll = -(goal * np.log(probability) + (1. - goal) * np.log1p(-probability))
    else:
        transitions = transition_probabilities(fitting, grid, bandwidth_m)
        probability = transitions[validation.origin, validation.target]
        cell_area = (PITCH_LENGTH / nx) * (PITCH_WIDTH / ny)
        nll = -np.log(probability / cell_area)
    return nll, probability


def paired_match_difference(differences, event_counts):
    """Event-weighted paired mean difference and delete-one-MATCH jackknife SE.

    Input differences d_m are mean-score differences on the SAME match.
    Delta=sum(n_m*d_m)/sum(n_m).
    Delta_(-m)=(sum(n*d)-n_m*d_m)/(sum(n)-n_m).
    SE=sqrt((M-1)/M * sum_m (Delta_(-m)-mean(Delta_(-m)))**2).
    Predictions stay fixed; there is no model refit inside this jackknife.
    """
    differences = np.asarray(differences, float)
    weights = np.asarray(event_counts, float)
    if len(weights) < 2 or (weights <= 0).any():
        raise ValueError("At least two nonempty match groups are required.")
    weighted_total = np.sum(weights * differences)
    mean_difference = np.average(differences, weights=weights)
    leave_one_out = (weighted_total - weights * differences) / (weights.sum() - weights)
    n_matches = len(weights)
    standard_error = np.sqrt((n_matches - 1) / n_matches
                             * np.sum((leave_one_out - leave_one_out.mean())**2))
    return float(mean_difference), float(standard_error)


# 2. Leave-one-match-out bandwidth selection, using ONLY the specified fit matches.

def select_bandwidth(events, component, fitting_matches, grid):
    """Select the largest bandwidth within one paired SE of the best score.

    For each candidate h and each fitting match m, estimate from all other
    fitting matches and score m. Pool with weights proportional to event counts.
    Compare every h to the minimum-score h using paired match differences.
    This is a smoothing/parsimony rule, not a test of statistical equivalence.
    """
    fitting_matches = sorted(fitting_matches)
    records = []
    for held_match in fitting_matches:
        train = events.loc[events.match_id.isin(fitting_matches) & events.match_id.ne(held_match)]
        validation = events.loc[events.match_id.eq(held_match)]
        if validation.empty:
            continue
        for bandwidth in BANDWIDTHS_M:
            losses, _ = component_scores(validation, train, component, grid, bandwidth)
            records.append(dict(match_id=held_match, bandwidth=bandwidth,
                                n=len(losses), nll=float(losses.mean())))
    scores = pd.DataFrame(records)
    pooled = scores.groupby("bandwidth", sort=True).apply(
        lambda group: np.average(group.nll, weights=group.n), include_groups=False,
    )
    best_bandwidth = float(pooled.idxmin())
    reference = scores.loc[scores.bandwidth.eq(best_bandwidth)].set_index("match_id")
    comparisons = []
    for bandwidth in BANDWIDTHS_M:
        candidate = scores.loc[scores.bandwidth.eq(bandwidth)].set_index("match_id").reindex(reference.index)
        delta, se = paired_match_difference(candidate.nll - reference.nll, candidate.n)
        comparisons.append(dict(bandwidth=bandwidth, pooled_nll=float(pooled.loc[bandwidth]),
                                delta_from_best=delta, paired_se=se, admitted=delta <= se + 1e-12))
    comparison = pd.DataFrame(comparisons)
    chosen = float(comparison.loc[comparison.admitted, "bandwidth"].max())
    comparison["selected"] = comparison.bandwidth.eq(chosen)
    return chosen, scores, comparison


def fit_on_matches(binned_tables, fitting_matches, grid):
    """Select three bandwidths within the fitting set, then fit its xT surface.

    Used for eight-match benchmark fits, six-match downstream feature maps and
    the all-ten production fit. No validation/test events enter these estimates.
    """
    tables = {kind: events.loc[events.match_id.isin(fitting_matches)]
              for kind, events in binned_tables.items()}
    bandwidths, tuning_scores, comparisons = {}, {}, {}
    for component in ["passes", "carries", "shots"]:
        bandwidth, scores, comparison = select_bandwidth(tables[component], component, fitting_matches, grid)
        bandwidths[component] = bandwidth
        tuning_scores[component] = scores
        comparisons[component] = comparison
    fitted = fit_xt(tables, grid, bandwidths)
    return fitted, bandwidths, tuning_scores, comparisons


def save_fit(folder, fitted, grid, fitting_matches, bandwidths, tuning_scores, comparisons):
    """Save xT and ALL components together, plus the actual smoothing selection."""
    folder.mkdir(parents=True, exist_ok=False)
    for name, value in fitted.items():
        if isinstance(value, np.ndarray):
            np.save(folder / (name + ".npy"), value)
    for component in tuning_scores:
        tuning_scores[component].to_csv(folder / f"{component}_validation_scores.csv", index=False)
        comparisons[component].to_csv(folder / f"{component}_bandwidth_comparison.csv", index=False)
    write_json(folder / "manifest.json", dict(
        grid=list(grid), fitting_matches=sorted(fitting_matches), bandwidth_m=bandwidths,
        iterations=fitted["iterations"], bellman_residual=fitted["bellman_residual"],
        selection="largest bandwidth within one paired-match SE of minimum LOMO score",
    ))


# 3. Five outer 8+2 splits for ONE grid. Every event is scored once out of sample.

def validate_grid(event_tables, test_pairs, grid, output):
    """Validate smoothed and unsmoothed components, holding out complete matches.

    Outer train: eight matches. Inner bandwidth selection: leave one of those
    eight out. Outer test: two matches, opened only after bandwidth selection.
    This protocol is separate from the six-match maps used by downstream tasks.
    """
    tables = bin_events(event_tables, grid)
    matches = sorted(set.union(*(set(table.match_id) for table in tables.values())))
    folder = output / "validation" / f"{grid[0]}x{grid[1]}"
    folder.mkdir(parents=True, exist_ok=False)
    predictions, selections, surfaces = [], [], []

    # Common physical evaluation points for comparing map variability across grids.
    xx, yy = np.meshgrid(np.linspace(-51.25, 51.25, 42), np.linspace(-32.75, 32.75, 28), indexing="ij")
    sample_cells = spatial_cells(xx.ravel(), yy.ravel(), grid)

    for fold, test_matches in sorted(test_pairs.items()):
        fitting_matches = sorted(set(matches) - set(test_matches))
        fitted, bandwidths, tuning, comparisons = fit_on_matches(tables, fitting_matches, grid)
        save_fit(folder / f"fold_{fold}", fitted, grid, fitting_matches, bandwidths, tuning, comparisons)
        surfaces.append(fitted["xt"].ravel()[sample_cells])
        for component in ["passes", "carries", "shots"]:
            train = tables[component].loc[tables[component].match_id.isin(fitting_matches)]
            test = tables[component].loc[tables[component].match_id.isin(test_matches)]
            selections.append(dict(fold=int(fold), component=component, bandwidth_m=bandwidths[component],
                                   fitting_matches=fitting_matches, test_matches=test_matches))
            for version, bandwidth in [("unsmoothed", 0.), ("selected_smoothing", bandwidths[component])]:
                nll, probability = component_scores(test, train, component, grid, bandwidth)
                result = pd.DataFrame(dict(
                    grid=f"{grid[0]}x{grid[1]}", fold=int(fold), component=component, version=version,
                    match_id=test.match_id.to_numpy(), row_index=test.index.to_numpy(),
                    nll=nll, probability=probability,
                ))
                result["goal"] = test.goal.to_numpy(bool) if component == "shots" else False
                result["brier"] = ((probability - test.goal.to_numpy())**2 if component == "shots" else np.nan)
                predictions.append(result)
        print("XT GRID", grid, "FOLD", fold, bandwidths, flush=True)

    held_out = pd.concat(predictions, ignore_index=True)
    for (component, version), rows in held_out.groupby(["component", "version"]):
        if len(rows) != len(tables[component]) or rows.row_index.duplicated().any():
            raise ValueError("Each component event must appear exactly once in the outer test.")
    held_out.to_parquet(folder / "predictions.parquet", index=False)
    write_json(folder / "selected_bandwidths.json", selections)

    # Descriptive SD over five OVERLAPPING training fits, not a confidence interval.
    stack = np.stack(surfaces)
    spread = stack.std(axis=0, ddof=1)
    stability = dict(grid=f"{grid[0]}x{grid[1]}", mean_map_sd=float(spread.mean()),
                     mean_relative_map_sd=float(np.mean(spread / np.maximum(stack.mean(axis=0), 1e-6))),
                     p90_map_sd=float(np.quantile(spread, .9)))
    shot_counts = np.bincount(tables["shots"].origin, minlength=grid[0] * grid[1])
    coverage = dict(grid=f"{grid[0]}x{grid[1]}", cells=len(shot_counts),
                    no_shots=int((shot_counts == 0).sum()), fewer_than_5_shots=int((shot_counts < 5).sum()))
    write_json(folder / "complete.json", dict(n_predictions=len(held_out), stability=stability, coverage=coverage))
    return held_out


# 4. Pooled scores, paired resolution comparison and smoothing diagnostics.

def collect_validation(output, grids):
    """Pool held-out EVENTS, preserving the distinction between the three tasks.

    Grid comparison uses goal-given-shot CE and paired-match jackknife errors.
    Neither component scores nor map SD are a direct validation of long-term xT.
    """
    tables, stability, coverage = [], [], []
    for grid in grids:
        folder = output / "validation" / f"{grid[0]}x{grid[1]}"
        record = json.loads((folder / "complete.json").read_text())
        tables.append(pd.read_parquet(folder / "predictions.parquet"))
        stability.append(record["stability"])
        coverage.append(record["coverage"])
    predictions = pd.concat(tables, ignore_index=True)
    metrics = predictions.groupby(["grid", "component", "version"]).agg(
        n=("nll", "size"), nll=("nll", "mean"), brier=("brier", "mean"),
    ).reset_index()
    by_match = predictions.groupby(["grid", "component", "version", "match_id"]).agg(
        n=("nll", "size"), nll=("nll", "mean"), brier=("brier", "mean"),
    ).reset_index()

    # a. Hold all saved goal predictions fixed for paired differences across grids.
    shots = predictions.loc[predictions.component.eq("shots") & predictions.version.eq("selected_smoothing")]
    best_grid = shots.groupby("grid").nll.mean().idxmin()
    reference = shots.loc[shots.grid.eq(best_grid)].set_index(["match_id", "row_index"])
    grid_comparisons = []
    for grid, group in shots.groupby("grid"):
        candidate = group.set_index(["match_id", "row_index"]).reindex(reference.index)
        if candidate.nll.isna().any():
            raise ValueError("Grid predictions must refer to the same shots.")
        delta_by_match = (candidate.nll - reference.nll).groupby("match_id").agg(["mean", "size"])
        delta, se = paired_match_difference(delta_by_match["mean"], delta_by_match["size"])
        nx, ny = map(int, grid.split("x"))
        grid_comparisons.append(dict(
            grid=grid, cells=nx * ny, ce=float(candidate.nll.mean()), brier=float(candidate.brier.mean()),
            reference_grid=best_grid, delta_ce=delta, paired_se=se,
            within_one_se=delta <= se + 1e-12,
            matches_lower_ce=int((delta_by_match["mean"] < 0).sum()),
            matches_higher_ce=int((delta_by_match["mean"] > 0).sum()),
        ))
    comparison = pd.DataFrame(grid_comparisons).sort_values("cells")
    chosen_grid = comparison.loc[comparison.within_one_se].iloc[0].grid

    # b. Smoothing vs no smoothing, paired on each event; bootstrap whole matches.
    smoothing = []
    for (grid, component), group in predictions.groupby(["grid", "component"]):
        paired = group.pivot(index=["row_index", "match_id"], columns="version", values="nll")
        differences = paired.selected_smoothing - paired.unsmoothed
        matches = differences.groupby("match_id").agg(["sum", "size"])
        weights = np.random.default_rng(20261015).multinomial(
            len(matches), np.full(len(matches), 1 / len(matches)), size=4000,
        )
        bootstrap = (weights @ matches["sum"]) / (weights @ matches["size"])
        smoothing.append(dict(grid=grid, component=component, n=len(differences), nll_delta=float(differences.mean()),
                              bootstrap_low=float(np.quantile(bootstrap, .025)),
                              bootstrap_high=float(np.quantile(bootstrap, .975)),
                              matches_improved=int((matches["sum"] < 0).sum())))

    folder = output / "scores"
    folder.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(folder / "oof_predictions.parquet", index=False)
    metrics.to_csv(folder / "metrics.csv", index=False)
    by_match.to_csv(folder / "metrics_by_match.csv", index=False)
    comparison.to_csv(folder / "grid_comparison.csv", index=False)
    pd.DataFrame(stability).to_csv(folder / "map_stability.csv", index=False)
    pd.DataFrame(coverage).to_csv(folder / "coverage.csv", index=False)
    pd.DataFrame(smoothing).to_csv(folder / "smoothing_comparisons.csv", index=False)
    write_json(folder / "selection.json", dict(selected_grid=chosen_grid, minimum_ce_grid=best_grid,
                                               rule="coarsest tested grid within one paired-match SE of minimum shot CE",
                                               scope="exploratory component comparison on ten matches"))
    return metrics
