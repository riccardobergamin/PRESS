# Accepted boundary-selection policy

Implemented on 2026-10-14 in the sibling `skillcorner-analysis` source scripts. This step changes boundary eligibility and provenance; it does not change the canonical action segmentation, physical pass model, target geometry, or trained artifacts. The subsequent raw-to-base regeneration is documented in `pipeline.qmd`; earlier processed files remain separate.

## Two distinct cohorts

| Dataset | Required matched boundaries | Boundaries not required |
|---|---|---|
| Individual possession intervals | That interval's start and end. | None of its own boundaries may be unmatched. |
| Pass examples | Sender's end (release) and the immediate next individual possession's start (reception/interception/recovery). | Sender's start and the next individual's end. |

Pass pairs are found in the full chronological provider stream before individual-interval filtering. An unmatched immediate next start is rejected; it is never skipped to use a later matched event. Pairing remains within the same match and period. Existing outcome, interception-type and target-availability filters still apply after the boundary criterion.

The point of interception/recovery is kept as the observed pass endpoint in the existing model. It is not labeled as the passer's intended destination or as proof of a technical mistake. The mathematical assumption and its benchmark are deferred to the pass-probability review. The target convention itself has not changed.

## Implementation

- `skillcorner-analysis/scripts/utils/possession_selection.py` contains the shared explicit-true flag handling, individual mask, broader pass-boundary pairing and release-state annotation.
- `scripts/utils/preprocessing.py::annotate_possession` now retains only start/end matched intervals. Receipt candidates come from the unfiltered stream, so excluding an intermediate individual interval cannot fabricate a pass link to a later interval. Boundary flags, receipt provenance, match and period are retained.
- `scripts/analysis/ball_drive_segmentation.py` passes the full individual-possession stream to that function instead of filtering first. Its action-duration and segmentation rules are unchanged.
- `scripts/analysis/preprocess_matches.py` preserves `player_possession_provider` and labels annotation provenance in `possession_source`. A valid broader pass can add a carrier/direction only at its release frame, with source `matched_pass_release`; it does not restore an excluded individual interval. Existing carrier conflicts are reported.
- `scripts/utils/passes.py::extract_passes` retains the broader release/next-start rule, handles missing flags without `bool(NaN)`, and prevents receiver coordinates being inherited from the preceding pass. Unobserved receiver coordinates remain missing.

The subsequent unified pipeline retains detection/extrapolation flags and native dimensions. Source paths and the wider pipeline migration have not been refactored in this step.

## Read-only check on the ten local matches

The updated functions were evaluated in memory, with no output-file regeneration and no training:

| Quantity | Count |
|---|---:|
| Raw individual possessions | 9,566 |
| Both-boundary matched individual candidates | 9,139 |
| Annotated matched individuals after existing recognized-ending filters | 9,106 |
| Pass pairs satisfying the boundary rule | 7,941 |
| Pass rows after the existing outcome/type/target filters | 7,467 |
| Boundary pairs whose sender start is unmatched | 2 |
| Boundary pairs whose receiver end is unmatched | 55 |
| Additional release-state annotations beyond the selected individual intervals | 19 |

The broader boundary pairs include cases that are subsequently excluded by other pass filters. These are coverage counts, not performance estimates. The in-memory interval tables also passed the package's team-link validation. Twenty-one targeted tests cover nullable flags, broader pass eligibility, raw-neighbor receipt lookup, match/period boundaries, first and consecutive failed passes, release-state preservation and conflicting carriers.

A separate comparison found 172 receipt-frame differences across the ten matches when the same annotation function received already-filtered intervals instead of the full stream. The active segmentation caller now uses the full stream; the selected individual cohort remains the same. This comparison does not establish the ground-truth correctness of every endpoint.

Regeneration subsequently wrote a separate versioned output under `opendata/data/processed/2026-10-14_base_v1`, recording the policy and preserving older data. No benchmark results should be recomputed until the remaining preprocessing and segmentation choices have been reviewed.
