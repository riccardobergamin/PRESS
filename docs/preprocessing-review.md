# First review: preprocessing, possession selection and segmentation

Reviewed on 2026-10-14 against the existing `skillcorner-analysis` scripts and the ten local open-data match files. This records the read-only review snapshot before the accepted boundary-selection change. The source selection rule was subsequently implemented as documented in [the accepted policy](possession-selection.md); a later full raw-to-base regeneration is documented in `pipeline.qmd`. This document remains a historical snapshot; model fitting has not been rerun. Benchmark review is deferred while these contracts are discussed.

## Current flow

1. `scripts/analysis/preprocess_matches.py` loads extrapolated tracking and match metadata through `load_enriched_tracking_data`.
2. `compute_velocity` differentiates coordinates with centered Savitzky–Golay filters: seven samples for players, five for the ball, at 10Hz. `compute_acceleration` applies another centered derivative to velocities. Gaps larger than one frame split processing sessions.
3. Player IDs are mapped to home/away jersey labels. `long_to_wide` pivots coordinates and kinematics into the package layout.
4. `annotate_possession` constructs individual intervals from provider player-possession events, retaining recognized endings and marking loss/non-loss.
5. The script replaces the tracking's initial provider carrier annotation with those interval-derived carrier/direction labels and saves tracking, individual intervals and the player map.
6. `scripts/analysis/ball_drive_segmentation.py` separately selects events with both provider start/end matched flags, applies the annotation function, and keeps endings in pass/clearance/possession_loss/foul_suffered.
7. It creates action rows within those individual intervals. The package separately reconstructs team intervals through conservative successful-pass linking.

## Observed selection counts

Counts are aggregates from the ten local `*_dynamic_events.csv` and processed `*_possession.parquet` files. They are coverage diagnostics, not model performance.

| Stage | Count |
|---|---:|
| Raw individual-possession events | 9,566 |
| Events with both start/end matched flags | 9,139 |
| Events meeting both matched flags and segmentation end-type list | 8,913 |
| Individual intervals currently saved by preprocessing | 9,528 |
| Saved intervals without both matched flags | 422 |
| Matched possessions ending in a shot, excluded by the segmentation end-type list | 212 |
| Overlapping saved individual intervals after sorting by frame | 0 |

Preprocessing and segmentation currently use different cohorts. That can be intentional: a reconstructed match may retain more intervals than a training set. The distinction needs explicit eligibility flags and reasons, rather than allowing one table to be mistaken for the other.

## Preprocessing contracts to settle first

**Preserve quality information.** Tracking loading contains player detection flags and `is_detected_ball`; the wide conversion drops them. None of the ten saved tracking schemas retains detection/extrapolation flags. Preserve these alongside coordinates so later eligibility and sensitivity checks can distinguish directly detected players from extrapolated positions. Retaining a flag does not imply discarding extrapolated players; that is a separate policy decision.

**State reconstruction versus live prediction.** The centered player velocity filter uses observations on both sides of a frame (three samples ahead for the nominal seven-sample interior window). That is defensible for offline reconstruction of a recorded match. It does not establish that the same state is available in a live causal prediction. Keep offline smoothing explicit; a causal alternative is a separate comparison, not an automatic replacement.

**Pitch geometry.** Match metadata contains lengths of 104, 105 and 106 metres, all with width 68 metres. Loading preserves metre coordinates; several inference utilities currently assume 105 × 68 bounds. Choose explicitly between retaining native geometry throughout and normalizing to a reference pitch, including the corresponding kinematic treatment. No normalization is applied by this review.

**Provider versus interval carrier labels.** `long_to_wide` initially maps the provider carrier, but the orchestration clears `player_possession` and rewrites it over recognized event intervals. Preserve both sources and their provenance before deciding which supplies the model state. Unannotated/in-flight frames should remain distinguishable from quality exclusions.

## Possession selection and continuity

Maintain a complete individual-interval table with provider IDs, boundary flags, raw ending, pass outcome and eligibility reasons. Derive component-specific training cohorts from it. Keep the team-possession reconstruction as a separate table.

A successful-pass receipt frame is currently assigned from the next row of the sorted event table by `annotate_possession`; it is not an independently verified receiver lookup. Among the 6,800 saved successful-pass rows, the next chronological saved carrier matches the recorded target for 6,455, and the stored receipt frame matches the next saved interval start for 6,650. These are separate checks; they do not prove that every mismatch is a failed pass or an annotation error. Dropped events, uncertain targets and unmatched boundaries need inspection before changing continuity rules. The package adapter checks receiver, side, receipt and interruption before linking and otherwise starts a new team interval.

`annotate_possession` labels clearance as loss and shot/foul suffered as non-loss. These are project endpoint conventions. In particular, treating a deliberate clearance as loss should be reviewed against the pressure/retention objective rather than silently equated to a failed control or pass.

Shot endings are retained in preprocessing but excluded before segmentation. That removes preceding carry examples too. Excluding an xG model does not by itself require discarding the whole lead-up to a shot. Discuss retaining the complete sequence and choosing which canonical steps are eligible for each component; shot transitions and incomplete final carry windows need their own handling.

## Segmentation facts to review next

The source script uses `HZ=10` and `BALL_DRIVE_FRAMES=10`.

- For possession-loss endings, full one-second blocks are emitted; a final shorter block is emitted when needed and labeled unsuccessful. If duration is an exact multiple, the last full block is unsuccessful.
- For pass/clearance/foul endings, only full one-second drive blocks are retained. The remaining subsecond part does not become a separate successful drive. For passes, the active source emits a terminal pass row at the release frame with zero duration; its label is the pass outcome. The earlier review misdescribed it as an interval from the previous drive boundary. The pre-release remainder is not a separate action in this policy.
- Clearance/foul terminal events are not represented as their own action rows.
- Adjacent drive rows share endpoint frame numbers. This is compatible with shared observations at boundaries, but the interval convention must be explicit before counting exposures or multiplying step risks.

The extended Fernandez framework defines drive actions over one second; the reviewed paper does not supply a full extraction algorithm for strides, shared endpoints or final partial windows. Current project details should remain documented as project conventions. See [the interpretation guide](../interpretation.qmd).

## Order of discussion

First settle the preprocessing output: native geometry, quality/provenance fields and offline kinematics. Next settle the complete possession table and component eligibility. Then settle canonical drive/pass boundaries, short terminal intervals and team continuity. Only after those decisions should datasets and benchmarks be rebuilt.
