# 09 — Possession patterns and reports

Reading order: `features.py` → `scores.py` → `cases.py` → `pipeline.py`.

- `classify_possessions` creates one row per `(match_id, possession_id)`, with initial/final states and the three pressure patterns, including the single-checkpoint convention for micropossessions.
- `pattern_shares` reports shares in the full cohort and the pattern-specific denominators.
- `entity_score` calculates E−O per 100 elevated actions, counts, mean h, observed failure frequency and High share.
- `bootstrap_entity` resamples whole possessions 2,000 times, retaining all elevated actions of each sampled possession together.
- `export_cases` exports checkpoint tables and optional real-frame plots. The next control illustrates the outcome without adding another h checkpoint.

```bash
python pipeline/reports/pipeline.py \
  --states /path/to/new_run/composed_risk/features/held_out.parquet \
  --output /path/to/new_run/reports \
  --tracking-root pipeline/inputs/shared/tracking
```

Omit `--tracking-root` to generate tables only. Pattern membership is checked before plotting a case. Reports use held-out predictions, not production-model predictions. Definitions and interval limitations are documented in the [report vignette](../../vignettes/07_case_studies/07_case_studies.ipynb).
