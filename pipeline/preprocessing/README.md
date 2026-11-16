# 01 — Preprocessing

Read `features.py` for the three preparation stages:

1. `prepare_tracking`: read original JSONL/JSON/CSV files, preserve native coordinates and identities, and estimate offline kinematics.
2. `prepare_possessions`: check matched boundaries and usable frames, link possessions, and construct canonical actions and separate snapshots.
3. `prepare_task_tables`: apply the task-specific filters and create observation/label tables. Expected Threat retains its distinct event cohort.

`tracking.py` contains sample processing and segmentation; `events.py` annotates individual possessions; `possession_selection.py` and `passes.py` implement release-to-next-control links. The [segmentation guide](../../vignettes/setup/possession_and_checkpoint_setup.ipynb) defines the conventions and context labels.

```bash
python pipeline/preprocessing/pipeline.py \
  --raw-root /path/to/opendata/data/matches \
  --output /path/to/new_run/preprocessing
```

Use `--match 1925299` for one match. Stages can be run separately with `--stage tracking`, `possessions` and `datasets`. Annotated tracking is saved as `matches/<match>/<match>_tracking.parquet`; task observations are in `task_inputs/<task>/rows.parquet`.

`--templates pipeline/inputs` adds the declared protocols and upstream artifacts without copying observations. It also creates pass-probability target arrays, pass-selection player metadata, binary-task observations and labels, and composed-risk/report metadata. Preprocessing does not refit upstream models.

Pass the generated task directory as `--data-root` and the preprocessing root as `--tracking-root`. The overall execution plan accepts `--input-root` and `--tracking-root`. Cohort checks compare row keys and labels, not only counts. There is no general minimum-possession-duration filter for h.
