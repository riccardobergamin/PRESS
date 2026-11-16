# 08 — h, xLoss and risk bands

`features.py` defines the composition and evaluates the corresponding fold models: temperature-calibrated flat pass selection, isotonic pass probability, physical ball-drive risk and the two native binary classifiers.

`scores.py` calculates pooled CE, gain over null, Brier score, AUC and ECE, with fold/match summaries and band-bootstrap intervals. Each target's null probability is its prevalence on the fold's six fitting matches in the verified cohort.

```bash
# Use generated features and fitted binary models.
python pipeline/composed_risk/pipeline.py --output /path/to/new_run/composed_risk \
  --binary-root /path/to/new_run/binary_classification

# Check composition using reference component predictions.
python pipeline/composed_risk/pipeline.py --replay-accepted \
  --output /path/to/another_new_run/composed_risk
```

`--replay-accepted` performs no new training. Both modes produce `features/held_out.parquet`, `scores/` and `bands/`. Overall metrics use the complete verified cohort; bands and reports additionally require a terminal canonical action. Bootstrap replicates are generated outputs and are not distributed with the website.
