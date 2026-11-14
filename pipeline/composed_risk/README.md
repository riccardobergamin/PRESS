# 08 — h, xLoss e bande

`features.py`: prima le due formule, poi l'inferenza dai modelli del fold.
Si usano la pass selection flat calibrata in temperatura, la pass probability
isotonic, la ball drive fisica e i due classificatori binari nativi.

`scores.py`: CE e guadagno sul nullo, Brier, AUC ed ECE pooled; riepiloghi per
fold/partita; bootstrap delle bande. Il nullo di ciascun target è la sua
frequenza sui sei match di training del fold, nella coorte verificata.

```bash
# Dopo le nuove feature e i nuovi fit binari:
python pipeline/composed_risk/pipeline.py --output /path/to/new_run/composed_risk \
  --binary-root /path/to/new_run/binary_classification

# Controllo della sola composizione dalle predizioni accettate:
python pipeline/composed_risk/pipeline.py --replay-accepted \
  --output /path/to/another_new_run/composed_risk
```

`--replay-accepted` è un controllo esplicito, non un nuovo training. Entrambe
le modalità producono `features/held_out.parquet`, `scores/` e `bands/`.
Le metriche usano tutta la coorte verificata; soltanto bande/report richiedono
copertura terminale canonica. I replicati del bootstrap rimangono locali.
