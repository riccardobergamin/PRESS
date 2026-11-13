# Input finali action selection / residual risk

- `action_rows.parquet`: 16.918 checkpoint accettati, 7.770 passaggi. Include
  gli identificativi reali e il portatore matched, i label e il contesto al frame.
- `residual_labels.parquet`: 7.820 carry riusciti con esito finale verificato;
  1.423 perdite terminali, età osservata del possesso e identificativi del parent.
- `protocol.json`: partizioni, grid di 144 configurazioni, seed e impostazioni
  finali BCOS. Entrambi i modelli mantengono probabilità native.
- `upstream/fold_0..4/`: pesi fisici e modello flat di scelta con xT dei sei
  match corrispondenti. `upstream/production/`: pesi su tutti i dieci match.
- `reference/`: tabelle numeriche, griglie e selezioni archiviate per i controlli.
  Le tabelle `action_features_subset` contengono solo un sottoinsieme held-out,
  non una cache completa da usare per nuovi training. Le nuove pipeline non
  leggono automaticamente feature o modelli binari da `reference/`.
- `manifest.json`: SHA-256 delle copie e conteggi delle coorti.

I tracking sono condivisi in `../shared/tracking/`. Le righe di action selection
sono state collegate alla tabella completa delle azioni tramite match/action/frame
per recuperare i portatori, senza eliminare i contesti unknown. Il label e tutte
le covariate decisionali restano quelli finali. Non sono stati aggiunti filtri.

Ogni nuova esecuzione scrive in `review/outputs/binary_classification/`, con
cartelle separate per task e per features/training/scores/production. Il
pacchetto e i suoi modelli restano invariati durante la revisione.

La procedura attiva mantiene P(carry) cross-fitted sul training residual,
con il modello action principale per validation/test. La prova semplificata
è stata scartata; le tabelle di riferimento conservano la procedura approvata.
