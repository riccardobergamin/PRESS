# 01 — Preprocessing

`features.py` contiene il flusso principale, senza un dispatcher nascosto:

1. `prepare_tracking`: JSONL/JSON/CSV pubblici → coordinate native, identità,
   derivate cinematiche offline e annotazioni originali.
2. `prepare_possessions`: estremi matched, collegamenti, frame utilizzabili,
   segmentazione canonica e snapshot separati. Nessun filtro di esito del task.
3. `prepare_task_tables`: applicazione esplicita dei filtri dei task e tabelle
   di righe/label. La preparazione degli eventi xT mantiene la sua coorte.

`tracking.py` raccoglie le operazioni sui campioni e la segmentazione;
`events.py` annota i possessi individuali; `possession_selection.py` e
`passes.py` definiscono il collegamento rilascio–controllo successivo.
Sono le operazioni del processing già verificato, senza nuove convenzioni.

```bash
python pipeline/preprocessing/pipeline.py \
  --raw-root /path/to/opendata/data/matches \
  --output /path/to/new_run/preprocessing
```

Si può scegliere `--match 1925299` e lanciare le tre fasi separatamente con
`--stage tracking`, `possessions` e `datasets`. Il tracking annotato si trova
in `matches/<match>/<match>_tracking.parquet`, compatibile con `frame_states.py`.
Le righe sono in `task_inputs/<task>/rows.parquet`; gli upstream addestrati e
i protocolli delle pipeline di fitting sono input distinti.

Con `--templates pipeline/inputs` vengono aggiunti esplicitamente i protocolli
e gli upstream accettati, senza copiare le osservazioni. Sono creati anche
`pass_probability/arrays.npz` (solo destinazione osservata), i metadati dei
giocatori per pass selection, `binary_classification/action_rows.parquet` e
`residual_labels.parquet`, e i label/metadata per h e report. I modelli non
sono ristimati dal preprocessing: la provenienza è registrata.

Per usare questi nuovi input si passa il loro percorso a `--data-root` del
task e la radice del preprocessing a `--tracking-root`. Il piano generale
accetta gli stessi percorsi tramite `--input-root` e `--tracking-root`.

Il confronto con le coorti accettate riguarda le **chiavi delle righe e i label**,
non soltanto i conteggi. Non si applicano filtri di durata ai possessi di h.
