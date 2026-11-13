# Pipeline procedurale di PRESS

Questa è la versione da leggere durante la revisione finale. Non sostituisce
ancora il pacchetto né i suoi modelli. I benchmark accettati usano il **BCOS
preliminare con hazard effettivi**; il nuovo trasporto con killing verrà
implementato e confrontato nella successiva revisione del pacchetto.

## Ordine numerato

`00_pipeline.py` mostra i comandi in questo ordine. Senza `--execute` non
esegue calcoli; si possono scegliere singole fasi con `--step 05`, ecc.

| Fase | Cartella | Ordine di lettura |
|---|---|---|
| 01 | `preprocessing/` | `features.py` → `pipeline.py`; `tracking.py` per derivate e segmentazione, `passes.py` per i collegamenti |
| 02 | `expected_threat/` | `model.py` → `training.py` → `pipeline.py` |
| 03 | `ball_drive/physical/` | `model.py` → `training.py` → `pipeline.py` |
| 04 | `ball_drive/catboost/` | `features.py` → `training.py` → `pipeline.py` |
| 05 | `pass_probability/` | `model.py` → `training.py` → `pipeline.py` |
| 06 | `pass_selection/` | `features.py` → `training.py` → `pipeline.py`; flat e CNL condividono le feature |
| 07 | `binary_classification/` | `features.py` → `training.py` → `pipeline.py`; prima action selection, poi residual risk |
| 08 | `composed_risk/` | `features.py` → `scores.py` → `pipeline.py` |
| 09 | `reports/` | `features.py` → `scores.py` → `cases.py` → `pipeline.py` |
| 10 | `benchmarks.py` | raccolta delle statistiche sui parametri fisici e sulle performance |

La numerazione riguarda le fasi, non aggiunge wrapper per ogni funzione.
Nei moduli scientifici i passaggi sono commentati e seguono lo stesso ordine.
I comandi di fitting completi sono per Orfeo; i controlli locali usano dati
reali ridotti e pochi alberi, senza ripetere tutte le ricerche.

```text
review/
  frame_states.py         # lettura uniforme dei frame tramite get_pv_players
  io.py                   # sola lettura/scrittura di tabelle e riepiloghi
  inputs/                 # input finali ordinati, separati dagli script
  preprocessing/          # tracking, possessi, segmentazione e coorti
  ball_drive/
    physical/             # model.py, training.py, pipeline.py
    catboost/             # features.py, training.py, pipeline.py
  pass_probability/       # model.py, training.py, pipeline.py
  pass_selection/         # features.py, training.py, pipeline.py (base e nidi)
  binary_classification/  # features.py, training.py, pipeline.py (action e residual)
  expected_threat/        # model.py, training.py, pipeline.py
  composed_risk/          # composizione held-out, metriche e bootstrap bande
  reports/                # pattern, squadre/giocatori, case studies reali
  outputs/                # creato dalle nuove esecuzioni, per task e modello
  local_history/          # prima run: script, input e output; SOLO locale
  release.py              # esportazione esplicita di codice e statistiche
```

- [Input finali](inputs/README.md): contenuto degli oggetti e dipendenze.
- [Ball drive](ball_drive/TRAINING.md): modello fisico e confronto CatBoost.
- [Pass probability](pass_probability/README.md): traiettorie, MLE e calibrazione.
- [Pass selection](pass_selection/README.md): feature comuni, ensemble base/nidi,
  aggiornamento di μ, screening 6+2, validation 6+2+2, temperatura e feature importance.
- [Expected Threat](expected_threat/README.md): componenti, smoothing, confronto
  delle griglie, mappe per fold e fit di produzione.
- [Action selection e residual risk](binary_classification/README.md): feature
  condivise, P(carry) cross-fitted nel training residual, training binario nativo, validation e refit.
- [Ball drive CatBoost](ball_drive/catboost/README.md): versione con lo stesso
  schema features/training/pipeline, naturale e resampling.

Le pipeline usano percorsi predefiniti interni a questa struttura. Non cercano
input in vecchie run e non installano gli output nel pacchetto PRESS. Le cartelle
storiche sono conservate; il pacchetto attuale non viene sostituito durante la revisione.

Le pipeline ricostruiscono gli stati dai tracking mediante `frame_states.py`,
che richiama `PRESS.utils.state.get_pv_players`. Per le conduzioni si usa
`frame_start`, per i passaggi il rilascio `frame_end`. I frame non vengono
spostati. Gli array iniziali archiviati rimangono disponibili per i controlli
di equivalenza; la destinazione osservata dei passaggi resta un input separato.

## Input e output

`inputs/<task>/` contiene le osservazioni accettate e i protocolli. Gli
upstream fisici, xT e pass selection necessari come feature sono in
`inputs/<task>/upstream/fold_<f>/` e `production/`. Sono **input espliciti**,
con provenienza verificabile, non una ricerca automatica delle ultime run.
Il piano di esecuzione riproduce i singoli task da questi input congelati:
non sostituisce automaticamente gli upstream con un nuovo fit sperimentale.

Il preprocessing può essere rieseguito dai file pubblici SkillCorner e salva
le coorti rigenerate in `outputs/preprocessing/task_inputs/<task>/`.
Il confronto delle chiavi con gli input congelati viene prima di un loro
eventuale aggiornamento: non si mescolano in silenzio due coorti diverse.
I filtri di esito restano nei moduli `filters.py` dei task.

Gli oggetti generati vanno in `outputs/<task>/features`, `training`, `scores`
e `production`, secondo la pipeline del task. I modelli binari condividono
il codice ma hanno output separati sotto `outputs/binary_classification/`.
I posterior e le matrici complete rimangono locali. I file di produzione
esportati qui non vengono installati automaticamente dentro PRESS.

## h, xLoss, bande e report

- h e xLoss: **15.085** stati canonici held-out con esito individuale verificato;
  metriche e calibrazione pooled, oltre ai riepiloghi per fold e partita.
- Bande e report: **14.919** stati di **7.265** possessi con azione terminale
  canonica. Nessuna durata minima generale; i micropossessi rimangono.
- Bande: 2.000 bootstrap per partita, seed 20261030, soglie/predizioni fisse,
  percentili 2,5–97,5, conteggi delle bande vuote e quasi vuote.
- Pattern: definizioni concordate e quote sia sulla coorte intera sia sui
  denominatori specifici; intervalli di Wilson sui possessi.
- E−O per 100 azioni elevated: bootstrap per possesso per giocatori e squadre,
  con almeno 20 possessi esposti per mostrare l'intervallo. La sensibilità
  per partita riguarda soltanto le squadre.
- Case studies: chiavi di possessi reali in `inputs/reports/cases.json`;
  h del fold held-out e plot dei frame tramite la funzione del pacchetto.

## Verifica ed esportazione

Leggere `checks/README.md` per i controlli fase per fase. Nessun risultato
ottenuto con pochi alberi è presentato come nuovo benchmark.

`release.py` copia solo il codice corrente e le statistiche elencate
esplicitamente: stime dei parametri fisici, performance/calibrazione,
bande e tabelle dei report/casi selezionati. Non copia ricorsivamente input,
posterior, booster di fold, vecchie prove o lo storico. I modelli preaddestrati
di inferenza continuano ad appartenere al pacchetto PRESS.

La prima run è conservata in `local_history/initial_run/<task>/`, con
`scripts/`, `inputs/` e `outputs/`. Gli originali in `history/initial_run/`
sono rimasti intatti. Nessuna delle due copie entra nel rilascio.
