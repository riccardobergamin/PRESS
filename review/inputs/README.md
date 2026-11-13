# Input finali delle pipeline in revisione

Questa cartella contiene copie indipendenti degli input finali accettati. Non
contiene la cronologia delle run. Gli originali sono conservati e le copie dei
file numerici sono state verificate tramite SHA-256, senza modificarne il contenuto.

```text
inputs/
  folds.json
  manifest.json
  shared/
    tracking/
      matches/<match_id>/<match_id>_tracking.parquet
  ball_drive/
    rows.parquet
    arrays.npz
    reference_parameters/
      physical_splits.json
  pass_probability/
    rows.parquet
    arrays.npz
    boundaries.parquet
    matches/<match_id>/
      boundaries.parquet
      trajectories.npz
```

## Input condivisi

`folds.json` riporta le due partite di test di ciascuno dei cinque fold.
Le pipeline mantengono gli stessi split del benchmark finale.

`shared/tracking/` contiene i dieci tracking preprocessati completi. CatBoost
legge la direzione d'attacco al frame iniziale; pass probability li usa per
stimare le condizioni iniziali e la traiettoria di ogni passaggio.
Non è un link a una cartella esterna. Le pipeline in revisione usano
`review/frame_states.py` e `get_pv_players` di PRESS per ricostruire gli stati
richiesti da ciascun task. Gli stati negli array archiviati restano un riferimento
per i controlli di equivalenza; per pass probability `r_target` rimane la
destinazione osservata letta dagli input.

## Ball drive

`rows.parquet` e `arrays.npz` sono comuni alla versione fisica e a CatBoost:
**9.114 osservazioni, di cui 456 fallimenti**. Le righe e gli array hanno lo
stesso ordine; non sono state eseguite nuove esclusioni.

`reference_parameters/physical_splits.json` contiene soltanto gli split e i
valori di dispersione s dei fit fisici finali accettati. Serve se si vuole
eseguire CatBoost senza rifare prima la stima fisica. Per una riproduzione
completa, CatBoost legge invece `physical_splits.json` generato dal nuovo fit
fisico in `review/outputs/ball_drive/physical/`.

## Pass probability

`rows.parquet` e `arrays.npz` contengono **6.706 passaggi, 5.742 riusciti e 964
falliti**, con gli stati iniziali e gli stessi criteri di selezione già accettati.

`boundaries.parquet` contiene gli errori di previsione e la diagnostica delle due
stime delle condizioni iniziali. Le sottocartelle per partita conservano gli
stessi record e le traiettorie ricostruite con `mean_velocity` e `fit_positions`.
Queste copie sono conservate per i controlli numerici. La pipeline in revisione genera le due candidate per ogni passaggio, confronta
l'errore medio sul moto reale e salva solo quella scelta. Il training usa TTI
della sola traiettoria scelta. La selezione aggregata per split rimane nella
pipeline precedente.

La fase di preparazione esegue in ordine le due funzioni:

```bash
python review/pass_probability/pipeline.py --stage features
```

`generate_trajectories` stima e salva le traiettorie e la diagnostica del fit;
`prepare_features` le carica, calcola TTI e quantità PPCF e salva gli input.
Il tracking predefinito è `review/inputs/shared/tracking/`.

## Oggetti rigenerati

Gli input rimangono separati dagli oggetti prodotti dalla procedura:

```text
review/outputs/
  ball_drive/
    physical/          # fit, validation, score, posterior, refit
    catboost/          # feature, tuning, bracci naturale/loss10, score
  pass_probability/
    trajectories/
    features/
    training/
    scores/
    production/
```

Queste cartelle sono create quando si eseguono le rispettive fasi; non sono
state riempite con risultati di vecchie run. Le pipeline rifiutano di
sovrascrivere gli output già presenti.

## Pass selection

`pass_selection/rows.parquet` contiene i **6.810 passaggi finali, di cui 1.009
falliti**. Il contesto è quello noto al rilascio. Non vengono aggiunti filtri.
`protocol.json` riporta gli split 6+2+2 e le due configurazioni finali.
`reference/screening.csv` conserva il confronto originale delle cinque
configurazioni per famiglia, sul solo split 6+2 del fold 0. È un riferimento
per il controllo; una nuova esecuzione salva i propri risultati in
`review/outputs/pass_selection/screening/`.

`matches/<match>/players.parquet` collega gli ID degli eventi agli alias dei
giocatori nei tracking condivisi. Gli stati vengono letti al rilascio.

`upstream/fold_0/` ... `fold_4/` contengono i pesi fisici di ball drive e
pass risk stimati sui rispettivi sei match, oltre alla superficie xT dello
stesso split e alla sua provenienza. `upstream/production/` contiene invece
i pesi e la superficie sulle dieci partite. Sono copie indipendenti degli
oggetti accettati; `pass_selection/manifest.json` ne registra fonte e SHA-256.
Non vengono usati i risultati sperimentali della nuova revisione di pass probability.

Le nuove esecuzioni salvano le feature comuni in
`review/outputs/pass_selection/features/`; training, score, refit e importance
sono separati in `review/outputs/pass_selection/flat/` e `cnl/`.

## Expected Threat

`expected_threat/events/` contiene le quattro tabelle congelate del confronto
xT accettato: carries, passes, shots e losses. La pipeline ricalcola i bin
dalle coordinate continue senza introdurre nuovi filtri o cambiare la coorte.
`protocol.json` distingue test pair del confronto 8+2, sei match di fit per
le mappe downstream e superficie di produzione sulle dieci partite.
`reference/` conserva solo gli oggetti utili al controllo di equivalenza.
Copie indipendenti e SHA-256 sono registrati in `expected_threat/manifest.json`.

Le nuove esecuzioni scrivono in `review/outputs/expected_threat/`, separando
validation, score, feature maps e produzione.

## Action selection e residual risk

`binary_classification/` contiene le due coorti finali e i modelli upstream
ordinati per fold/produzione, con le tabelle di riferimento per i controlli.
Si vedano [contenuto e conteggi](binary_classification/README.md).
Le nuove feature, i fit e i refit vanno in
`review/outputs/binary_classification/action_selection/` e `residual_risk/`.

## Preprocessing

`preprocessing/matches/<match>/` contiene le tabelle accettate di possessi,
azioni, identità ed eventi con i metadati. Sono riferimenti locali per la
ricostruzione da raw; il tracking completo è già in `shared/tracking/`.
Gli script di preprocessing leggono i file pubblici originali e non questi
risultati, salvo un controllo esplicito di equivalenza.

## Composed risk e report

`composed_risk/labels.parquet` contiene i label e i metadati dei 15.085 stati
verificati. `band_eligible` identifica i 14.919 stati con copertura terminale.
I confini temporali dei possessi, il portatore, la squadra e il controllo
successivo sono metadati osservati, non feature del modello.

`composed_risk/reference/components.parquet` conserva le probabilità accettate
dei singoli rami, utili a ricomporre h e xLoss senza un nuovo training.
Questa modalità richiede `--replay-accepted`; il percorso ordinario usa le
nuove feature e i modelli binari del fold della nuova esecuzione.
`reference/fold_stack/` consente inoltre il controllo d'inferenza dai cinque
modelli binari accettati e dalle rispettive feature held-out, senza rifarli.

`reports/cases.json` seleziona i tre possessi reali del report; `reference/`
contiene soltanto le tabelle pubblicate da confrontare, non modelli nuovi.
Gli output dei report non vengono scritti negli input.

`pass_selection/smoke/` contiene un passaggio reale per partita con tutta la
griglia e le appartenenze ai nidi. È un input dei test brevi, non una coorte
aggiuntiva di benchmark. Nessuna di queste tabelle viene copiata nel sito.
