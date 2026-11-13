# Ball drive: le due procedure finali in tre file

Entrambe le versioni ricostruiscono lo stato al `frame_start` dal tracking,
usando `review/frame_states.py` e il getter `get_pv_players` di PRESS. Gli array
archiviati restano come riferimento numerico, non sostituiscono la lettura
dei frame nella pipeline. Coorte, target e segmentazione rimangono quelli accettati.

Tutto rimane in `skillcorner-analysis/review/ball_drive/`.
Il pacchetto PRESS e la pipeline attualmente distribuita non sono sostituiti.
Il solver di predizione già rivisto rimane nel file `ball_drive.py` di questa
cartella; i sei script qui sotto organizzano la stima e il benchmark.

## Ordine di lettura

| Parte | Modello fisico | CatBoost |
|---|---|---|
| 1. Modello | `physical/model.py`: covariate, spline, sopravvivenza, likelihood e prior. | `catboost/features.py`: le 50 covariate finali, encounter curve e likelihood Bernoulli nativa. |
| 2. Procedura | `physical/training.py`: MAP, Laplace, scelta dei rho, metriche e refit. | `catboost/training.py`: tuning, sampling, calibrazione, metriche. |
| 3. Esecuzione | `physical/pipeline.py`: carica i dati ed esegue i fold, poi aggrega ed eventualmente rifitta. | `catboost/pipeline.py`: calcola le feature ed esegue i bracci naturale/loss10, poi aggrega. |

Gli script hanno funzioni procedurali. Non importano le librerie della pipeline
rifattorizzata: le equazioni e la procedura sono visibili nei tre file di ciascun
modello. I calcoli di TTI richiamano i moduli fisici di PRESS.

## Input comune

Entrambe le procedure partono dalla **coorte finale già selezionata**, conservata
in `review/inputs/ball_drive/`: `rows.parquet` e `arrays.npz` con posizioni/velocità iniziali e
label. Sono 9.114 esempi, 456 dei quali falliti. Nessuna nuova esclusione viene
introdotta qui. Il preprocessing e la definizione della coorte sono a monte.

La selezione delle partite di test è quella del benchmark accettato. Gli ID e
l'ordine delle righe sono conservati. Il target fisico è successo/perdita entro
un secondo anche se una conduzione fallisce prima; CatBoost usa loss=1.

## Fisico

Nel file `physical/model.py` la preparazione dei dati e quella delle prior sono
separate:

- `build_spline_basis` costruisce la base di una singola covariata usando soltanto
  i valori del fitting. Nel training sono scritte esplicitamente le tre chiamate
  per velocità, bordo e angolo.
- `evaluate_spline_basis` applica una base già costruita a nuovi valori.
- `prepare_model_data` assembla le covariate e le matrici spline per training,
  validation o test. Non costruisce prior e non riceve i valori dei rho.
- `build_prior_precision` costruisce le precisioni gaussiane dai rho e dalle
  penalità delle basi di training. Serve alla stima MAP, non alla predizione.

La costruzione per una covariata, la valutazione per una covariata e
l'applicazione a tutte le covariate rimangono tre funzioni distinte. Nei commenti
sono riportate le formule accanto alle trasformazioni: C, L, H, G, W, R e Q
sono definiti all'inizio di `model.py`.

In `physical/training.py`, `fit_split` mostra esplicitamente i quattro passaggi:
preparazione dei due set, costruzione delle prior, fit sul training e predizione
sul set di valutazione con coefficienti fissi.

Il file è ordinato per la lettura: ottimizzazione MAP e Laplace; fit di un singolo
split; griglia e selezione della regolarizzazione; procedura sui cinque fold;
refit finale di produzione sulle dieci partite; funzioni di salvataggio; infine
metriche, bin di calibrazione e aggregazione delle predizioni held-out.
Il refit di produzione non contribuisce alle metriche del benchmark.

Per seguire il MAP, i cinque passi in `fit_map` sono: punti iniziali, ottimizzazioni,
eventuali partenze aggiuntive, scelta della soluzione convergente e Laplace.
`select_regularization` segue invece: CE pooled, differenze appaiate per partita,
jackknife, ammissibilità entro un errore standard e scelta nell'ordine prefissato.
Le configurazioni sono chiamate `model_configuration`; la chiave `candidate`
rimane soltanto nel formato degli oggetti/record salvati per compatibilità.

- Covariate: velocità del portatore, distanza firmata dal bordo più vicino,
  angolo d'approccio dei difensori; TTI alla Spearman con rotazione e overspeed.
- Finestra di 1 s; step 0.04 s; integrazione esponenziale della sopravvivenza.
- Basi ristimate solo sui fit match: cinque nodi per velocità/bordo, quattro
  per l'angolo, con le stesse trasformazioni e normalizzazioni del modello finale.
- Likelihood Bernoulli e prior finali, inclusa la Gamma su s e il Jacobiano in log(s).
- 33 punti prescored, L-BFGS-B, controllo della stazionarietà e Laplace.
- Quattro rotazioni interne 6+2 entro ciascun outer fold 8+2. Griglia di 16
  candidati e regola one-standard-error appaiata tra partite.
- Dopo la scelta: fit su otto partite, predizione sulle due test, metriche pooled.
- Refit di produzione opzionale, con scelta concorde tra fold e basi sulle dieci
  partite. Gli output nuovi rimangono nella cartella fornita, senza installazione.

`physical_splits.json` riporta anche gli s dei fit interni su sei partite e del
fit esterno su otto: CatBoost può usarli per le encounter curve.

## CatBoost

Solo il set finale di 50 feature e i due bracci riportati nel benchmark:
`natural` e `loss10`. Nessuna vecchia variante è inclusa.

- Griglia completa: depth 4/6/8; learning rate .02/.05/.1; L2 1/3/10/30.
- Alberi simmetrici, Plain boosting e altre impostazioni native della run finale.
- 3.000 alberi per fit di ricerca, valutati a checkpoint ogni 25 alberi.
- La CE pooled delle quattro validation 6+2 sceglie configurazione e prefisso.
  Nel braccio loss10 si applica la correzione analitica prima di questa CE.
- Il ricampionamento mantiene tutte le loss e sceglie fino a nove successi per
  loss, senza reinserimento, **solo nel sottoinsieme di fitting**.
- Tre seed: 19, 41, 73. Predizioni interne OOF sulle otto partite di sviluppo
  per stimare Platt e isotonic; successivo fit su otto e test sulle due escluse.
- Si riportano raw, prior correction, Platt e isotonic separatamente, senza
  scegliere la calibrazione dai risultati del test.
- Per ogni inner split, s viene dal suo fit fisico su sei match; per il test
  esterno viene dal fit fisico sulle otto partite di sviluppo.
- CatBoost ottimizza direttamente `loss_function="Logloss"`. Non ci sono
  gradienti personalizzati né prior Bayesiane esplicite sui suoi parametri.

Il controllo delle versioni mantiene CatBoost **1.2.10**, come nella run finale.
Non viene prodotto un sostituto operativo del modello fisico del pacchetto.

## Comandi

Gli input sono descritti in [../inputs/README.md](../inputs/README.md).
I percorsi predefiniti delle pipeline sono risolti rispetto alla cartella degli
script, senza dipendere dalla directory da cui vengono avviate.

Dalla cartella `skillcorner-analysis`, selezionare il checkout PRESS aggiornato:

```bash
export PYTHONPATH=../PRESS-portfolio-work
export JAX_ENABLE_X64=1

STATE_CACHE=review/inputs/ball_drive
```

Run fisica completa, con output nuovi:

```bash
python review/ball_drive/physical/pipeline.py \
  --cache "$STATE_CACHE" \
  --output review/outputs/ball_drive/physical
```

Aggiungere `--fold 0` per eseguire un singolo fold. Dopo i cinque job, ripetere
il comando con `--collect-only` per l'aggregazione, eventualmente con `--refit`
per il fit di produzione. La run completa è onerosa: non è stata lanciata durante
questa ristrutturazione.

Per CatBoost lo schema è ora `features.py`, `training.py`, `pipeline.py`,
con il flusso e le funzioni descritti nel [README dedicato](catboost/README.md).
La precedente versione è conservata in `catboost/old/`.

```bash
python review/ball_drive/catboost/pipeline.py --stage features \
  --physical-splits review/outputs/ball_drive/physical/physical_splits.json
python review/ball_drive/catboost/pipeline.py --stage train --threads 4
python review/ball_drive/catboost/pipeline.py --stage scores
```

La copia di parametri già accettati in
`review/inputs/ball_drive/reference_parameters/physical_splits.json`
è disponibile come input esplicito. `--stage train --fold 0 --sampling natural`
seleziona un solo job e carica le feature già salvate, senza ricalcolarle.

## Verifiche

Il file `check_training_versions.py`, esterno ai sei script, confronta formule,
gradienti, scelta dei rho, covariate e metriche con le versioni finali archiviate.
Comprende un fit fisico breve su dati reali. Non lancia il benchmark completo.
I risultati si trovano in `training_checks/`.

Il controllo è passato sui 9.114 stati: obiettivo e gradienti fisici coincidono
entro `3e-13`, le 50 feature CatBoost coincidono con quelle archiviate e le
metriche held-out vengono riprodotte. È stato eseguito anche un breve MAP su
128 stati reali, con Hessiana di Laplace positiva e verifica del modello
esportato tramite il reader pubblico di PRESS.

CatBoost non è installato nell'ambiente locale usato per questa revisione:
non sono stati eseguiti nuovi fit nativi degli alberi. La verifica CatBoost
copre covariate, indici di ricampionamento, protocollo e metriche delle
predizioni finali archiviate. Non è un nuovo benchmark né una verifica
end-to-end del training degli alberi.

Il controllo ha anche individuato un problema nel precedente refactoring della
pipeline (`pipeline/library/ball_drive/likelihood.py`): il controllo Python su
`ratio` entra in conflitto con il tracing JAX. Le run originali del benchmark
precedono quel refactoring. La nuova likelihood procedurale usa direttamente le
proporzioni naturali e non contiene quel ramo. Il file precedente resta invariato
in attesa della revisione/sostituzione concordata.
