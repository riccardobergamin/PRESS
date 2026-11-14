# Pass selection: revisione procedurale

Tre file: `features.py` costruisce le covariate; `training.py` contiene formule
e apprendimento; `pipeline.py` esegue le fasi.

Parti da `pipeline.py`, funzione **`run_pipeline`**: contiene in ordine le
cinque fasi complete. In `training.py`, **`train_validate_test`** mostra un
singolo fold e **`train_model`** il ciclo che costruisce gli alberi.

Il percorso di lettura è:

```text
run_pipeline
    1. prepare_features             covariate e membership per fold
    2. compare_configurations       cinque configurazioni sul fold 0
       select_best_configuration    scelta sulla CE di validation nativa
    3. train_validate_test          6 fit -> 2 validation -> 2 test
           train_model             ciclo esplicito degli alberi, mostrato sotto
           fit_temperature         alberi e rho fermi; temperatura su validation
           compute_pass_losses     valutazione sui due test
    4. compute_pooled_metrics       unione dei cinque test
    5. train_production_model       fit nuovo sulle dieci partite
```

Dentro `train_model`, a ogni iterazione:

```text
u_fit -> xgboost_gradients -> g,h -> model.boost -> nuovi u_fit
    -> aggiornamento rho ogni 20 alberi, solo per CNL
    -> u_val -> CE di validation -> early stopping
```

`model` è l'ensemble XGBoost; `D_fit` e `D_val` sono i suoi input (`DMatrix`);
`u_fit` e `u_val` sono le utilità sulle rispettive destinazioni candidate.
`best_tree_count` conserva il numero di alberi del minimo di validation;
`rounds_without_improvement` conta quante iterazioni sono trascorse senza
migliorarlo. Si restituiscono quegli alberi con il loro `best_rho`.

Le formule usano `u`, `p`, `y`, `ell`, `g`, `h`, `C`, come nella vignetta.
`y` è l'indice della destinazione osservata. Per CNL: `alpha`, `rho=1/mu`,
`S`, `w=P(nido)`, `b=P(cella|nido)`, `q=P(nido|cella osservata)`.
La legenda riporta le dimensioni; le formule sono commentate vicino al calcolo.
Dimensioni: `P` passaggi, `C` celle per passaggio, `F` feature e `N` compagni/nidi.
`T` rimane la temperatura. Negli array appiattiti `C_tot = sum_i C_i`;
se tutte le griglie hanno la stessa dimensione, `C_tot = P*C`.

`L(..., condition=True)` restituisce le loss delle destinazioni osservate;
`L(..., condition=False)` restituisce -log p per tutte le destinazioni.
La calibrazione usa quindi `-L(..., condition=False)` per rinormalizzare p^(1/T).
`loss_gradient` restituisce g,h per l'albero successivo.

Gli input sono ora **tre DataFrame**, caricati da `load_pass_tables`:

| Oggetto | Indice | Colonne | Righe |
|---|---|---|---:|
| `candidate_features` | `(pass_index, destination)` | Le F feature | C_tot |
| `observed_choices` | `pass_index` | `match_id`, `pass_id`, `frame_end`, `outcome`, `chosen_cell`, `n_candidates` | P |
| `log_memberships` | Come `candidate_features` | `log_alpha_0`, ..., `log_alpha_9` | C_tot |

`pass_index` identifica un passaggio nel set appena caricato; `destination`
è l'indice della cella candidata. I veri ID partita/passaggio sono nella
seconda tabella. `chosen_cell` contiene la destinazione osservata y_i.
La terza tabella è presente solo per CNL e conserva i log(alpha) già usati
nel fit; i compagni sono gli slot del roster al frame, non identità fisse
fra passaggi. Le membership non sono feature dell'ensemble.

```python
candidate_features.loc[i]   # C righe x F feature del passaggio i
observed_choices.loc[i]     # destinazione osservata e ID reali del passaggio i
log_memberships.loc[i]      # C righe x N appartenenze logaritmiche ai nidi
```

`L(u, y, rho, log_alpha, condition=True)` e
`loss_gradient(u, y, rho, log_alpha)` ricevono direttamente gli array
matematici: u `(P,C)`, y `(P,)`, log_alpha `(P,C,N)` oppure None per flat.
Le conversioni per XGBoost sono nella sezione 2: `build_xgboost_input`,
`xgboost_losses` e `xgboost_gradients`. I gruppi da 32 servono solo a
contenere la memoria; la likelihood e il training usano tutti i passaggi.
Le utilities sono u_ij=f_theta(z_ij), con f_theta somma degli alberi: ogni
`model.boost(..., grad=g, hess=h)` aggiunge un albero usando le derivate
calcolate esplicitamente alla riga precedente. È la stessa operazione nativa
che `Booster.update` esegue dopo il callback della loss custom; il confronto
numerico è stato verificato con XGBoost 3.2.0.

I salvataggi comuni e il formato di esportazione sono in fondo allo script.

La sezione 5 distingue `compute_pass_losses` (punteggi NLL per passaggio),
`compute_metrics` (CE e gain su una tabella di punteggi) ed `compute_pooled_metrics`
(unione dei cinque test e benchmark pooled, con dettaglio per fold/partita).

## Feature e dati

Restano i 6.810 passaggi selezionati. Gli stati al rilascio usano
`pipeline/frame_states.py` e `get_pv_players` di PRESS. Ogni fold usa i pesi
fisici e xT delle proprie sei partite di fit; produzione usa le dieci partite.
Le dispersioni `pass_s` e `ball_drive_s` restano distinte. Il vocabolario delle
categorie è costruito sul fit. I pesi sono quelli del benchmark accettato,
indipendenti dalla prova sulle traiettorie in `pipeline/pass_probability`.

Le covariate descrivono distanza/angolo del passaggio, velocità del portatore,
pitch control grezzo alla destinazione, xT e indici di linea al portatore e
alla destinazione, rischio di perdita in conduzione a un secondo, minuto e
punteggio, start type/previous action e conteggi oltre le due linee trasversali.
Per entrambi i roster si usa la direzione d'attacco del portatore; non si
contano soltanto i giocatori nel segmento fra origine e destinazione.

Le due famiglie condividono le feature. Le appartenenze CNL sono i contributi
individuali di pitch control grezzo dei dieci compagni escluso il portatore,
normalizzati e fissi. Non sono covariate aggiuntive delle utilities.

## Screening e training finale

Sul fold 0 si confrontano queste cinque configurazioni per famiglia, usando
sei partite di fit e due di validation. Selezione sulla CE **nativa**, senza
temperature scaling né valutazione del test. Ogni fit riparte da u=0 e rho=1.

| Nome | Depth | Child weight | L2 | Eta | Limite alberi |
|---|---:|---:|---:|---:|---:|
| `01_depth6_child5` | 6 | 5 | 3 | 0,05 | 2.000 |
| `02_depth3_child5` | 3 | 5 | 1 | 0,05 | 3.000 |
| `03_depth4_child20` | 4 | 20 | 1 | 0,05 | 2.000 |
| `04_depth4_child1` | 4 | 1 | 1 | 0,05 | 2.000 |
| `05_depth4_child5_baseline` | 4 | 5 | 1 | 0,05 | 2.000 |

Vincitori accettati: flat depth 4 / child 20 / L2 1; CNL depth 6 / child 5 /
L2 3. La tabella originale è in `inputs/pass_selection/reference/screening.csv`.
Impostazioni comuni: CPU hist, subsample/colsample 0,8, max_bin 256, patience
100, seed 20261019 + fold. CNL aggiorna rho con SLSQP sul fit ogni 20 alberi,
con rilassamento 0,1. Nello screening SLSQP parte dal rho corrente; nei fit
finali/refit si aggiunge la scansione [1,1.5,2,3,5,10]. Flat usa il fattore
Friedman C/(C-1); CNL usa la curvatura CPU di RUMBoost.

Seguono cinque fold **6+2+2**. Le due validation scelgono prefisso/rho e poi T;
non si rifitta il modello di scelta sulle otto partite prima del test.
Si concatenano i test prima di calcolare CE e gain. Lo screening esplorativo
non è tuning annidato indipendente per ogni fold.

Il refit di produzione riparte sulle dieci partite con la mediana dei numeri
di alberi selezionati (run accettata: 1.444 flat, 359 CNL). La sua T è stimata
sulle distribuzioni native outer-OOF, non sulle predizioni in-sample del refit.

## Esecuzione

```bash
python pipeline/pass_selection/pipeline.py --stage features --fold 0
python pipeline/pass_selection/pipeline.py --stage screen --threads 4
```

Per job separati, scegliere `--family cnl --configuration 01_depth6_child5`,
per esempio; dopo tutti e cinque usare `--stage screen-report --family cnl`.
Generare gli altri fold con `--stage features --fold 1`, ecc., quindi:

```bash
python pipeline/pass_selection/pipeline.py --stage train --threads 4
python pipeline/pass_selection/pipeline.py --stage scores
python pipeline/pass_selection/pipeline.py --stage features --production-features
python pipeline/pass_selection/pipeline.py --stage refit --threads 4
```

`--stage all` parte da una cartella output nuova ed esegue tutto. Saltando lo
screening usa le configurazioni accettate; uno screening parziale va concluso.
Le cartelle già esistenti non vengono sovrascritte.

## Risultati e controlli

- `features/`: cache condivisa, per fold e produzione.
- `screening/<famiglia>/<configurazione>/`: riepilogo, history e aggiornamenti
  rho. Non si conservano booster intermedi o tabelle duplicate di predizioni.
- `<famiglia>/training/fold_N/`: modello, parametri/T, tracce, predizioni di test
  e log-probabilità native necessarie alla calibrazione di produzione.
- `<famiglia>/scores/`: metriche pooled/per fold/per partita e importance.
- `<famiglia>/production/`: modello esportabile e metadati.

Importance: total split gain normalizzato, sommando prima le categorie one-hot.
Media/minimo/massimo dei fold restano separati dal valore di produzione;
non sono intervalli di confidenza o effetti causali.

I controlli della nuova interfaccia sono in `checks/verify_pipeline.py`, fuori dal
ciclo di apprendimento: dieci release reali, allineamento delle tre tabelle,
loss/derivate, fit brevi di 22 alberi, early stopping, calibrazione e salvataggi.
`checks/tables_equivalence.json` riporta il confronto con la versione precedente;
i precedenti controlli fisici sono conservati in `checks/equivalence.json`.
Restano nel flusso
i controlli su problemi numerici, convergenza, copertura held-out e sovrascrittura.
I fit brevi di verifica non possono selezionare iperparametri finali.

Riferimento: [RUMBoost](https://doi.org/10.1016/j.trc.2024.104897).
