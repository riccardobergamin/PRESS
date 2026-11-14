# Expected Threat: revisione procedurale

Ordine di lettura:

1. **`model.py`**: coordinate e celle, transizioni smussate, probabilità di goal
   dato tiro, frequenze locali dei quattro tipi di evento, iterazione di Bellman.
   Le formule e gli assi degli array sono commentati accanto al codice.
2. **`training.py`**: score delle componenti, scelta del bandwidth mediante
   leave-one-match-out, differenze appaiate e jackknife, validation 8+2,
   confronto delle griglie e calcolo delle metriche pooled.
3. **`pipeline.py`**: carica le tabelle, esegue il confronto, genera le mappe
   di training per i task successivi e rifitta la superficie di produzione.

Non si modifica il pacchetto o il sito. Non vengono sostituiti gli oggetti
già accettati; gli output nuovi sono scritti in una cartella distinta.

## Input

```text
pipeline/inputs/expected_threat/
  events/
    carries.parquet
    passes.parquet
    shots.parquet
    losses.parquet
  protocol.json
  manifest.json                # copie indipendenti con fonti e SHA-256
  reference/
    production/                # superficie 10x6 e componenti del fit accettato
    six_match_maps/fold_0..4/   # superfici accettate per pass selection
    validation/                # risultati accettati per i controlli
```

Sono le **stesse quattro tabelle archiviate** del confronto finale: 9.071
spostamenti durante possessi individuali, 6.355 passaggi completati, 212 tiri
(23 goal), 2.214 perdite terminali. Non vengono estratti nuovamente dai raw.
La popolazione di xT rimane quella approvata, diversa dalla coorte di pass risk.
In particolare, questi spostamenti non sono le conduzioni canoniche da un secondo.

Si riparte dalle coordinate continue, mantenendone la convenzione, e si
ricalcolano le celle per ciascuna griglia. Non si riutilizzano le vecchie
colonne dei bin. Le coordinate del modello sono su un campo fisso di 105x68 m.
Qui non servono stati di tracking: xT è stimato da eventi e luoghi, mentre i
modelli fisici e di scelta successivi usano `get_pv_players` per il frame corrente.

## Modello

Per origine i e destinazione j:

```text
q_i  = p_shot,i * p_goal,i
M_ij = p_pass,i*T_pass,ij + p_carry,i*T_carry,ij
xT   = q + M*xT
```

Le frequenze dei quattro eventi condividono un denominatore per cella.
La perdita ha valore futuro zero. Le transizioni sono condizionate al tipo
di evento della rispettiva tabella. Queste frequenze non sono la policy
di action selection né le probabilità del modello fisico di conduzione.

Lo smoothing delle transizioni riguarda le destinazioni, separatamente per
origine; il bandwidth in metri viene convertito in sigma lungo x e y. Le
origini mai osservate hanno una distribuzione uniforme delle destinazioni.
Per i tiri si smussano separatamente conteggi dei tiri e dei goal, poi si
calcola il rapporto. Le frequenze dei quattro tipi di evento non sono smussate.
La stessa costruzione viene conservata in questa revisione.

Bellman parte dalla superficie nulla e si ferma quando il massimo cambiamento
è sotto 1e-12. Non ci sono gradient descent, learning rate o alberi da addestrare.
Superficie e componenti sono sempre salvate insieme per controllarne la coerenza.

## Tre usi distinti dei dati

**Confronto delle griglie.** 8x5, 10x6, 12x8, 16x12 sugli stessi eventi.
Cinque split esterni con otto partite di fit e due test. Dentro gli otto,
leave-one-match-out per scegliere separatamente lo smoothing di passaggi,
conduzioni e goal dato tiro. Griglia dei bandwidth: 0, 3, 5, 8, 12, 18 m.

Lo score interno è event-weighted: NLL della densità di destinazione per le
transizioni, CE Bernoulli per i tiri. Fra i bandwidth entro un errore standard
appaiato dal migliore si sceglie il più grande. Il codice mostra esplicitamente
il jackknife sulle partite che produce quell'errore standard.

I test dei cinque fold vengono concatenati per il calcolo pooled delle
metriche. Il confronto delle risoluzioni usa CE e Brier del goal dato tiro,
con differenze CE appaiate e jackknife per partita. 10x6 è la più grossolana
fra quelle entro una SE dalla CE minima: un criterio esplorativo di parsimonia,
non una dimostrazione di equivalenza statistica. La sparsità e la SD descrittiva
delle mappe completano il confronto. Non si valida direttamente un nuovo target
di goal futuro dell'intero xT; gli score delle transizioni includono l'area
delle celle e non sono CE Bernoulli.

**Mappe per le feature.** Per ogni fold di pass selection si rifitta xT solo
sui suoi sei match di fit, inclusa la scelta dello smoothing. Validation e test
saranno valutati su quella superficie congelata. Questa procedura è separata
dalla validation 8+2 del confronto delle risoluzioni.

**Produzione.** Fit sulle dieci partite con griglia 10x6 e bandwidth scelti via
leave-one-match-out. Il riferimento accettato è 8 m per i passaggi e 5 m per
conduzioni e tiri, convergenza in 139 iterazioni. Nei sei-match fit di pass
selection, i tiri usano 8 m nei fold 0 e 4 e 5 m negli altri. La mappa di
produzione non sostituisce le mappe di fitting nel benchmark.

## Esecuzione e output

Dalla root del progetto, nell'ambiente `soccer`:

```bash
python pipeline/expected_threat/pipeline.py --stage validation
python pipeline/expected_threat/pipeline.py --stage scores
python pipeline/expected_threat/pipeline.py --stage maps
python pipeline/expected_threat/pipeline.py --stage production
```

Per una sola risoluzione: `--stage validation --grid 10 6`.
Per una sola mappa a sei match: `--stage maps --fold 0`.
`--stage all` esegue tutte le fasi. `--output` permette di scegliere una nuova
destinazione; le cartelle di risultato già presenti non vengono sovrascritte.

```text
pipeline/outputs/expected_threat/
  validation/<grid>/fold_0..4/  # componenti del fit sugli otto e scelta smoothing
  scores/                      # pooled, per-match, confronti, coverage, stabilità
  feature_maps/fold_0..4/       # componenti dei fit sui sei
  production/                  # componenti sulle dieci + heatmap con valori
```

Riferimento concettuale: [Singh, Introducing Expected Threat](https://karun.in/blog/expected-threat.html).
La separazione pass/carry e il ramo loss sono la formulazione di questo progetto.
`checks/verify_pipeline.py` controlla la corrispondenza con la procedura e gli
oggetti accettati, senza installare risultati nel pacchetto.
