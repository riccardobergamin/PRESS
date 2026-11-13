# Ball drive: revisione del codice

Apri **ball_drive.py**. Contiene il calcolo del modello finale in un solo file,
con funzioni procedurali e sezioni nello stile degli script originari:

1. `load_parameters`: legge gli stessi parametri e basi spline salvati nel pacchetto.
2. `spline_basis`: applica le trasformazioni delle basi già stimate.
3. `tackle_rates`: costruisce lambda da velocità, distanza dal bordo e angolo.
4. `integrate_loss`: integra i rischi concorrenti con exponential Euler.
5. `ball_drive_risk`: collega i passaggi per un frame e un orizzonte generico.

Il TTI è calcolato da `PRESS.physics.times.time_to_intercept`, includendo la
rotazione e la gestione di velocità iniziali superiori al massimale.
Non vengono ristimati coefficienti, nodi o trasformazioni delle spline.

Questo file contiene il solver di predizione. Il codice di training e il
pacchetto installato non sono stati sostituiti.

Per la **stima e il benchmark**, apri [TRAINING.md](TRAINING.md): descrive i tre
file del modello fisico e i tre del confronto CatBoost finale. In ciascuna
cartella l'ordine di lettura è `model.py`, `training.py`, `pipeline.py`.

## Uso

Usa il checkout aggiornato `PRESS-portfolio-work` come sorgente del pacchetto.
Nell'ambiente locale esiste anche un'installazione che punta al vecchio checkout
`PRESS`: per il controllo seguente la sorgente aggiornata è scelta esplicitamente
tramite `PYTHONPATH`, senza cambiare l'installazione.

Dalla cartella `skillcorner-analysis`:

```bash
PYTHONPATH=../PRESS-portfolio-work \
JAX_ENABLE_X64=1 \
python review/ball_drive/check_equivalence.py \
  --cache review/inputs/ball_drive \
  --output review/ball_drive/checks
```

Il file di controllo serve solo alla verifica; non occorre leggerlo per seguire
il modello. Usa stati reali del campione finale e confronta con `BallDriveModel`.

## Controllo effettuato

- 151 frame reali distribuiti sulle dieci partite, inclusi casi estremi per
  velocità, prossimità dei difensori e distanza dal bordo.
- 223 confronti complessivi, includendo orizzonti 0, 0.017, 0.4, 1 e 1.4 s,
  e accelerazione di rotazione diversa da quella lineare.
- Scarto massimo della probabilità totale: circa `1.74e-7` in scala 0–1.
- Scarto massimo su tutti gli output confrontati: inferiore a `2e-7`.
- Verificate conservazione della probabilità e monotonia della sopravvivenza.

Gli scarti sono numerici: il richiamo diretto al TTI e l'aggiornamento scalare
possono arrotondare diversamente rispetto all'implementazione vettorializzata.
I dettagli sono in `checks/verification.json` e `checks/equivalence.csv`.
