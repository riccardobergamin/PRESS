# Ball-drive risk: CatBoost

Stesso schema di action/residual: **features → training → pipeline**.
La precedente versione dei tre script è conservata in
`old/before_binary_review_20261110/`. Nessun modello del pacchetto viene sostituito.

## Ordine di lettura

1. `pipeline.run_pipeline`: vede tutto il flusso in tre fasi.
2. `features.py`: `calculate_defender_tti`, `build_spatial_features`,
   `build_features`; `prepare_features` le applica agli stati reali.
3. `training.train_model`: singolo fit **nativo** CatBoost.
4. `training.select_configuration`: ricerca completa sulle quattro rotazioni.
5. `training.train_validate_test`: calibrazione inner-OOF, refit su otto e test.
6. `training.compute_pooled_metrics`: concatena i cinque test prima delle metriche.

## Feature

Coorte finale invariata: **9.114 stati, 456 loss**. Target y=1 per la perdita
entro un secondo, anche quando avviene prima della fine della finestra.
Il modello fisico resta quello distribuito; CatBoost è il confronto esplorativo.

Le 50 feature sono le ultime approvate: velocità BC; bordo più vicino a
0/0.5/1 s; tre TTI minimi a quegli istanti; undici probabilità di encounter a
passo 0.1 s; posizione iniziale/proiettata e geometria rispetto alla porta;
distanza, velocità, closing, trasversale e posizione relativa dei tre difensori
inizialmente più vicini; due conteggi locali e geometria della coppia più vicina.

Lo stato viene ricostruito dal tracking con lo stesso getter degli altri task.
TTI usa vmax=7 e accelerazioni lineare/centripeta=7, con overspeed preservato.
La curva encounter usa **s della ball drive fisica dello split corrispondente**:
sei match nei fit interni, otto nel refit che predice il test.

La geometria è preparata una sola volta per entrambi i bracci e tutti i fold.
`--stage train` legge questi file senza riaprire tracking o ricalcolare TTI.

## Training

`train_model` passa X e y a `CatBoostClassifier.fit`, con `loss_function=Logloss`.
Nessun gradiente custom. Alberi simmetrici, boosting Plain, bootstrap Bayesian;
le altre impostazioni fisse e la grid sono scritte all'inizio di `features.py`.

La grid comprende **36 configurazioni**: depth=(4,6,8), learning rate=
(0.02,0.05,0.1), L2=(1,3,10,30). Per ogni configurazione si fittano 3.000 alberi
in quattro rotazioni 6+2 all'interno degli otto match di sviluppo. Si valutano
prefissi ogni 25 alberi. La CE pooled di validation sceglie configurazione e
lunghezza; nel braccio ricampionato usa la correzione analitica della prior.

- **Naturale:** tutte le osservazioni di training.
- **loss10:** tutte le loss e nove successi per loss, senza reinserimento.
- **Validation/test:** proporzioni naturali, nessun ricampionamento.

Con la configurazione fissata, tre seed (19,41,73) generano predizioni inner-OOF
che coprono gli otto match e tre modelli nuovi su tutti gli otto match.
Si mediano i tre seed. Platt e isotonic sono stimati sulle predizioni inner-OOF
e label naturali; i due test restano esclusi fino alla valutazione finale.
Si riportano nativo, correzione della prior, Platt e isotonic. Le metriche sono
pooled su cinque coppie di test. Non si rifitta un CatBoost all-ten per il pacchetto.

## Esecuzione

```bash
python review/ball_drive/catboost/pipeline.py --stage features \
  --physical-splits review/inputs/ball_drive/reference_parameters/physical_splits.json
python review/ball_drive/catboost/pipeline.py --stage train --fold 0 --sampling natural --threads 4
# Dopo i cinque fold di entrambi i bracci:
python review/ball_drive/catboost/pipeline.py --stage scores
```

Per una riproduzione completa, il percorso predefinito di `physical-splits`
è l'output della nuova stima fisica in `review/outputs/ball_drive/physical/`.
La copia accettata sopra è un'opzione esplicita. Gli output già presenti non
vengono sovrascritti. La versione del benchmark è CatBoost **1.2.10**.

## Verifica

`checks/verify_review.py` conferma identità di TTI, feature, sampling e metriche
rispetto alla versione precedente, sui 9.114 stati reali. Nessuna grid è stata
rilanciata. CatBoost non è installato nell'ambiente locale: non sono stati
eseguiti nuovi fit nativi, e il controllo non li presenta come verificati.
