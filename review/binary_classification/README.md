# Action selection e residual risk

Una cartella, tre script. Il codice del pacchetto e i suoi pesi restano invariati.

1. **`features.py`**: definisce gli scalari, calcola le feature dai frame reali,
   poi aggiunge P(carry) per residual risk.
2. **`training.py`**: likelihood Bernoulli, fit nativo, grid/early stopping,
   valutazione, refit di produzione e feature importance.
3. **`pipeline.py`**: `run_pipeline` mostra la sequenza completa. La CLI in
   fondo permette di eseguire una sola fase o un solo fold.

## Percorso di lettura

```text
pipeline.run_pipeline
    features.prepare_action_features
        stato dal getter -> pitch control grezzo -> scelta della destinazione
        -> pass probability calibrata -> ball-drive risk -> BCOS
        -> build_features: scalari con nomi e formule esplicite
    training.train_validate_test
        sei match: encoding e training
        due match: grid e early stopping
        due match: test, dopo aver fissato modello ed encoding
    training.compute_pooled_metrics
    training.train_production_model

    features.prepare_residual_features
        riusa le feature action -> aggiunge P(carry) cross-fitted
        -> unisce esiti verificati e tempo trascorso osservato
    stesso training/valutazione/refit, con target e feature residual
```

Per seguire l'apprendimento parti da `training.train_model`: una chiamata
esplicita a `xgb.train` usa la likelihood nativa `binary:logistic`.
Con score f, p=sigmoid(f), le derivate sono g=p−y e h=p(1−p). Non ci sono
derivate custom, resampling o pesi di classe. L'intercetta iniziale segue
il comportamento nativo di XGBoost; non imponiamo l'inizializzazione delle
utilità usata nel modello categoriale di pass selection.

## Cosa cambia tra i task

| | Action selection | Residual risk |
|---|---|---|
| Stato | Decisione canonica: passaggio o carry tentato | Inizio di un carry canonico riuscito da 1 s |
| y=1 | Passaggio scelto | Perdita al termine del possesso individuale |
| Coorte finale | 16.918 stati: 7.770 pass, 9.148 carry | 7.820 stati: 1.423 perdite terminali |
| Feature | 28 numeriche + start type e previous action | Stessa famiglia; tempo trascorso al posto del conteggio carry; P(carry) in più |
| Calibrazione distribuita | Nessuna: probabilità native | Nessuna: probabilità native |

Gli input sono le coorti già approvate. Qui non si applicano nuovi filtri.
In particolare l'esito finale non filtra action selection. Residual usa gli
esiti verificati e mantiene falli subiti/clearance secondo le convenzioni finali.

## Feature, una alla volta

`build_features` restituisce un dizionario di scalari nominati. Il dizionario
non contiene matrici nascoste: ogni chiave è una colonna della tabella finale.
I vettori di scelta q e completamento p hanno una voce per destinazione.
I pesi delle ipotesi riuscita/fallita sono q·p e q·(1−p); le medie pesate non
vengono rinormalizzate per il solo ramo. Conservano quindi la sua probabilità.

La pass selection riceve **pitch control grezzo**. Le sintesi success/failure
ricevono **pass probability dopo isotonic**. Il modello fisico restituisce
drive loss; il classificatore riceve **1−drive loss**, cioè carry success.
P(carry) è un'altra quantità: probabilità di *scegliere* la conduzione.

Gli stati sono letti con `review/frame_states.state_at_frame`, che richiama
`PRESS.utils.state.get_pv_players`. Layout: dieci compagni, portatore, undici
avversari. Il frame e il portatore sono quelli identificati dall'azione.

BCOS usa il valore di s del ball drive di quel fold; vmax portatore=6.6,
vmax difensore=7 m/s e scale di accelerazione=7 m/s². Raggio 40 m, griglia
di area 1 m, passo del percorso 0.03125 m, dt=0.04 s, T=15 s. Il calcolo
completo rimane costoso: non viene rilanciato dai controlli brevi.

## P(carry) per residual risk

`cross_fit_carry_probability` mostra esplicitamente i gruppi di partite:

- **Training del fold:** tre modelli action su quattro dei sei match;
  ciascuno predice i due esclusi. Configurazione e lunghezza sono quelle del
  modello action selezionato nel fold; i vocabolari sono ristimati sui quattro.
- **Validation/test:** predizioni native del modello action sui sei match
  dello stesso fold, senza rifit su otto e senza usare il modello di produzione.
- **Produzione:** cinque modelli action su otto match, ciascuno predice i due
  esclusi. All'inferenza il pacchetto usa il modello action di produzione.

Gli upstream fisici, pass choice, xT e BCOS rimangono quelli del relativo
stack esterno: nei cross-fit interni si riaddestra soltanto il classificatore
action. La provenienza di ogni gruppo è salvata accanto al modello.

La precauzione riguarda il classificatore action perché si allena sugli stessi
checkpoint carry poi usati da residual: predizioni in-sample potrebbero
riconoscere la scelta osservata meglio di quanto farebbero su una partita nuova.
Pass selection si allena sui rilasci dei passaggi; i modelli fisici sono più
vincolati e xT è una superficie aggregata. Questo è un cross-fitting mirato
al classificatore action, non una ristima fuori campione dell'intera catena.
Le tre rotazioni non scelgono configurazione o alberi: usano quelli già
selezionati dal modello action principale sulle sei più due partite.


## Training e refit

La grid completa è il prodotto di depth=(3,4,6,8), child weight=(1,5,20),
L2=(0.1,1,3,10), eta=(0.03,0.05,0.1): **144 configurazioni per fold**.
CPU hist, max_bin=256, subsample/colsample=0.8, limite 2.000 alberi, patience
100 e seed 20261026+fold. Il target resta alla proporzione naturale.

La coppia di validation è fissa per tutte le configurazioni del fold.
La CE nativa sceglie configurazione e prefisso. Non ci sono rotazioni interne
6+2 o un refit su otto prima del test. Vocabolario e one-hot vengono appresi
sui sei match e applicati invariati a validation/test. I cinque test vengono
concatenati prima di calcolare CE, gain, Brier, AUC ed ECE a dieci bin di rango.

Per la produzione, `select_production_configuration` sceglie la configurazione
con CE media di validation minima sui cinque grid. La lunghezza è la mediana
dei cinque prefissi **di quella stessa configurazione**, non dei cinque vincitori.
Ritrova 277 alberi per action selection e 118 per residual risk. Si rifitta
da zero sui dieci match, senza early stopping né scaling.

`save_feature_importance` usa **total_gain**: somma dei miglioramenti degli
split di una feature attraverso tutti gli alberi, poi normalizzazione.
Salva sia le colonne codificate sia le categorie riaggregate.

## Input e comandi

Input in `review/inputs/binary_classification/`, tracking condivisi in
`review/inputs/shared/tracking/`. I modelli upstream sono copie dei finali
approvati, per fold e produzione. Le nuove esecuzioni scrivono soltanto in
`review/outputs/binary_classification/<task>/`.

```bash
# Esempio di fasi separate; non avviare il calcolo completo per leggere il codice.
python review/binary_classification/pipeline.py --task action_selection --stage features --fold 0
python review/binary_classification/pipeline.py --task action_selection --stage train --fold 0 --threads 4

# Dopo i cinque fold action e le loro feature, costruire quelle residual.
python review/binary_classification/pipeline.py --task residual_risk --stage features --fold 0 --threads 4
python review/binary_classification/pipeline.py --task residual_risk --stage train --fold 0 --threads 4
```

`--stage scores` aggrega i cinque test. `--stage features --production-features`
prepara gli input all-ten; `--stage refit` crea il modello pubblico nella
cartella di revisione. Completa action selection prima di preparare residual
risk, anche per produzione. `--stage all --task both` esegue tutto nell'ordine.
Gli output esistenti non sono sovrascritti.

## Verifiche eseguite

`checks/verify_review.py`: confronto delle formule su due stati reali, encoding
identico ai reader pubblici, fit nativi brevi, cross-fitting 4+2 e replay delle
metriche finali. Le metriche riprodotte sono CE=0.5685917102 per action e
CE=0.4195055303 per residual. Non è stata rilanciata la grid completa.
I controlli del refit/export sono separati in `checks/verify_production.py`:
reader ed ensemble coincidono con una chiamata nativa locale sugli stessi
input, per entrambe le architetture binarie. Questi refit temporanei non
riproducono esattamente le predizioni dei pesi Orfeo archiviati; il controllo
registra lo scarto, senza attribuirlo a una causa non verificata. Non sono
stati sostituiti i pesi approvati né le predizioni del benchmark. La grid
completa e un nuovo benchmark non sono stati eseguiti.


La prova con un unico modello action anche sui dati di training residual è
conservata separatamente in `review/experiments/residual_single_action_model_20261110/`.
È stata scartata dopo il confronto Orfeo: restano attivi procedura, benchmark
e pesi originali. Il relativo codice è in `old/single_model_rejected_20261110/`.
