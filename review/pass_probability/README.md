# Pass probability — procedural review

Versione separata della specifica finale accettata: **vmax = 5.5 m/s, a = a_turn = 7 m/s², gestione v > vmax, MLE sulle traiettorie, PPCF con integrazione esponenziale e calibrazione isotonic**. Non modifica il pacchetto, le pipeline esistenti o i pesi distribuiti.

## Ordine di lettura

1. **`model.py`** — equazioni e covariate fisiche.
   - Fit di v0, theta e phi sulle prime cinque posizioni a 0, 0.1, 0.2, 0.3
     e 0.4 s, più proiezione dalla media delle cinque velocità. Il confronto
     avviene durante la generazione; si salva una sola traiettoria per passaggio.
   - Proiezione del volo con le funzioni fisiche di PRESS.
   - TTI lungo il percorso e al punto di destinazione, con la stessa routine in tutti i passaggi.
   - Selezione del volo ipotetico per il PPCF al punto finale.
   - Integrazione esplicita dei rischi concorrenti e likelihood Bernoulli. **Qui non ci sono prior.**
2. **`training.py`** — prima ottimizzazione, poi selezione e calibrazione, refit di produzione e infine metriche.
   - Tre ottimizzazioni BFGS complete da `(lambda, s) = (4.3, .43), (3, .3), (5, .5)`; nessun learning rate fisso.
   - Quattro rotazioni interne 6+2 per ottenere i punteggi PPCF fuori campione sulle otto partite di sviluppo.
   - Fit di isotonic e Platt su questi punteggi. Refit fisico sulle otto e valutazione sulle due partite esterne.
   - Refit fisico sulle dieci per la produzione; isotonic stimata sui punteggi raw dei cinque test esterni, non sui punteggi in-sample del refit.
   - Pooled-out CE, gain sul nullo, Brier, ECE a dieci bin di uguale numerosità e AUC.
3. **`pipeline.py`** — `generate_trajectories` stima le traiettorie di tutti i
   passaggi dal tracking, confronta gli errori medi di posizione rispetto al moto
   reale e salva solo la candidata scelta; `prepare_features` carica questi
   risultati e gli stati iniziali, calcola TTI e input PPCF e li salva.
   Il training riutilizza poi questi input per tutti i fold.

Gli stati iniziali vengono ricostruiti dal tracking tramite
`review/frame_states.py`, che usa `get_pv_players` al rilascio. Il vecchio
`arrays.npz` conserva gli stati per confronto; la pipeline ne legge soltanto
la destinazione osservata `r_target`, che non è uno stato futuro dei giocatori.

Le funzioni della procedura sono contenute in questi tre file. Le routine fisiche
condivise sono `PRESS.physics.times.time_to_intercept`,
`PRESS.physics.flight_ball.ball_trajectory` e `generate_ball_trajectories`.
La scelta del tempo di volo è esplicita in `select_flight`, dentro
`prepare_pitch_control_inputs`: interpola tempo e altezza dei voli candidati,
scarta quelli non disponibili e sceglie l'arrivo più vicino al TTI del ricevente.

Per seguire il pitch control: `prepare_pitch_control_inputs` prepara TTI, tempo
di volo e maschera dei giocatori; `integrate_pitch_control_logs` integra le
probabilità concorrenti; `predict_pitch_control` restituisce il punteggio PPCF
grezzo della squadra in possesso. La calibrazione avviene successivamente.

## Due probabilità da distinguere

La MLE usa la probabilità di completamento lungo la **traiettoria ricostruita**. Dopo la stima, gli stessi lambda e s entrano nel **PPCF al punto finale**, calcolato dal solo stato iniziale e da un volo ipotetico. È quest'ultimo punteggio che viene calibrato e usato operativamente. La calibrazione non ristima lambda e s.

La probabilità di fallimento include sia il controllo avversario sia la massa residua non controllata all'orizzonte. Il passatore è escluso dalla competizione; gli altri giocatori, inclusi i portieri, possono controllare la palla.

Si conserva la discretizzazione della run accettata: 251 campioni di traiettoria da 0 a 10 s con passo 0.04 s, ciascuno con il proprio aggiornamento sinistro (orizzonte effettivo della somma 10.04 s); PPCF al punto finale con 1,500 aggiornamenti di 0.01 s, fino a 15 s. Questa riscrittura non cambia la convenzione numerica.

## Input locali

La coorte accettata comprende **6,706 passaggi: 5,742 riusciti e 964 falliti**, durata tra 0.5 e 10 s inclusi, senza selezionare soltanto quelli dritti.

```text
review/inputs/pass_probability/
    rows.parquet             # chiavi, esito, rilascio e controllo successivo
    arrays.npz               # stato iniziale, nello stesso ordine delle righe
    boundaries.parquet       # copia di riferimento per i controlli numerici
    matches/<match_id>/
      boundaries.parquet
      trajectories.npz       # copie di riferimento, non lette dalla pipeline
```

`arrays.npz`: `p_pos`, `p_vel` hanno forma `(N,22,2)`; `x_bc`, `r_target` hanno forma `(N,2)`. Ordine dei giocatori: dieci compagni, passatore all'indice 10, undici avversari. Posizioni in metri e velocità in m/s; padding a distanza >= 500 m, mascherato.

Nel codice di training i nomi degli oggetti indicano il loro contenuto:

- `pass_table`: tabella dei passaggi, con partita ed esito osservato.
- `model_inputs`: dizionario degli array precalcolati di TTI, altezza utile al
  controllo e giocatori validi; contiene anche gli input del PPCF alla destinazione.
- `parameter_estimates`: stime di lambda e s con diagnostica della MLE.

`fit_on_matches` seleziona le righe delle partite di training, estrae TTI,
maschere ed esiti, esegue la MLE e registra il risultato.

La selezione e la verifica dei passaggi sono a monte: qui si leggono **gli stessi
casi finali**, senza introdurre filtri ulteriori. Questa revisione confronta i
candidati **per singolo passaggio nella generazione**, usando l'errore medio
3D da 0.7 s fino a prima del controllo successivo, entro 10 s. Conserva solo
la traiettoria con errore minore. A parità di errore, o se mancano frame di
confronto, preferisce il fit delle posizioni quando utilizzabile; altrimenti
usa la media delle velocità.

Il confronto aggregato per split di training resta nella pipeline precedente.
Il criterio per passaggio è una modifica del protocollo: può cambiare lambda,
s e calibrazione. Le metriche archiviate non valutano questa nuova procedura;
per confrontarla occorre eseguire nuovamente training e validazione.

Il tracking è sempre usato per stimare le traiettorie. Il percorso predefinito è:

```text
review/inputs/shared/tracking/
```

`--tracking-root` permette di indicare un'altra cartella. La fase `features`
esegue sempre prima `generate_trajectories`, poi `prepare_features`.

## Esecuzione esplicita per fasi

Gli input sono raccolti in [../inputs/README.md](../inputs/README.md).
`--data-root` e `--output` hanno già come default queste cartelle di input e
output, risolte rispetto agli script. Possono essere sostituite esplicitamente.

Dalla cartella `skillcorner-analysis`, con l'ambiente scientifico attivo e **la versione corrente di PRESS** nel percorso Python:

```bash
export PYTHONPATH=/Users/riccardobergamin/Documents/GitHub/PRESS-portfolio-work
export JAX_ENABLE_X64=1

python review/pass_probability/pipeline.py --stage features \
  --data-root review/inputs/pass_probability \
  --output review/outputs/pass_probability

# Un fold oppure tutti i cinque, omettendo --fold.
python review/pass_probability/pipeline.py --stage train \
  --fold 0 --output review/outputs/pass_probability

# Quando sono terminati tutti i cinque fold.
python review/pass_probability/pipeline.py --stage scores \
  --output review/outputs/pass_probability

python review/pass_probability/pipeline.py --stage refit \
  --output review/outputs/pass_probability
```

`--stage all` esegue l'intera procedura. Le fasi scrivono in sottocartelle nuove e rifiutano di sovrascrivere quelle già presenti. Il refit salva un JSON compatibile con `PassRiskModel`, ma **non lo installa** nel pacchetto.

```text
review/outputs/pass_probability/
  trajectories/<match_id>/   # trajectory.npy e diagnostics.parquet
  features/                  # pass_table e cinque array di input
  training/fold_0 ... fold_4/ # fit, punteggi interni e predizioni test
  scores/                    # pooled predictions, metriche e calibrazione
  production/                # nuovo pass_risk_model.json
```

## Controlli della riscrittura

`check_equivalence.py` è un controllo separato: può importare il codice precedente per confrontarlo con questi tre file; non è una dipendenza della pipeline. Usa casi reali, un piccolo fit MLE e le predizioni held-out archiviate. Risultati in `checks/equivalence.json`.

Non viene rilanciato automaticamente il benchmark completo, né sostituito alcun artefatto di produzione.

Controlli eseguiti:

- 36 passaggi reali, con diversi esiti, partite e velocità: identici TTI, tempi di volo, log-probabilità, NLL, gradienti e Hessiana rispetto al codice di riferimento.
- MLE completa dalle tre partenze sullo stesso campione: identici parametri e obiettivo.
- Ricostruzione con il solo fit delle posizioni: confronto delle traiettorie con il riferimento, incluso il fallback quando presente.
- Scelte precedenti controllate: `fit_positions` in tutti i cinque fit esterni
  e in 19 dei 20 interni; l'eccezione è registrata nei risultati del controllo.
- Applicazione dei calibratori salvati: scarto massimo inferiore a `3e-16`.
- Ricostruita la isotonic di produzione dai punteggi raw outer-OOF: stessi nodi e valori.
- Confronto con il solver operativo del pacchetto: scarto raw massimo circa `8.1e-9`.
- Ricalcolo delle metriche sulle 6,706 predizioni archiviate: CE `0.13112077`, gain `68.1917%`, ECE `0.18331%`, coincidenti con la specifica finale.
- Fase di feature e lettura del relativo cache provate su 20 passaggi reali delle dieci partite; aggregazione degli score provata sulle predizioni archiviate complete; verificato il rifiuto della sovrascrittura.

Questi controlli verificano la riscrittura e il ricalcolo delle metriche. **Non equivalgono a un nuovo addestramento integrale dei cinque fold.** I pesi di produzione sono rimasti invariati.
