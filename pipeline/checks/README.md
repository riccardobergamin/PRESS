# Controlli locali della pipeline

Il piano copre tutte le fasi; non ripete il training completo destinato al
cluster. Le ricerche XGBoost/CatBoost complete non fanno parte degli smoke test.

```bash
python pipeline/checks/run.py --output /path/to/new_checks
# Aggiungere --execute per eseguire, oppure --step pass_selection, ecc.
```

`--package-root` seleziona esplicitamente il checkout PRESS da verificare senza
modificare l'installazione Python esistente. `--raw-root` indica i dati pubblici.
Ogni comando ha un log e il riepilogo è in `checks.json`.

| Fase | Verifica |
|---|---|
| Sintassi e CLI | parsing di tutti gli script correnti e `--help` di ogni pipeline |
| Preprocessing | una partita reale: lettura completa, derivate, matching, segmentazione e coorti; confronto per chiavi/label |
| xT | eventi, quattro risoluzioni, smoothing, selezione, mappe per fold e produzione; confronto con gli oggetti accettati |
| Ball drive fisica | covariate, basi, prior, piccolo MAP, Laplace, predizione, metriche e caricamento dell'export |
| CatBoost | TTI, feature, sampling training-only e correzione prior, replay di calibrazione/metriche; il fit nativo richiede CatBoost |
| Pass probability | traiettoria su frame reali, TTI, PPCF, piccolo MLE multistart, Platt/isotonic, metriche e serializzazione |
| Pass selection | feature e destinazione, allineamento delle tre tabelle, probabilità, gradienti, curvatura, screening ridotto, alberi/ρ, early stopping, T, score ed export |
| Action/residual | feature, encoding, piccoli fit binari nativi, selezione, cross-fitting di P(carry), refit ed export |
| h/xLoss | ricomposizione di tutte le predizioni accettate, target e nulli per fold, CE/gain/ECE pooled e bootstrap delle quattro bande |
| Report | micropossessi, classificazione, pattern, squadre/giocatori, intervalli e tabelle dei casi reali |
| Rilascio | copia esplicita di solo codice e statistiche, senza input numerici |

## Pass selection

`pass_selection_steps.py` non riduce la griglia delle destinazioni: usa un
passaggio reale per partita con tutte le celle e i dieci nidi. Confronta le
feature con il builder del pacchetto; verifica `CNL(ρ=1)=flat`, normalizzazione
e gradienti tramite autodifferenziazione. La curvatura flat è verificata contro
la formula; quella CNL è la convenzione RUMBoost, non l'Hessiana esatta.
Il controllo di equivalenza già presente in `pass_selection/checks/verify_pipeline.py`
confronta inoltre gradienti, curvature e short fit con la versione accettata
prima della ristrutturazione delle tabelle.

Un fit a budget fisso di 22 alberi esercita l'update di ρ al ventesimo albero;
un secondo fit esercita la selezione del prefisso e del ρ corrispondente.
La temperatura è stimata sulle due partite di validation. Si verificano poi
le predizioni test, salvataggio/caricamento, importance e l'export di produzione
con un piccolo ensemble. Cinque short fold esercitano anche la copertura
pooled, la scelta del numero mediano di alberi e la temperatura OOF del
refit di produzione. Le feature fisiche congelate del piccolo fixture sono
input di test dell'interfaccia, non nuovi benchmark fold-specifici.
I suoi score non sono benchmark scientifici.

I test non attestano la convergenza del futuro training completo. Un test
nativo non eseguito perché manca una dipendenza deve essere segnalato,
non contato come superato.
