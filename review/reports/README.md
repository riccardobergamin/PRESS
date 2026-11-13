# 09 — Pattern, squadre, giocatori e case studies

Ordine: `features.py` → `scores.py` → `cases.py` → `pipeline.py`.

- `classify_possessions`: una riga per `(match_id, possession_id)`, stato
  iniziale/finale, micropossessi e tre pattern concordati.
- `pattern_shares`: quote sulla coorte completa e sui denominatori specifici.
- `entity_score`: E−O/100, numerosità, h medio, fallimenti osservati e quota High.
- `bootstrap_entity`: 2.000 ricampionamenti di possessi interi per gli intervalli;
  si conservano insieme tutte le azioni elevated di ciascun possesso estratto.
- `export_cases`: tabella dei checkpoint e, se richiesto, plot dei frame reali.
  Il frame del controllo successivo mostra l'esito ma non genera un nuovo h.

```bash
python review/reports/pipeline.py \
  --states /path/to/new_run/composed_risk/features/held_out.parquet \
  --output /path/to/new_run/reports \
  --tracking-root review/inputs/shared/tracking
```

Omettere `--tracking-root` genera soltanto le tabelle. Il caso illustrato deve
continuare a soddisfare il pattern; il codice lo verifica prima del plot.
I profili usano le predizioni held-out, mai i modelli di produzione.
