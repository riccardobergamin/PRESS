# Convenzioni cinematiche configurabili della ball drive

La routine resta ritardo di rotazione a posizione fissa, poi traslazione
rettilinea accelerata. Le nuove opzioni preservano i risultati precedenti
quando omesse. I sorgenti precedenti sono archiviati, con hash, in
`old/ball_drive_physics_pre_closure_20261016/`.

|Parametro|Significato|Default compatibile con i pesi precedenti|
|---|---|---|
|`a_max`|Accelerazione futura della traslazione|Invariato|
|`a_turn`|Scala del ritardo `speed * theta / a_turn`|`None`: usa `a_max`|
|`preserve_initial_overspeed`|Mantiene `v0 > v_max` senza accelerare ulteriormente|`False`: convenzione storica|
|`boundary_mode`|`absolute`, `signed` o `clipped`|`absolute`|

Le opzioni cinematiche sono disponibili in `time_to_intercept` e `F`;
tutte sono inoltrate dal solver ball drive. L'API per frame le accetta
in `physics_kwargs`, insieme ai parametri del modello. Non modificano
automaticamente pitch control o pass probability.

```python
import jax.numpy as jnp
from PRESS.physics.times import time_to_intercept

tau = time_to_intercept(
    p=jnp.array([[0., 0.]]), v=jnp.array([[8., 0.]]),
    r_target=jnp.array([[16., 0.]]),
    v_max=7., a_max=7., a_turn=7.,
    preserve_initial_overspeed=True,
)
# Circa 2 s, mantenendo gli 8 m/s osservati.
```

Il prossimo benchmark mantiene `a_turn=a_max=7`; l'opzione indipendente
serve all'utente. Il ritardo non descrive un arco esplicito. L'accelerazione
osservata non è input del TTI: non esiste un ramo `a0 > a_max` da tagliare.
Usare a0 richiederebbe un modello di moto diverso.

Per un rettangolo centrato L×W, il margine signed è
`min(x+L/2, L/2-x, y+W/2, W/2-y)`: positivo dentro, negativo fuori.
Fuori da un corner non è la distanza euclidea dal vertice. `clipped`
satura a zero; `absolute` conserva il minimo delle distanze assolute.
Nessuna variante ferma il portatore o rende certa la loss all'uscita.

Queste opzioni **non costituiscono nuovi pesi allenati**. Cambiare dinamica
o bordo richiede un fit coerente; non applicare la spline storica a valori
negativi interpretandola come validata. L'angolo in lambda è un'estensione
distinta, ancora da allenare.

Parità dei default verificata contro i sorgenti archiviati: scarto zero
nei casi verificati per TTI e probabilità, Euler ed exponential, float64.
Risultati in `ball-drive-conventions-parity.json`. Le correzioni epsilon
dell'angolo storico sono conservate: in float64 possono lasciare circa
0,00005–0,00007s di ritardo per vettori esattamente allineati. Due vecchi
test analitici troppo stretti falliscono anche sullo storico in float64;
il confronto di parità è separato.45 test pertinenti passano nella
precisione standard float32; i nuovi casi sono verificati anche in float64.
