# resultados/

Números de los experimentos. Todo esto va a git: son archivos chicos y es lo que se compara entre corridas.

## registro.csv: el log de todo lo que se probó

Cada corrida de [entrenar_sb3.py](../entrenar_sb3.py) agrega una fila al terminar (también si se corta con Ctrl+C). Es el primer lugar donde mirar para saber qué se probó y cómo dio.

| Columna | Qué es |
|---|---|
| `fecha`, `nombre` | cuándo y cuál corrida (el nombre lleva a la curva, la configuración y el modelo) |
| `escenario` | el escenario de [escenarios.yaml](../escenarios.yaml) que se usó; vacío si se pasó todo por argumentos |
| `cambios` | lo que se cambió por la terminal respecto del escenario, por ejemplo `gamma=0.95 seed=1` |
| `decisiones` | cuántas decisiones se entrenó |
| `eval_final`, `eval_max` | % de éxito en la última evaluación y en la mejor |
| `eval_max_decision` | decisión en la que se dio la mejor evaluación (con empate en éxito, la de menos pasos). Desde el 20/11 ahí se guarda `modelos/<nombre>_mejor.zip`; en las corridas anteriores se completó a partir de la curva |
| `qmax_final` | el mayor valor Q al final, crudo. Con shaping depende de lo lejos que esté la oveja: no sirve para comparar entre corridas |
| `qmax_sin_shaping_final` | el mismo valor sin la parte del shaping (Q + Φ). Con cualquier recompensa, el retorno real máximo es 1: **si pasa claramente de 1, hay sobreestimación**. Solo en corridas desde el 19/11 a la noche |
| `minutos`, `estado` | duración, y `listo` o `interrumpido` |

Las corridas anteriores al registro (las del 19/11 con fecha "reconstruido") se cargaron a partir de sus archivos y no tienen `qmax_final`.

## strombom_baseline.csv

La línea de base: el pastor de Strömbom dentro de `PastoreoEnv`. Lo genera [medir_strombom.py](../medir_strombom.py) con 100 episodios por fila, sobre los mapas de las semillas 10000 a 10099.

| Columna | Qué es |
|---|---|
| `politica` | `strombom_continuo` (se mueve en cualquier dirección, como en el paper) o `strombom_9_acciones` (elige la más parecida de las 9 acciones del DQN) |
| `n_ovejas`, `cerco` | configuración del entorno |
| `exito_pct` | % de episodios en que el centro del rebaño llegó a menos de 10 m de la meta antes de 6000 pasos |
| `pasos_media`, `pasos_p50`, `pasos_p95`, `pasos_max` | pasos de simulación hasta la meta, contando solo los episodios exitosos. El p95 es el valor que no supera el 95% de ellos |
| `segundos` | lo que tardó esa fila |

## entrenamientos/

Una corrida de [entrenar_sb3.py](../entrenar_sb3.py) deja dos archivos con el mismo nombre. El esquema del nombre está explicado en [modelos/README.md](../modelos/README.md), y el modelo entrenado queda en `modelos/<nombre>.zip`.

**`<nombre>.csv`: la curva de aprendizaje.** Cada fila es una evaluación durante el entrenamiento:

| Columna | Qué es |
|---|---|
| `paso` | decisiones del agente hasta ese momento. Con `k4`, cada decisión son 4 pasos de simulación |
| `exito_pct` | % de éxito en 50 mapas fijos (semillas 20000 a 20049), sin exploración |
| `pasos_media` | pasos de simulación promedio en los episodios exitosos (vacío si no hubo ninguno) |
| `segundos` | tiempo transcurrido desde que empezó el entrenamiento |
| `qmax` | el mayor valor Q que la red le da al estado de entrenamiento en ese momento, crudo (solo en corridas nuevas) |
| `qmax_sin_shaping` | lo mismo sin la parte del shaping; no debería pasar de 1 (solo en corridas nuevas) |

**`<nombre>.json`: la configuración completa.** Tiene los argumentos del script, los parámetros del entorno y los hiperparámetros del DQN. Sirve para reproducir la corrida o para cargar el modelo con el mismo entorno.

### Cómo comparar corridas

- **Por % de éxito y pasos, nunca por recompensa.** Cada variante de recompensa tiene otra escala, y con shaping la suma de recompensas de un episodio no es comparable.
- **Mirar las 3 semillas juntas** (`_s0`, `_s1`, `_s2`). En RL la diferencia entre semillas es enorme, y una sola no alcanza para sacar conclusiones.
- **Comparar contra una política al azar en el mismo nivel.** Por ejemplo, con el rebaño a 20 m el azar ya logra ~16%.

### Corridas hasta ahora (19/11/2026)

Todas con 1 oveja, recompensa `potencial_gcm`, pastor a 5–20 m del rebaño, `max_pasos=1000` y 3 semillas.

| Nombre (sin `_s{n}`) | Decisiones | Éxito al final | Al azar | Nota |
|---|---|---|---|---|
| `n1_potencial_gcm_reborig_pas20_k1` | 200k | 0% | 0% | Valores Q iguales para las 9 acciones: no aprendió nada |
| `n1_potencial_gcm_reb40_pas20_k1` | 200k | ~6–16% | 7% | Plano, no mejora |
| `n1_potencial_gcm_reb40_pas20_k4` | 100k | ~6–14% | 8% | Repetir la acción no alcanzó |
| `n1_potencial_gcm_reborig_pas20_k4` | 100k | 0% | — | |
| `n1_potencial_gcm_reb20_pas20_k4` | 60k | ~6–12% | 16% | Llega a ~40% y se desarma: el pastor aprende a irse para siempre |
| `n1_potencial_gcm_reb20_pas20_k4_cerco` | 60k | ~30–38% | — | El cerco evita que se vaya; estable, pero lejos de 100% |
| `n1_potencial_gcm_reb40_pas20_k4_cerco` | 60k | ~18–32% | — | |
