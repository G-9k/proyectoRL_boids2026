# resultados/

Números de los experimentos. Todo esto va a git: son archivos chicos y es lo que se compara entre corridas.

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
