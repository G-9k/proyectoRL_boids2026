# Reporte de avance: Track B (entorno, recompensa y líneas base)

**Fecha:** 19/11/2026. **Rama:** `entorno-gym-sb3` (todavía sin mergear a `main`).

## En una línea

El entorno de pastoreo ya está en Gymnasium, Strömbom está medido (100% de éxito sin cerco) y el DQN de SB3 está andando, pero todavía **no resuelve ni el nivel más fácil** (1 oveja). Una prueba trivial mostró que no es solo un problema de recompensa: los valores Q del DQN quedan sobreestimados y casi iguales entre acciones. El hito B2 sigue abierto.

## Qué había hecho Javier

Revisé `PastoreoDqn/` (es su repo, está en `.gitignore`). Resolvió **otra tarea** que la del plan: campo de 300 m con alambrado, tranquera, 30 ovejas en 3 grupos, y éxito = entran las 30. Armó una observación de 10 valores a mano, un currículum interno de dificultad (d de 0 a 1,5) y un Double DQN propio, adaptado de una plantilla de MinAtar. En `resumen_modelos.csv`:

| Dificultad | Rama A | Rama B |
|---|---|---|
| d = 1,0 | 72% | 85% |
| d = 1,5 | 55% | 72% |

Es un logro real. Lo que no sirve para nuestra pregunta:

- Iteró red y recompensa a la vez, justo lo que el plan quería separar.
- El corte por tiempo se trata como fin de episodio, y encima da bonus o castigo según cómo termine. La recompensa mezcla un avance telescópico (sin γ) con penalización por tiempo y premios terminales, así que no es shaping basado en potencial.
- Una sola semilla de entrenamiento: sus tests de significancia comparan dos redes, no dos métodos.
- Evaluaciones de 20 episodios, ruidosas, y sin comparación con Strömbom.
- Su entorno corre a ~1.500 pasos/s, contra ~4.000 del nuestro con 30 ovejas.

## Qué decidimos y por qué

Construir sobre el entorno de la raíz ([Pastoreo_flechitas.py](Pastoreo_flechitas.py)): física estándar de Strömbom, sin bordes, meta en el origen y éxito con el centro del rebaño a menos de 10 m. Es la pregunta del plan (¿DQN contra Strömbom?), ya tiene la heurística en [Pastoreo_clásico.py](Pastoreo_clásico.py), corre más rápido y es chico.

De Javier rescatamos sus hiperparámetros, su script de estadística (`evaluación.py`), la idea de premiar a la oveja más lejana y el alambrado como plan B.

## Qué construimos

| Archivo | Para qué |
|---|---|
| [pastoreo_env.py](pastoreo_env.py) | Entorno Gymnasium. Recompensa intercambiable (`esparsa`, `potencial_gcm`) y parámetros de currículum: ovejas, distancia del rebaño a la meta y del pastor al rebaño, ángulo del pastor, cerco, repetición de acción |
| [strombom_politica.py](strombom_politica.py), [medir_strombom.py](medir_strombom.py) | Strömbom como política, continua y discretizada a las 9 acciones |
| [entrenar_sb3.py](entrenar_sb3.py) | DQN de SB3, evaluado cada tanto en 50 mapas fijos. Nombres de corrida automáticos ([modelos/README.md](modelos/README.md)) |
| [resultados/](resultados/README.md) | CSV y JSON de cada corrida, con la tabla completa |

## Qué encontramos

**Strömbom** ([strombom_baseline.csv](resultados/strombom_baseline.csv), 100 episodios por fila):

- Sin cerco, 100% con 1, 3, 10 y 30 ovejas, en ~450–550 pasos.
- Discretizado a 9 acciones rinde igual: las 9 direcciones alcanzan.
- Con cerco falla solo con 1 oveja (74–77%): queda trabada en una esquina y el pastor no puede ponerse detrás. Con 3 ovejas y 9 acciones llega a 95%.

**DQN de SB3**, 1 oveja y 3 semillas por configuración ([tabla completa](resultados/README.md)):

| Rebaño a la meta | Variante | Éxito final | Azar |
|---|---|---|---|
| distancia original (~150 m) | k1 y k4 | 0% | 0% |
| 40 m | k1 | 6–16% | 7% |
| 40 m | k4 | 6–14% | 8% |
| 20 m | k4, sin cerco | 6–12% (picos de ~40%) | 16% |
| 20 m | k4, con cerco | 30–38% | n/d |
| 40 m, pastor detrás y alineado (trivial) | k4, con cerco, ambas recompensas | 26–36% | 36% (60% con k1) |

(k = veces que se repite cada acción. Strömbom resuelve el nivel trivial al 100%.)

- **Desde la distancia original no aprende nada.** Los valores Q dieron iguales para las 9 acciones: una acción mueve la oveja ~1 m, que vale ~0,007 de recompensa. El shaping es demasiado débil.
- Repetir la acción ×4 aceleró el entrenamiento pero no mejoró el éxito.
- Con el rebaño a 20 m llegó a ~40% y se derrumbó: el pastor aprendió a irse caminando para siempre, porque el mundo no tiene bordes. Con cerco queda estable, pero lejos del 100%.
- **En el nivel trivial, el DQN queda por debajo del azar.** La política aprendida es casi siempre la misma acción, alejándose de la oveja hacia una esquina. Con la recompensa esparsa el retorno máximo posible es 1, pero la red estima valores Q de 1,1 a 1,3: **sobreestimación**, con diferencias entre acciones de apenas 0,03–0,1. Es el sesgo del `max` que corrige Double DQN (el de SB3 es el DQN común; el de Javier era Double DQN). Una semilla tenía 92% antes de empezar a actualizar la red, y el entrenamiento la empeoró.

## Qué sigue

1. **Atacar la sobreestimación en el nivel trivial** antes de seguir con recompensas: Double DQN, γ más bajo (0,95), tasa de aprendizaje más baja y más entrenamiento. Mientras el nivel trivial no llegue a ~100%, comparar recompensas no tiene sentido.
2. **Abrir el ángulo del pastor** (0° → ±30° → ±90° → 180°) para ver dónde se rompe. Mi sospecha es que lo difícil de verdad es **ponerse detrás de la oveja**.
3. **Recompensa de compactación para muchas ovejas.** Un círculo de radio f(N) = r_a·N^(2/3) centrado en el centro del rebaño (el umbral que usa Strömbom para pasar de recolectar a arrear). Se gana al entrar una oveja y se descuenta al salir. Hay que escribirlo como shaping basado en potencial, con γ: sin ella, "entrar ahora y salir después" da ganancia neta con descuento y se puede farmear. Tiene un tope natural, porque apretar un rebaño que ya está adentro no suma. Variante continua: penalizar cuánto se pasa cada oveja del círculo.
4. **Criterio de éxito con muchas ovejas:** pasar a "centro en la meta y rebaño compacto", para que no valga un rebaño partido en dos con el centro justo en la meta.
5. **Más adelante:** DQN contra PPO (mismas 9 acciones y también con acción continua), y una ablación del replay buffer y la red target, mejor con el DQN propio del Track A en CartPole.
