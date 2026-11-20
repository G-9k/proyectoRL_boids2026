# Reporte de avance: Track B (entorno, recompensa y líneas base)

**Fecha:** 20/11/2026 (actualizado; incluye la fase 3 y los cambios de red y observación). **Rama:** `entorno-gym-sb3` (todavía sin mergear a `main`).

## En una línea

Con **Double DQN + una penalización por tiempo**, el DQN ya resuelve 1 oveja al 100% en las 3 semillas, incluso con el pastor arrancando del lado equivocado (180°) y en casi la misma cantidad de pasos que Strömbom. Quedan dos problemas: **a veces olvida lo aprendido** (pasa de 100% a 0% después de mucho entrenamiento; por ahora lo cubrimos guardando el mejor checkpoint) y **desde la distancia del paper (~150 m) no aprende nada**.

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
| [pastoreo_env.py](pastoreo_env.py) | Entorno Gymnasium. Recompensa intercambiable (`esparsa`, `potencial_gcm`, `potencial_gcm_tiempo`) y parámetros de currículum: ovejas, distancia del rebaño a la meta y del pastor al rebaño, ángulo del pastor, cerco, repetición de acción |
| [strombom_politica.py](strombom_politica.py), [medir_strombom.py](medir_strombom.py) | Strömbom como política, continua y discretizada a las 9 acciones |
| [entrenar_sb3.py](entrenar_sb3.py) | DQN de SB3, evaluado cada tanto en 50 mapas fijos. Escenarios con nombre en [escenarios.yaml](escenarios.yaml) (`python entrenar_sb3.py angulo_90`), registro de todas las corridas en [resultados/registro.csv](resultados/registro.csv), y guarda el **mejor checkpoint** además del modelo final. Red, buffer, tasa de aprendizaje y observación se eligen por la terminal (`--red 256,256`, `--buffer`, `--lr`, `--observacion`); ver el [README](README.md) |
| [double_dqn.py](double_dqn.py) | Double DQN como subclase del DQN de SB3: cambia una sola línea del objetivo (la red principal elige la acción y la target la evalúa). Se activa con `--double` |
| [ver_agente.py](ver_agente.py) | Animación de cualquier modelo entrenado, con el mismo dibujo que `Pastoreo_flechitas.py`. Lee el entorno de la corrida, así que sirve para cualquier escenario. `--strombom` muestra la heurística en los mismos mapas |
| [simulador.py](simulador.py) | Simulador interactivo: cualquier modelo en cualquier escenario, con pausa, paso a paso, velocidad en pasos por segundo, control manual y los Q de cada acción en vivo |
| [resultados/](resultados/README.md) | CSV y JSON de cada corrida |

## Qué encontramos

**Strömbom** ([strombom_baseline.csv](resultados/strombom_baseline.csv), 100 episodios por fila):

- Sin cerco, 100% con 1, 3, 10 y 30 ovejas, en ~450–550 pasos.
- Discretizado a 9 acciones rinde igual: las 9 direcciones alcanzan.
- Con cerco falla solo con 1 oveja (74–77%): queda trabada en una esquina y el pastor no puede ponerse detrás. Con 3 ovejas y 9 acciones llega a 95%.

### 1. El DQN común no resolvía ni el nivel trivial (19/11)

Nivel trivial: 1 oveja a 40 m de la meta, pastor detrás y alineado, con cerco. Strömbom lo resuelve al 100% en ~32 pasos y el azar saca 36%. El DQN de SB3 quedó en 26–36%, por debajo del azar. La política era casi siempre la misma acción, alejándose hacia una esquina. Con un retorno máximo posible de 1, la red estimaba valores Q de 1,14–1,29 en 2 de 3 semillas, con apenas ~0,002–0,013 de diferencia entre la mejor acción y la segunda: **sobreestimación** por el `max` del objetivo.

Antes, con el rebaño más lejos, tampoco había aprendido: desde ~150 m los valores Q daban iguales para las 9 acciones (el shaping es demasiado débil), y sin cerco el pastor aprendía a irse caminando.

### 2. Double DQN + penalización por tiempo lo resuelve (fase 1)

Nivel trivial, 100k decisiones, 3 semillas. Se compara el promedio de las **últimas 3 evaluaciones**, para premiar la estabilidad y no un pico suelto:

| Variante | Últimas 3 evals | Al final (s0 / s1 / s2) |
|---|---|---|
| Double DQN | 90% | 100 / 100 / 100 |
| **Double DQN + penalización por tiempo** | **100%** | **100 / 100 / 100**, en 31–34 pasos |
| ídem + tasa de aprendizaje 1e-4 | 56% | 64 / 84 / 46 |
| ídem + γ = 0,95 | 77% | 100 / 100 / 100 |

La penalización por tiempo es −0,01 por decisión, sumada al shaping por potencial del centro del rebaño. No es shaping (cambia la tarea a "llegar rápido"), pero es seguro: la única forma de terminar el episodio es ganar. Aporta dos cosas: quedarse quieto deja de ser gratis, y los estados donde el agente no llega a la meta dejan de parecer buenos. Con esta variante, Q sin la parte del shaping queda en 0,89–0,96. Está dentro de lo posible (el valor real va de ~0,85 a 1 según el estado), cuando antes daba 1,1–1,3, imposible.

Bajar la tasa de aprendizaje o γ no ayudó.

### 3. Aprende a ponerse detrás de la oveja (fase 2)

Misma variante, con el pastor cada vez más desviado de la línea oveja–meta:

| Escenario | Decisiones | Primer 100% (s0 / s1 / s2) | Al final | Pasos (Strömbom ≈ 32) |
|---|---|---|---|---|
| ±30° | 150k | 60k / 70k / 60k | 98 / 100 / 6 | 31–32 |
| ±90° | 250k | 90k / 100k / 80k | 0 / 100 / 0 | 33–35 |
| 180° (cualquier lado) | 400k | 140k / 100k / 140k | 100 / 0 / 100 | 38–40 |
| distancia del paper, sin cerco | 500k | nunca | 0 / 0 / 0 | — |

- **Las 9 corridas con ángulo llegaron al 100%**, incluso con el pastor del lado equivocado. A 180° tarda unos pasos más porque primero tiene que rodear a la oveja.
- **El problema nuevo es el olvido.** Las corridas que terminaron en 0% no fallaron por no aprender: estuvieron en 100% durante 50k–140k decisiones y después se derrumbaron. En cada caso, Q sin shaping pasa de 1 y sigue subiendo (hasta 8–9) **antes** de que caiga el éxito, así que sirve de alarma. Double DQN reduce la sobreestimación, pero no la elimina con entrenamientos largos. Mi hipótesis, todavía sin comprobar: cuando ya resuelve todo, el replay buffer (100k) se llena de episodios cortos y exitosos, y la red pierde el ancla en los estados malos.
- **Desde la distancia del paper no despega.** Nunca encuentra la meta explorando: la oveja está a ~150 m, no hay cerco y el azar también saca 0%. Hace falta un currículum de distancia, no más entrenamiento.

Como respuesta inmediata al olvido, [entrenar_sb3.py](entrenar_sb3.py) ahora guarda el **mejor checkpoint** (`modelos/<nombre>_mejor.zip`, el de la evaluación con más éxito). Volví a correr ±90° con 250k decisiones. Las curvas salieron idénticas a las de la fase 2 (con la misma semilla, el entrenamiento es reproducible), así que las semillas 0 y 2 se volvieron a derrumbar. Esta vez quedó guardado el mejor modelo, que se elige con los mismos 50 mapas de la evaluación. Para descartar que fuera suerte de esa selección, probé en **100 mapas nuevos** (semillas 30000–30099):

| ±90° | s0 | s1 | s2 |
|---|---|---|---|
| Mejor checkpoint | 100% | 100% | 100% |
| Modelo final | 0% | 100% | 0% |

El checkpoint resuelve el problema práctico (tener un buen modelo), pero no la causa: el entrenamiento sigue siendo inestable.

### Cómo se ve: el agente a 180°

[videos/ang180_s0_v2_mejor.gif](videos/ang180_s0_v2_mejor.gif)

Es el mejor checkpoint del escenario 180° (semilla 0), jugando 5 mapas de evaluación seguidos (semillas 20000 a 20004): los resolvió todos, en 32 a 61 pasos. Se generó con `python ver_agente.py n1_potencial_gcm_tiempo_reb40_pas10_ang180_k4_cerco_ddqn_s0_v2 --episodios 5 --guardar videos/<nombre>.gif`.

Cómo leerlo:

- **Punto rojo:** el pastor, con su recorrido en rojo claro. **Flecha oscura:** la oveja, hacia donde camina. **Cruz azul:** el centro del rebaño (con 1 oveja coincide con ella), con su recorrido en azul. **Círculo verde:** la meta (10 m alrededor del origen).
- **Línea punteada roja grande:** el radio de 65 m dentro del cual la oveja siente al pastor y se aleja de él. **Círculo rojo chico:** la distancia de 6 m a la que, según Strömbom, el pastor debería frenar para no dispersar al rebaño.
- **Recuadro:** paso de simulación, la acción que eligió la red, su valor Q sin shaping (el retorno que espera, como mucho 1) y la distancia del centro a la meta.
- La vista hace zoom al área de cada episodio. La línea marrón (abajo y a la izquierda) es el cerco.

Lo que muestra: el pastor arranca en cualquier lado de la oveja. En el primer y el último mapa arranca casi del lado opuesto (desviado 161° y 178° de la posición "detrás" respecto de la meta). **Primero la rodea hasta ponerse del lado opuesto a la meta y recién ahí la empuja en línea recta.** En unos 20 pasos baja a menos de ~20° de desvío y se mantiene ahí hasta llegar. Es la maniobra de arreo de Strömbom, aprendida solo a partir de la recompensa. Por eso esos dos mapas tardan más (46 y 61 pasos, contra 32–34 en los que arranca bien ubicado). Durante el arreo va pegado a la oveja, a 9–15 m en promedio, y casi no usa la acción "quieto" (0–12% de las decisiones).

### 4. Lejos de la oveja: recompensa por la posición del pastor (fase 3, 20/11)

Desde lejos, el potencial del centro del rebaño no da ninguna señal: si el pastor está a más de 65 m, la oveja no reacciona y nada de lo que haga el pastor cambia la recompensa. Probamos un segundo potencial que premia la **posición del pastor**: detrás del rebaño respecto de la meta y cerca:

$$\Phi_{\text{pos}} = \frac{1+\cos\theta}{2} \cdot \frac{50}{50 + d}$$

con $\theta$ el ángulo pastor–rebaño–meta y $d$ la distancia pastor–rebaño. Se suma al del centro del rebaño con un peso. Como es potencial, no se puede farmear. Además encontramos que lejos de la meta el shaping del centro del rebaño da +(1−γ)·d/L por decisión con el pastor quieto, lo que se come buena parte de la penalización por tiempo. Así que probamos también una penalización doble.

Escenario `rebano_80m` (oveja a 80 m, pastor en cualquier lado, sin cerco; Strömbom 100% en 253 pasos, azar 4%), 300k decisiones, 3 semillas:

| Recompensa | Últimas 3 evals | Mejor checkpoint en 100 mapas nuevos |
|---|---|---|
| Actual | 9% | 87 / 29 / 22 |
| Penalización doble | 59% | 100 / 62 / 57 |
| **Posición, peso 0,5** | **68%** | **80 / 95 / 99**, en 151–189 pasos |
| Posición, peso 1 | 7% | 44 / 4 / 82 |

- **La recompensa de posición funciona con peso 0,5** y es contraproducente con peso 1. Con peso alto, llegar a la meta casi no paga en el último paso: se pierde el potencial de posición, porque Φ(terminal) = 0.
- La penalización doble también ayuda: la anulación pesaba.
- En el escenario del paper aparecen los primeros éxitos (mejor checkpoint: 45% de mapas nuevos con peso 1), pero inestables.

**Buffer de 1M contra el olvido:** no lo evita. En ±90°, con 1M (nada se descarta en 250k decisiones), 2 de 3 semillas igual se derrumbaron. La sospechosa principal sigue siendo la sobreestimación: Q sin shaping pasa de 1 antes de cada derrumbe.

Explicación detallada de todos los conceptos en [GUIA_CONCEPTOS.md](GUIA_CONCEPTOS.md).

### 5. Cambios en la red y la observación (20/11, todavía sin resultados)

- **Red configurable.** La red es un MLP de 2 capas ocultas de 128 neuronas con ReLU (18.825 parámetros). Ahora se cambia con `--red`, por ejemplo `--red 256,256`, y el nombre de la corrida lleva `red256x256`.
- **Observación normalizada.** La última de las 8 entradas, la distancia a la oveja más cercana, se dividía por 6 m: con el pastor a 80 m valía ~13, y a la distancia del paper, ~25, mientras las otras rondan entre −1 y 1. Una entrada desproporcionada le complica el aprendizaje a la red, y justo fallamos en los escenarios lejanos. Ahora se divide por r_s = 65 m. Es una sospecha, no un resultado: falta medir si cambia algo.
- **Consecuencia:** todos los modelos anteriores, incluidos los que resuelven `rebano_80m`, usan la observación vieja (`original`). Siguen funcionando en `ver_agente.py` y el simulador, que la detectan solos, pero no se pueden mezclar con modelos nuevos. Para el currículum hay que reentrenar primero `rebano_80m` con la observación nueva, o arrancar con `--observacion original`.

## Qué sigue

1. **Escenario del paper:** posición 0,5 + penalización doble juntas, y currículum, arrancando desde los modelos que resuelven `rebano_80m`. Antes: repetir `rebano_80m` con la observación normalizada (sección 5), que además dice si normalizar ayuda.
2. **Atacar el olvido por el lado de la sobreestimación:** bajar la tasa de aprendizaje después de converger, actualizar la red target más despacio, o cortar cuando Q sin shaping pasa de ~1,2.
3. **Recompensa de compactación para muchas ovejas.** Un círculo de radio f(N) = r_a·N^(2/3) centrado en el centro del rebaño (el umbral que usa Strömbom para pasar de recolectar a arrear). Se gana al entrar una oveja y se descuenta al salir. Hay que escribirlo como shaping basado en potencial, con γ: sin ella, "entrar ahora y salir después" da ganancia neta con descuento y se puede farmear. Tiene un tope natural, porque apretar un rebaño que ya está adentro no suma. Variante continua: penalizar cuánto se pasa cada oveja del círculo.
4. **Criterio de éxito con muchas ovejas:** pasar a "centro en la meta y rebaño compacto", para que no valga un rebaño partido en dos con el centro justo en la meta.
5. **Más adelante:** DQN contra PPO (mismas 9 acciones y también con acción continua), y una ablación del replay buffer y la red target, mejor con el DQN propio del Track A en CartPole.
