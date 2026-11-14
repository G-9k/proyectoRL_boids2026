# Plan: aprender DQN y aplicarlo al pastoreo

**Objetivo:** que el equipo entienda DQN implementándolo desde cero, primero en entornos simples y después en el entorno de pastoreo ([Pastoreo_flechitas.py](Pastoreo_flechitas.py)).

**Criterio para recortar:** lo que más enseña DQN es programarlo. La teoría se limita a lo necesario para escribir el código y entender por qué falla.

---

## 0. Setup (todos, 15 min)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se activa con `.\.venv\Scripts\Activate.ps1` en Windows o `source .venv/bin/activate` en Linux/Mac. En VS Code: `Ctrl+Shift+P` → "Python: Select Interpreter" → `.venv`.

Se instala:
- `numpy` y `matplotlib`: la simulación y la visualización.
- `torch` (versión CPU): la red Q.
- `gymnasium`: la interfaz estándar de entornos, más CartPole, FrozenLake y CliffWalking.
- `stable-baselines3`: un DQN ya hecho y probado, que usamos como referencia (ver sección 5).

**¿Hace falta GPU?** No. Una red de 8 entradas y 9 salidas es tan chica que la GPU no acelera nada; el cuello de botella es la simulación, que corre en CPU igual. Ver los datos medidos en la sección 6.

---

## 1. Base común (todos, cada uno por su cuenta)

Esta fase la hacen todos porque es el núcleo de lo que hay que aprender, y es corta.

| # | Qué | Tiempo aprox. |
|---|-----|---------------|
| 1.1 | Leer el [post de LessWrong](https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained), **secciones 3, 4 y 5**. Saltear lo específico de Atari (convoluciones, apilado de frames, wrappers) y la sección 6. Quien no tenga base de redes, leer también la 1 y la 2. | 1 h |
| 1.2 | **Q-learning tabular desde cero** en `FrozenLake-v1` o `CliffWalking-v1` de Gymnasium. Solo Q-learning, sin comparar con Sarsa. | 2–3 h |
| 1.3 | **Repaso de PyTorch**, ejercicio A: una red que aproxime `y = sin(x)` con datos generados. Si alguien se traba: tutorial ["Learn the Basics"](https://pytorch.org/tutorials/beginner/basics/intro.html). | 1 h |
| 1.4 | **Repaso de PyTorch**, ejercicio B, "la mitad supervisada de DQN" (ver abajo). | 1 h |

**Ejercicio 1.4 en detalle.** Inventen una función Q verdadera, por ejemplo `Q(s, a) = s · w_a`: un vector de pesos al azar `w_a` por cada una de las 9 acciones, con estados `s` de 8 dimensiones. Generen lotes de `(s, a, y)` con acciones al azar y entrenen una red que, dado `s`, devuelva los 9 valores Q. La pérdida usa solo el valor de la acción tomada, que se selecciona con `q.gather(1, a.unsqueeze(1))`. Verifiquen que la red aprende la función.

Este es exactamente el paso de actualización de DQN. Lo único que cambia en DQN es de dónde sale `y`: deja de ser un dato conocido y pasa a ser `r + γ · max_a' Q_target(s', a')`.

**Teoría opcional**, para quien quiera entender *por qué* DQN necesita replay buffer y red target: Sutton & Barto, sección **11.3 ("The Deadly Triad")**, unas pocas páginas. El resto de los capítulos 9, 10 y 16.5 lo salteamos a propósito.

**Hito 1:** reunión corta en la que cada uno muestra su Q-learning tabular y su ejercicio 1.4.

---

## 2. Dos tracks en paralelo

El entorno de pastoreo no depende de tener DQN listo, así que se arma en paralelo. Cuando DQN funcione en CartPole, ya vamos a tener un entorno de pastoreo que sabemos que se puede aprender.

### Track A: DQN desde cero (Personas 1, 2 y 3)

Primero acuerden la **interfaz entre las piezas** (ver más abajo), así cada uno avanza sin esperar al otro.

| Persona | Pieza |
|---------|-------|
| **Persona 1** | **Replay buffer + exploración + loop de entrenamiento.** El buffer guarda transiciones y devuelve lotes al azar. La exploración es ε-greedy, con ε bajando linealmente. El loop interactúa con el entorno, decide cada cuántos pasos entrenar y cada cuántos sincronizar la red target, y distingue `terminated` de `truncated` al guardar transiciones. |
| **Persona 2** | **Red Q + paso de actualización.** La red es un MLP. La actualización hace `gather`, calcula el objetivo con la red target dentro de `torch.no_grad()`, aplica la pérdida (MSE o Huber) y el optimizador, y copia los pesos a la red target. |
| **Persona 3** | **Entornos de prueba + evaluación.** Arma los 5 entornos de prueba descritos abajo como entornos de Gymnasium, el script de evaluación (retorno por episodio sin exploración), los gráficos de curvas de aprendizaje con varias semillas (media ± desvío) y un script que corra varias semillas en paralelo. |

**Interfaz sugerida.** Son solo las firmas; la implementación es de cada uno:

- `ReplayBuffer(capacidad, obs_dim)`
  - `.add(obs, accion, recompensa, obs_siguiente, terminado)`
  - `.sample(batch_size)` devuelve tensores: `obs` (B, obs_dim) float32, `acciones` (B,) int64, `recompensas` (B,) float32, `obs_siguiente` (B, obs_dim) float32, `terminado` (B,) float32.
- `QNetwork(obs_dim, n_acciones)`: `forward(obs)` toma (B, obs_dim) y devuelve (B, n_acciones).
- `actualizar(red_q, red_target, optimizador, lote, gamma)` devuelve la pérdida como `float`.

**Entornos de prueba.** La idea es de Andy Jones, ["Debugging RL, Without the Agonizing Pain"](https://andyljones.com/posts/rl-debugging.html). Cada uno prueba una sola pieza y corre en segundos:

1. Una acción, observación constante, recompensa +1, episodio de 1 paso. La red tiene que aprender Q = 1.
2. Una acción, observación ±1 al azar, recompensa igual a la observación. La red tiene que usar la observación.
3. Una acción, episodio de 2 pasos, recompensa +1 solo al final. Q del primer paso tiene que dar γ; prueba el descuento y el bootstrapping.
4. Dos acciones, observación constante, recompensa +1 solo con una de ellas. Prueba que el argmax elija bien.
5. Dos acciones, observación ±1, y la acción correcta depende de la observación. Combina todo lo anterior.

Si algo falla en un entorno de prueba, saben exactamente qué pieza está rota. En CartPole el mismo bug aparece recién después de horas, como "no aprende".

**Code review cruzado obligatorio.** Cada pieza la revisa alguien que no la escribió. Al final del track, las tres personas tienen que poder explicar el DQN completo, no solo su parte.

**Hitos del Track A:**
- **A1:** el DQN pasa los 5 entornos de prueba.
- **A2:** resuelve `CartPole-v1`: retorno promedio ≥ 475 en 100 episodios de evaluación, en al menos 2 de 3 semillas.
  - Ojo: DQN en CartPole a veces llega a 500 y después colapsa. No es necesariamente un bug; por eso hay que mirar varias semillas.
  - Para tener un punto de partida en los hiperparámetros, ver los de CartPole en `rl-baselines3-zoo`, archivo `hyperparams/dqn.yml`.
  - Recién después de A2, comparar con [`dqn.py` de CleanRL](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py). Si lo miran antes, terminan copiándolo.

### Track B: el entorno de pastoreo (Personas 4 y 5)

| Persona | Pieza |
|---------|-------|
| **Persona 4** | **Pasar el entorno a la interfaz de Gymnasium**, en un archivo nuevo (por ejemplo `pastoreo_env.py`) que reutilice `ManualStrombomEnv` sin romper el original. Detalle abajo. |
| **Persona 5** | **Recompensa + líneas base.** (a) Diseñar la recompensa (ver abajo). (b) Medir el pastor de Strömbom de [Pastoreo_clásico.py](Pastoreo_clásico.py): tasa de éxito y pasos promedio en unos 100 episodios, para cada nivel del currículum. (c) Entrenar el DQN de Stable-Baselines3 en el entorno, empezando por el nivel más fácil. |

**Lo que necesita el entorno (Persona 4):**
- `observation_space = Box(shape=(8,))` y `action_space = Discrete(9)`.
- `reset(seed=...)` devuelve `obs, info`. Ojo: hoy `ManualStrombomEnv` usa un generador aleatorio sin semilla, así que hay que pasarle la semilla para que los experimentos sean reproducibles.
- `step(accion)` devuelve `obs, recompensa, terminated, truncated, info`:
  - `terminated` cuando el rebaño llega a la meta.
  - `truncated` al llegar a un máximo de pasos. El mundo no tiene bordes, así que sin esto un episodio puede no terminar nunca.
- Parámetros para el currículum: cantidad de ovejas, dónde arranca el pastor, radio de la meta y máximo de pasos. Ojo: hoy `N` y `f_N` son constantes del módulo, así que hay que convertirlas en parámetros.
- Verificar con `stable_baselines3.common.env_checker.check_env`.

**Diseño de la recompensa (Persona 5):**
- **Por qué no alcanza con recompensar al final.** Al principio el centro del rebaño está a unos 160 m de la meta y las ovejas avanzan como mucho 1 m por paso, así que un episodio exitoso dura cientos o miles de pasos. Con γ = 0.99 el agente "ve" unos 100 pasos hacia adelante; una recompensa solo al final casi no le llega.
- **Reward shaping basado en potencial** (Ng, Harada y Russell, 1999). Se suma `F = γ·Φ(s') − Φ(s)` a la recompensa, por ejemplo con `Φ = −distancia(centro del rebaño, meta)`. Esta forma garantiza que no cambie cuál es la política óptima.
- Pensar en qué otras cosas podría medir Φ, como la dispersión del rebaño, y qué trampas habilita cada diseño (ver las preguntas de la sección 4).

**Currículum sugerido:** 1 oveja con el pastor cerca → 3 ovejas → 10 → 30 (como en el paper).

**Hitos del Track B:**
- **B1:** `check_env` pasa y el pastor de Strömbom ya está medido.
- **B2:** el DQN de SB3 aprende el nivel más fácil. Si no aprende, el problema está en el entorno o en la recompensa, y hay que iterar ahí. Esto es justamente lo que queremos descubrir *antes* de usar nuestro DQN.

### Al terminar la fase 2: intercambio

El Track A le explica DQN al Track B, y el Track B le explica el entorno y la recompensa al Track A. Así todos terminan entendiendo todo.

---

## 3. Juntar (todos)

1. Nuestro DQN en el nivel más fácil del pastoreo.
2. Escalar por el currículum, con experimentos repartidos: cada persona toma un nivel o una variante de recompensa.
3. Comparar siempre tres cosas, con varias semillas:
   - nuestro DQN,
   - el DQN de SB3,
   - el pastor de Strömbom.

---

## 4. Preguntas para discutir en equipo

- **¿El estado es Markov?** Las 8 variables de `get_dqn_state()` contienen justo lo que usa el pastor de Strömbom en modo global: centro del rebaño, oveja más lejana, meta, dispersión y oveja más cercana. O sea que con ese estado se puede representar una política tan buena como la heurística. ¿Qué información queda afuera? Por ejemplo, hacia dónde se mueven las ovejas, o dónde están las que no son la más lejana.
- **Trampas de la recompensa:** si solo premiamos que el centro del rebaño se acerque a la meta, ¿qué pasa si el rebaño se parte en dos?
- Con 9 direcciones discretas, ¿alcanza para imitar al pastor de Strömbom, que se mueve en cualquier dirección?

---

## 5. ¿Para qué usamos Stable-Baselines3?

Es una librería con implementaciones de algoritmos de RL ya probadas, entre ellas DQN. **No la usamos para aprender**, sino como herramienta de control:

- **En el Track B**, para saber si el entorno y la recompensa se pueden aprender, sin depender de que nuestro DQN funcione.
- **En la fase 3**, como referencia: si SB3 resuelve un nivel y nuestro DQN no, el bug está en nuestro código.

Separa las dos fuentes de error que, si no, se mezclan: "nuestro DQN tiene un bug" y "el problema es demasiado difícil tal como lo planteamos".

---

## 6. Datos medidos y consejos de rendimiento

Medido en una laptop con Ryzen 7 7730U, sin dibujar:

- Entorno de pastoreo con 30 ovejas: **~4.400 pasos/s**, con acciones al azar.
- DQN de SB3 en CartPole: **~2.000 pasos/s con 1 hilo** de PyTorch, contra ~1.700 con 8 hilos.

Consejos:
- **Usar `torch.set_num_threads(1)`** en cada corrida. Con redes chicas, repartir entre hilos cuesta más de lo que ahorra.
- **Paralelizar corriendo varias semillas o configuraciones a la vez**, una por núcleo. Esa es la forma real de ganar tiempo.
- **No dibujar durante el entrenamiento.** Matplotlib es solo para mirar una política ya entrenada.

---

## 7. Errores típicos

- **Confundir `terminated` con `truncated`.** Cuando el episodio se corta por límite de pasos (`truncated`), igual hay que hacer bootstrapping desde `s'`. Solo `terminated` anula el término `γ · max Q`.
- **Propagar el gradiente por el objetivo.** El objetivo se calcula con la red target, dentro de `torch.no_grad()`.
- **Tipos y formas en `gather`:** las acciones tienen que ser `int64` y con forma `(B, 1)`.
- **Que ε nunca baje**, o que baje tan rápido que no explora.
- **Juzgar con una sola semilla.** En RL la varianza entre semillas es enorme.

---

## 8. Recursos

- [Deep Q-Networks Explained](https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained): Jay Bailey, LessWrong, 2022.
- [Sutton & Barto, 2.ª edición](http://incompleteideas.net/book/the-book-2nd.html): cap. 6 (ya leído) y sección 11.3 (opcional).
- [Mnih et al. 2015, *Nature*](https://www.nature.com/articles/nature14236): mirar solo el Algoritmo 1, como referencia mientras se programa.
- [Debugging RL, Without the Agonizing Pain](https://andyljones.com/posts/rl-debugging.html): Andy Jones, sobre los entornos de prueba.
- [PyTorch: Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html)
- [Gymnasium](https://gymnasium.farama.org): ver la guía para crear un entorno propio.
- [Stable-Baselines3: DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [CleanRL `dqn.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py): solo después del hito A2.
- Ng, Harada y Russell (1999), *Policy invariance under reward transformations*: el paper del reward shaping basado en potencial.
