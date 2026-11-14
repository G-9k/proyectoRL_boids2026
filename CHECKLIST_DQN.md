# Checklist del proyecto DQN

Seguimiento de las tareas de [PLAN_DQN.md](PLAN_DQN.md). Cada tarea está pensada para que se pueda marcar en unas horas de trabajo como mucho.

**Cómo se usa:**
- Al terminar una tarea, cambiar `[ ]` por `[x]` y agregar quién y cuándo. Ejemplo: `- [x] Replay buffer implementado — Ana, 20/11`.
- Al empezar algo nuevo, actualizar la columna **"Ahora está en"** de la tabla de asignación. Así se ve de un vistazo por dónde va cada uno.
- Un **hito** se marca recién cuando están todas las tareas que lo componen.

---

## Asignación

| Rol | Nombre | Track | Ahora está en |
|-----|--------|-------|---------------|
| Persona 1 | | A: replay buffer + loop de entrenamiento | |
| Persona 2 | | A: red Q + actualización | |
| Persona 3 | | A: entornos de prueba + evaluación | |
| Persona 4 | | B: entorno Gymnasium | |
| Persona 5 | | B: recompensa + líneas base | |

---

## Tablero de hitos

- [ ] **Hito 0:** todos tienen el entorno instalado.
- [ ] **Hito 1:** todos terminaron la fase 1 y hubo reunión de puesta en común.
- [ ] **A1:** nuestro DQN pasa los 5 entornos de prueba.
- [ ] **A2:** nuestro DQN resuelve `CartPole-v1` (retorno promedio ≥ 475 en al menos 2 de 3 semillas).
- [ ] **B1:** el entorno de pastoreo pasa `check_env` y el pastor de Strömbom está medido.
- [ ] **B2:** el DQN de SB3 aprende el nivel más fácil del pastoreo.
- [ ] **Intercambio:** cada track le explicó su parte al otro.
- [ ] **Final:** nuestro DQN comparado contra SB3 y Strömbom en todos los niveles del currículum.

---

## Fase 0: setup (todos)

| Tarea | P1 | P2 | P3 | P4 | P5 |
|-------|----|----|----|----|----|
| `.venv` creado e instalado con `requirements.txt` | [ ] | [ ] | [ ] | [ ] | [ ] |
| `python -c "import torch, gymnasium, stable_baselines3"` corre sin errores | [ ] | [ ] | [ ] | [ ] | [ ] |
| Leyó [PLAN_DQN.md](PLAN_DQN.md) | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## Fase 1: base común (todos, cada uno por su cuenta)

| Tarea | P1 | P2 | P3 | P4 | P5 |
|-------|----|----|----|----|----|
| 1.1 Leyó el post de LessWrong, secciones 3 a 5 | [ ] | [ ] | [ ] | [ ] | [ ] |
| 1.2a Programó Q-learning tabular (`FrozenLake-v1` o `CliffWalking-v1`) | [ ] | [ ] | [ ] | [ ] | [ ] |
| 1.2b Su agente tabular aprende: la curva de retorno sube y la política llega a la meta | [ ] | [ ] | [ ] | [ ] | [ ] |
| 1.3 PyTorch, ejercicio A: la red aproxima `sin(x)` | [ ] | [ ] | [ ] | [ ] | [ ] |
| 1.4 PyTorch, ejercicio B: la red recupera `Q(s, a)` usando `gather` | [ ] | [ ] | [ ] | [ ] | [ ] |
| *(Opcional)* Leyó Sutton & Barto, sección 11.3 | [ ] | [ ] | [ ] | [ ] | [ ] |

---

## Fase 2, Track A: DQN desde cero (Personas 1, 2 y 3)

### Entre los tres
- [ ] Acordamos la interfaz (`ReplayBuffer`, `QNetwork`, `actualizar`) y la dejamos escrita.
- [ ] Acordamos la estructura de archivos y carpetas del código.

### Persona 1: replay buffer, exploración y loop de entrenamiento
- [ ] `ReplayBuffer.add(...)` guarda transiciones y no supera la capacidad máxima.
- [ ] `ReplayBuffer.sample(B)` devuelve tensores con las formas y tipos acordados.
- [ ] Test chico del buffer: llenarlo de más, samplear y chequear formas y tipos.
- [ ] ε-greedy con ε que baja linealmente de 1 a un valor final.
- [ ] Loop de entrenamiento: interactúa con el entorno, entrena cada *k* pasos y sincroniza la red target cada *m* pasos.
- [ ] El loop guarda `terminated` (no `truncated`) como "terminado" en el buffer.
- [ ] Registra el retorno de cada episodio, para poder graficarlo.

### Persona 2: red Q y actualización
- [ ] `QNetwork` (MLP): toma (B, obs_dim) y devuelve (B, n_acciones).
- [ ] `actualizar(...)` selecciona Q(s, a) con `gather`.
- [ ] El objetivo `r + γ · (1 − terminado) · max Q_target(s')` se calcula dentro de `torch.no_grad()`.
- [ ] Pérdida (MSE o Huber) + paso del optimizador.
- [ ] Copia de pesos de la red Q a la red target.
- [ ] Test: con un lote fijo, la pérdida baja al llamar `actualizar` muchas veces seguidas.

### Persona 3: entornos de prueba y evaluación
- [ ] Prueba 1: una acción, observación constante, recompensa +1 (Q tiene que dar 1).
- [ ] Prueba 2: observación ±1, recompensa igual a la observación.
- [ ] Prueba 3: dos pasos, recompensa solo al final (Q del primer paso tiene que dar γ).
- [ ] Prueba 4: dos acciones, recompensa según la acción.
- [ ] Prueba 5: la acción correcta depende de la observación.
- [ ] Script de evaluación: N episodios sin exploración, devuelve el retorno promedio.
- [ ] Gráfico de curvas de aprendizaje con varias semillas (media ± desvío).
- [ ] Script que corre varias semillas en paralelo, con `torch.set_num_threads(1)`.

### Code review cruzado
- [ ] El código de la Persona 1 lo revisó: ______
- [ ] El código de la Persona 2 lo revisó: ______
- [ ] El código de la Persona 3 lo revisó: ______

### Camino al hito A1: el DQN completo pasa cada prueba
- [ ] Prueba 1
- [ ] Prueba 2
- [ ] Prueba 3
- [ ] Prueba 4
- [ ] Prueba 5

### Camino al hito A2: CartPole
- [ ] Primera corrida completa en `CartPole-v1` sin errores.
- [ ] Semilla 1 llega a ≥ 475.
- [ ] Semilla 2 llega a ≥ 475.
- [ ] Semilla 3 llega a ≥ 475.
- [ ] *(Después de A2)* Comparamos nuestro código con `dqn.py` de CleanRL y anotamos las diferencias.

---

## Fase 2, Track B: el entorno de pastoreo (Personas 4 y 5)

### Persona 4: entorno Gymnasium
- [ ] Archivo nuevo (por ejemplo `pastoreo_env.py`) con una clase que hereda de `gym.Env` y reutiliza `ManualStrombomEnv`, sin romper el original.
- [ ] `observation_space = Box(shape=(8,))` y `action_space = Discrete(9)`.
- [ ] `N` y `f_N` pasan a ser parámetros (hoy son constantes del módulo).
- [ ] `reset(seed=...)` es reproducible: la misma semilla da el mismo estado inicial.
- [ ] `step(accion)` devuelve `obs, recompensa, terminated, truncated, info`.
- [ ] `terminated` cuando el rebaño llega a la meta, `truncated` al llegar al máximo de pasos.
- [ ] Parámetros de currículum: cantidad de ovejas, dónde arranca el pastor, radio de la meta y máximo de pasos.
- [ ] Pasa `stable_baselines3.common.env_checker.check_env`.
- [ ] Script para mirar con matplotlib cómo juega un agente ya entrenado.

### Persona 5: recompensa y líneas base
- [ ] Recompensa esparsa (solo al llegar a la meta) implementada como variante base.
- [ ] Recompensa con shaping basado en potencial (`F = γ·Φ(s') − Φ(s)`) implementada.
- [ ] Lista de trampas posibles de cada recompensa, discutida con el equipo.
- [ ] Parametrizar `N` también en [Pastoreo_clásico.py](Pastoreo_clásico.py), para poder medir Strömbom con distinta cantidad de ovejas.
- [ ] Strömbom medido con 1 oveja: tasa de éxito y pasos promedio en 100 episodios.
- [ ] Strömbom medido con 3 ovejas.
- [ ] Strömbom medido con 10 ovejas.
- [ ] Strömbom medido con 30 ovejas.
- [ ] DQN de SB3 entrenado en el nivel de 1 oveja.
- [ ] Si SB3 no aprende: anotamos qué cambiamos (recompensa, currículum, máximo de pasos) y el resultado de cada cambio.

---

## Intercambio entre tracks
- [ ] El Track A le explicó DQN al Track B.
- [ ] El Track B le explicó el entorno y la recompensa al Track A.

---

## Fase 3: juntar (todos)

- [ ] Nuestro DQN corre sin errores en el entorno de pastoreo.
- [ ] Discutimos las preguntas de la sección 4 de [PLAN_DQN.md](PLAN_DQN.md).

### Resultados por nivel del currículum

Completar con la tasa de éxito y los pasos promedio. Para los dos DQN, usar al menos 3 semillas.

| Nivel | Responsable | Strömbom | SB3 DQN | Nuestro DQN | Notas |
|-------|-------------|----------|---------|-------------|-------|
| 1 oveja | | | | | |
| 3 ovejas | | | | | |
| 10 ovejas | | | | | |
| 30 ovejas | | | | | |
