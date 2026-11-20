# modelos/

Acá [entrenar_sb3.py](../entrenar_sb3.py) guarda los DQN de Stable-Baselines3 ya entrenados. Cada corrida deja dos archivos:

- `<nombre>.zip`: el modelo al final del entrenamiento.
- `<nombre>_mejor.zip`: el **mejor checkpoint**, es decir, el modelo de la evaluación con más % de éxito (si empata, el de menos pasos; si empata en todo, el primero). Hace falta porque DQN a veces aprende la tarea y después la olvida (en `angulo_90` pasó de 100% a 0%), así que el modelo final puede ser mucho peor que el mejor. La decisión en la que se guardó está en la columna `eval_max_decision` de [resultados/registro.csv](../resultados/registro.csv). Solo existe en corridas desde el 20/11/2026.

Para verlo en acción: `python ver_agente.py <nombre>` (ver [ver_agente.py](../ver_agente.py)).

**No se suben a git**: son binarios y se regeneran corriendo de nuevo el entrenamiento. Si alguien necesita un modelo puntual, se lo pasa aparte. La curva de aprendizaje y la configuración completa de cada modelo sí están en git, en [resultados/entrenamientos/](../resultados/entrenamientos/), con el mismo nombre.

## Cómo leer el nombre

```
n1_potencial_gcm_reb40_pas20_k4_cerco_s0
│  │             │     │     │  │     └─ semilla del entrenamiento
│  │             │     │     │  └─────── con cerco (si no aparece, mundo sin bordes)
│  │             │     │     └────────── repetir_accion: pasos de simulación por decisión
│  │             │     └──────────────── dist_inicio_pastor: el pastor arranca a 5–20 m del borde del rebaño
│  │             └────────────────────── dist_inicio_rebano: el centro del rebaño arranca a 40 m de la meta
│  └──────────────────────────────────── recompensa (ver RECOMPENSAS en pastoreo_env.py)
└─────────────────────────────────────── cantidad de ovejas
```

| Parte | Parámetro de `PastoreoEnv` | Valores |
|---|---|---|
| `n{N}` | `n_ovejas` | 1, 3, 10, 30… |
| `{recompensa}` | `recompensa` | `esparsa`, `potencial_gcm`, … |
| `reb{d}` | `dist_inicio_rebano` | metros entre el centro del rebaño y la meta. `reborig` = como en el paper (cuarto superior derecho, ~150 m) |
| `pas{d}` | `dist_inicio_pastor` | metros máximos entre el pastor y el borde del rebaño. `pasorig` = como en el paper (cuarto inferior izquierdo) |
| `ang{a}` (opcional) | `angulo_pastor` | el pastor arranca detrás del rebaño respecto de la meta, desviado como mucho ±a grados. `ang0` = perfectamente alineado. Si no aparece, el pastor arranca en cualquier dirección alrededor del rebaño |
| `k{k}` | `repetir_accion` | 1 = el agente decide en cada paso; 4 = cada acción dura 4 pasos |
| `cerco` | `cerco=True` | ovejas y pastor no pueden salir de [-30, 180]² |
| `g{γ}` (opcional) | `--gamma` | solo aparece si γ no es 0.99, por ejemplo `g0.95` |
| `mp{n}` (opcional) | `--max-pasos` | solo aparece si no es 1000, por ejemplo `mp1500` |
| `ddqn` (opcional) | `--double` | entrenado con Double DQN ([double_dqn.py](../double_dqn.py)) en vez del DQN común |
| `lr{x}` (opcional) | `--lr` | solo aparece si la tasa de aprendizaje no es 0.0005, por ejemplo `lr0.0001` |
| `v{n}` (al final) | — | ya existía una corrida con el mismo nombre (por ejemplo, la misma configuración con más decisiones); ver `resultados/registro.csv` para saber qué cambió |
| `s{s}` | semilla | normalmente 0, 1 y 2 |

El nombre **no** incluye todo. Quedan afuera, por ejemplo, la cantidad de pasos de entrenamiento, `max_pasos`, gamma y los hiperparámetros del DQN. Esos datos están en `resultados/entrenamientos/<nombre>.json`.

Si al entrenar se pasa `--nombre`, se usa ese nombre en vez del automático. Conviene evitarlo para no romper el esquema.

## Cómo cargar un modelo

```python
import json
from stable_baselines3 import DQN
from pastoreo_env import PastoreoEnv

nombre = "n1_potencial_gcm_reb20_pas20_k4_cerco_s0"
config = json.load(open(f"resultados/entrenamientos/{nombre}.json", encoding="utf-8"))

env = PastoreoEnv(**config["entorno"])   # el mismo entorno con el que se entrenó
model = DQN.load(f"modelos/{nombre}_mejor")   # o f"modelos/{nombre}" para el modelo final

obs, _ = env.reset(seed=0)
accion, _ = model.predict(obs, deterministic=True)
```
