# modelos/

Acá [entrenar_sb3.py](../entrenar_sb3.py) guarda los DQN de Stable-Baselines3 ya entrenados, uno por corrida (`<nombre>.zip`).

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
| `k{k}` | `repetir_accion` | 1 = el agente decide en cada paso; 4 = cada acción dura 4 pasos |
| `cerco` | `cerco=True` | ovejas y pastor no pueden salir de [-30, 180]² |
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
model = DQN.load(f"modelos/{nombre}")

obs, _ = env.reset(seed=0)
accion, _ = model.predict(obs, deterministic=True)
```
