# Pastoreo con DQN (Track B)

Entorno Gymnasium del modelo de pastoreo de Strömbom (2014) y entrenamiento de un DQN de Stable-Baselines3 que hace de pastor. Qué se probó y qué dio: [REPORTE_AVANCE.md](REPORTE_AVANCE.md). Conceptos y decisiones de diseño: [GUIA_CONCEPTOS.md](GUIA_CONCEPTOS.md).

## Instalación

```
python -m venv .venv
.venv\Scripts\activate          # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenar

```
python entrenar_sb3.py --help                          # todos los argumentos y escenarios
python entrenar_sb3.py trivial_case                    # un escenario de escenarios.yaml, tal cual
python entrenar_sb3.py rebano_80m --double --recompensa potencial_gcm_pos05_tiempo --seed 1
```

Un escenario ([escenarios.yaml](escenarios.yaml)) fija la tarea: ovejas, distancias iniciales, cerco, decisiones de entrenamiento. Lo que se pase por la terminal pisa al escenario. Opciones más usadas:

| Opción | Qué cambia | Por defecto |
|---|---|---|
| `--recompensa` | la función de recompensa (ver `RECOMPENSAS` en [pastoreo_env.py](pastoreo_env.py)) | `potencial_gcm` |
| `--double` | Double DQN en vez de DQN ([double_dqn.py](double_dqn.py)) | DQN |
| `--pasos` | decisiones de entrenamiento | según el escenario |
| `--seed` | semilla (conviene correr 0, 1 y 2) | 0 |
| `--lr` | tasa de aprendizaje | 0.0005 |
| `--buffer` | tamaño del replay buffer, en transiciones | 100000 |
| `--red` | neuronas de cada capa oculta, separadas por coma: `--red 256,256`, `--red 64,64,64`. Las capas usan ReLU | `128,128` |
| `--observacion` | `normalizada` u `original` (ver abajo) | `normalizada` |
| `--sin-barra` | una línea por evaluación en vez de la barra (para correr varias en paralelo) | con barra |

Otros hiperparámetros del DQN (batch, red target, exploración) están en `HIPERPARAMETROS`, en [entrenar_sb3.py](entrenar_sb3.py). Todavía no tienen opción de terminal.

Cada corrida deja:
- `modelos/<nombre>.zip`, el modelo final, y `modelos/<nombre>_mejor.zip`, el mejor checkpoint. No están en git.
- `resultados/entrenamientos/<nombre>.csv`, la curva de aprendizaje, y `<nombre>.json`, la configuración completa.
- Una fila en [resultados/registro.csv](resultados/registro.csv).

El nombre se arma con los argumentos; [modelos/README.md](modelos/README.md) explica cómo leerlo.

### La observación

La red ve 8 números: la posición relativa pastor→rebaño, rebaño→oveja más lejana y rebaño→meta, la dispersión y la distancia a la oveja más cercana.

Desde el 20/11/2026 esa distancia se divide por r_s = 65 m (`normalizada`). Antes se dividía por 6 m (`original`): con el pastor lejos valía 13–25, mientras las otras entradas rondan entre −1 y 1.

**Un modelo solo sirve con la observación con la que se entrenó.** El JSON de cada corrida la guarda, y `ver_agente.py` y `simulador.py` la usan solos. A las corridas anteriores, que no la guardaban, les asignan `original`.

## Ver un modelo

```
python simulador.py                                   # el mejor checkpoint más reciente, en vivo
python simulador.py --modelo <nombre> --escenario paper_1_oveja
python ver_agente.py <nombre>                         # animación de unos episodios
python ver_agente.py <nombre> --guardar videos/<nombre>.gif
python ver_agente.py <nombre> --strombom              # la heurística de Strömbom en los mismos mapas
```

En el simulador se puede:
- cambiar de escenario y de pastor (DQN, Strömbom o manual con las flechas);
- pausar y avanzar de a un paso;
- elegir la velocidad en pasos por segundo;
- mover al pastor (click izquierdo) o al rebaño (click derecho).

El panel muestra el valor Q de cada acción.

## Línea de base

```
python medir_strombom.py      # Strömbom en cada configuración -> resultados/strombom_baseline.csv
```
