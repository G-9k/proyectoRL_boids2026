"""
Entrena el DQN de Stable-Baselines3 en PastoreoEnv y lo evalúa cada tanto sobre mapas fijos.

Ejemplo (nivel más fácil del currículum):
    python entrenar_sb3.py --n-ovejas 1 --dist-inicio-pastor 20 --pasos 200000 --seed 0

Cada evaluación usa los mismos 50 mapas (semillas fijas), sin exploración, y mide el % de éxito y los
pasos promedio de los episodios exitosos. No se compara la recompensa: cada variante tiene otra escala.
Resultados en resultados/entrenamientos/<nombre>.csv (curva) y <nombre>.json (configuración),
modelo en modelos/<nombre>.zip. El nombre se arma solo a partir de los argumentos: ver modelos/README.md.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from pastoreo_env import PastoreoEnv, RECOMPENSAS

SEED_EVAL = 20_000  # mapas de evaluación: SEED_EVAL + episodio (distintos de los de medir_strombom.py)

HIPERPARAMETROS = dict(
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=128,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[128, 128]),
)


def nombre_por_defecto(args) -> str:
    """
    n{ovejas}_{recompensa}_reb{dist}_pas{dist}_k{repetir}[_cerco]_s{semilla}
    Ver modelos/README.md. "orig" = como en el paper (sin el parámetro de currículum).
    """
    def dist(x):
        return "orig" if x is None else f"{x:g}"

    partes = [
        f"n{args.n_ovejas}",
        args.recompensa,
        f"reb{dist(args.dist_inicio_rebano)}",
        f"pas{dist(args.dist_inicio_pastor)}",
        f"k{args.repetir_accion}",
    ]
    if args.cerco:
        partes.append("cerco")
    partes.append(f"s{args.seed}")
    return "_".join(partes)


def evaluar(model, env_kwargs, episodios=50):
    env = PastoreoEnv(**env_kwargs)
    exitos, pasos = 0, []
    for ep in range(episodios):
        obs, _ = env.reset(seed=SEED_EVAL + ep)
        terminado = truncado = False
        while not (terminado or truncado):
            accion, _ = model.predict(obs, deterministic=True)
            obs, _, terminado, truncado, info = env.step(int(accion))
        if terminado:
            exitos += 1
            pasos.append(info["pasos"])
    return 100.0 * exitos / episodios, (float(np.mean(pasos)) if pasos else float("nan"))


class EvalPeriodica(BaseCallback):
    def __init__(self, env_kwargs, cada, ruta_csv, episodios):
        super().__init__()
        self.env_kwargs, self.cada, self.ruta_csv, self.episodios = env_kwargs, cada, ruta_csv, episodios
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps % self.cada == 0:
            exito, pasos = evaluar(self.model, self.env_kwargs, self.episodios)
            fps = self.num_timesteps / (time.time() - self.t0)
            print(f"  paso {self.num_timesteps:>8,} | éxito {exito:5.1f}% | pasos {pasos:6.0f} | "
                  f"eps {self.model.exploration_rate:.2f} | {fps:.0f} pasos/s", flush=True)
            with open(self.ruta_csv, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, exito, pasos, round(time.time() - self.t0, 1)])
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-ovejas", type=int, default=1)
    p.add_argument("--recompensa", default="potencial_gcm", choices=list(RECOMPENSAS))
    p.add_argument("--dist-inicio-pastor", type=float, default=None)
    p.add_argument("--dist-inicio-rebano", type=float, default=None)
    p.add_argument("--repetir-accion", type=int, default=1)
    p.add_argument("--cerco", action="store_true")
    p.add_argument("--max-pasos", type=int, default=1000)
    p.add_argument("--pasos", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval-cada", type=int, default=20_000)
    p.add_argument("--eval-episodios", type=int, default=50)
    p.add_argument("--nombre", default=None)
    args = p.parse_args()

    torch.set_num_threads(1)  # redes chicas: un hilo es más rápido (ver PLAN_DQN.md, sección 6)

    env_kwargs = dict(
        n_ovejas=args.n_ovejas,
        recompensa=args.recompensa,
        gamma=args.gamma,  # el mismo gamma que el agente, para que el shaping sea correcto
        max_pasos=args.max_pasos,
        cerco=args.cerco,
        dist_inicio_pastor=args.dist_inicio_pastor,
        dist_inicio_rebano=args.dist_inicio_rebano,
        repetir_accion=args.repetir_accion,
    )
    nombre = args.nombre or nombre_por_defecto(args)
    carpeta = os.path.join("resultados", "entrenamientos")
    os.makedirs(carpeta, exist_ok=True)
    ruta_csv = os.path.join(carpeta, f"{nombre}.csv")
    with open(ruta_csv, "w", newline="") as f:
        csv.writer(f).writerow(["paso", "exito_pct", "pasos_media", "segundos"])

    # Configuración completa de la corrida, para poder reproducirla (el nombre no lo dice todo)
    with open(os.path.join(carpeta, f"{nombre}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "nombre": nombre,
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "argumentos": vars(args),
            "entorno": env_kwargs,
            "hiperparametros_dqn": HIPERPARAMETROS,
        }, f, indent=2, ensure_ascii=False)

    model = DQN(
        "MlpPolicy",
        PastoreoEnv(**env_kwargs),
        gamma=args.gamma,
        seed=args.seed,
        verbose=0,
        **HIPERPARAMETROS,
    )

    print(f"[{nombre}] entrenando {args.pasos:,} pasos ...", flush=True)
    model.learn(args.pasos, callback=EvalPeriodica(env_kwargs, args.eval_cada, ruta_csv, args.eval_episodios))
    os.makedirs("modelos", exist_ok=True)
    model.save(os.path.join("modelos", nombre))
    print(f"[{nombre}] listo. Curva en {ruta_csv}, modelo en modelos/{nombre}.zip")


if __name__ == "__main__":
    main()
