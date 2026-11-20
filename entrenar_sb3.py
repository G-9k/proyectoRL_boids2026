"""
Entrena el DQN de Stable-Baselines3 en PastoreoEnv y lo evalúa cada tanto sobre mapas fijos.

Uso con un escenario con nombre (definidos en escenarios.yaml):
    python entrenar_sb3.py trivial_case
    python entrenar_sb3.py trivial_case --gamma 0.95 --seed 1    # lo que se pase por la terminal pisa al escenario

Uso sin escenario (todo por argumentos):
    python entrenar_sb3.py --dist-inicio-rebano 40 --dist-inicio-pastor 10 --angulo-pastor 0 --cerco --repetir-accion 4 --pasos 30000

Todos los argumentos y escenarios:  python entrenar_sb3.py --help
Muestra una barra de progreso en vivo (ver EvalPeriodica). Ctrl+C corta y guarda el modelo igual.
Cada corrida agrega una fila a resultados/registro.csv (el log de todo lo que se probó).

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
from tqdm import tqdm

import yaml

from double_dqn import DoubleDQN
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

# Valores por defecto de los argumentos. Un escenario (escenarios.yaml) pisa algunos; la terminal pisa todo.
DEFAULTS = dict(
    n_ovejas=1,
    recompensa="potencial_gcm",
    dist_inicio_rebano=None,
    dist_inicio_pastor=None,
    angulo_pastor=None,
    repetir_accion=1,
    cerco=False,
    max_pasos=1000,
    pasos=200_000,
    seed=0,
    gamma=0.99,
    eval_cada=20_000,
    eval_episodios=50,
    double=False,
    lr=HIPERPARAMETROS["learning_rate"],
    nombre=None,
    sin_barra=False,
)

RUTA_ESCENARIOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "escenarios.yaml")
RUTA_REGISTRO = os.path.join("resultados", "registro.csv")
COLUMNAS_REGISTRO = [
    "fecha", "nombre", "escenario", "cambios", "decisiones", "eval_final", "eval_max", "eval_max_decision",
    "qmax_final", "qmax_sin_shaping_final", "minutos", "estado",
]


def nombre_por_defecto(args) -> str:
    """
    n{ovejas}_{recompensa}_reb{dist}_pas{dist}[_ang{grados}]_k{repetir}[_cerco][_g{gamma}][_mp{max_pasos}][_ddqn]_s{semilla}
    Ver modelos/README.md. "orig" = como en el paper (sin el parámetro de currículum).
    gamma y max_pasos solo aparecen si no son los valores por defecto (0.99 y 1000).
    """
    def dist(x):
        return "orig" if x is None else f"{x:g}"

    partes = [
        f"n{args.n_ovejas}",
        args.recompensa,
        f"reb{dist(args.dist_inicio_rebano)}",
        f"pas{dist(args.dist_inicio_pastor)}",
    ]
    if args.angulo_pastor is not None:
        partes.append(f"ang{args.angulo_pastor:g}")
    partes.append(f"k{args.repetir_accion}")
    if args.cerco:
        partes.append("cerco")
    if args.gamma != DEFAULTS["gamma"]:
        partes.append(f"g{args.gamma:g}")
    if args.max_pasos != DEFAULTS["max_pasos"]:
        partes.append(f"mp{args.max_pasos}")
    if args.double:
        partes.append("ddqn")
    if args.lr != DEFAULTS["lr"]:
        partes.append(f"lr{args.lr:g}")
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
    """
    Evalúa cada `cada` decisiones y, si `barra` es True, muestra una barra de progreso en vivo:
      eps     = epsilon actual (probabilidad de acción al azar)
      ret     = recompensa promedio de los últimos 20 episodios de entrenamiento
      exito_t = % de éxito de los últimos 100 episodios de entrenamiento (con exploración)
      Qmax    = el mayor valor Q de la red en el estado actual, SIN la parte que agrega el shaping
                (Q + Phi(s)). Con cualquiera de nuestras recompensas, el retorno real máximo es 1
                (llegar a la meta), así que si Qmax pasa claramente de 1 hay sobreestimación.
                En el CSV y en las líneas [eval] también se guarda el valor crudo (con shaping),
                que depende de lo lejos que esté la oveja y no sirve para comparar.
      eval    = % de éxito de la última evaluación (50 mapas fijos, sin exploración)

    Si se pasa `ruta_mejor`, guarda ahí el modelo cada vez que una evaluación supera a la mejor
    anterior: más % de éxito, o el mismo % en menos pasos. Ante un empate total se queda con el
    primero, porque más adelante la sobreestimación puede crecer y terminar en un derrumbe.
    """

    ACTUALIZAR_BARRA_CADA = 200  # decisiones; actualizar en cada paso haría más lento el entrenamiento

    def __init__(self, env_kwargs, cada, ruta_csv, episodios, barra=True, ruta_mejor=None):
        super().__init__()
        self.env_kwargs, self.cada, self.ruta_csv, self.episodios = env_kwargs, cada, ruta_csv, episodios
        self.barra = barra
        self.ruta_mejor = ruta_mejor
        self.mejor = None          # (éxito, pasos, decisión) de la mejor evaluación hasta ahora
        self.pbar = None
        self.ultima_eval = float("nan")
        self.evals = []            # % de éxito de cada evaluación, para el registro
        self.ultimo_qmax = float("nan")              # crudo, con shaping
        self.ultimo_qmax_sin_shaping = float("nan")
        self.t0 = time.time()

    def _on_training_start(self):
        self.t0 = time.time()
        if self.barra:
            self.pbar = tqdm(total=self.locals["total_timesteps"], unit="dec", dynamic_ncols=True, smoothing=0.05)

    def _on_step(self):
        if self.pbar is not None and self.num_timesteps % self.ACTUALIZAR_BARRA_CADA == 0:
            self.pbar.update(self.num_timesteps - self.pbar.n)
            self.pbar.set_postfix_str(self._resumen_vivo(), refresh=False)

        if self.num_timesteps % self.cada == 0:
            exito, pasos = evaluar(self.model, self.env_kwargs, self.episodios)
            self.ultima_eval = exito
            self.evals.append(exito)
            q_sin, q_crudo = self._qmax()
            linea = (f"  [eval] decisión {self.num_timesteps:>8,} | éxito {exito:5.1f}% | pasos {pasos:6.0f} | "
                     f"eps {self.model.exploration_rate:.2f} | Qmax {q_sin:.2f} (crudo {q_crudo:.2f})")
            if self._es_mejor(exito, pasos):
                self.mejor = (exito, pasos, self.num_timesteps)
                if self.ruta_mejor:
                    self.model.save(self.ruta_mejor)
                    linea += " | * mejor, guardado"
            if self.pbar is not None:
                self.pbar.write(linea)  # escribe arriba de la barra sin romperla
            else:
                fps = self.num_timesteps / (time.time() - self.t0)
                print(f"{linea} | {fps:.0f} dec/s", flush=True)
            with open(self.ruta_csv, "a", newline="") as f:
                csv.writer(f).writerow([self.num_timesteps, exito, pasos, round(time.time() - self.t0, 1),
                                        round(q_crudo, 3), round(q_sin, 3)])
        return True

    def _es_mejor(self, exito: float, pasos: float) -> bool:
        if exito <= 0:
            return False  # un modelo que no resuelve nada no vale la pena guardarlo
        if self.mejor is None or exito > self.mejor[0]:
            return True
        return exito == self.mejor[0] and pasos < self.mejor[1]

    def _qmax(self) -> tuple[float, float]:
        """
        (Qmax sin shaping, Qmax crudo) en el estado actual del entrenamiento.
        Con shaping por potencial, la red aprende Q_crudo(s, a) = Q_sin_shaping(s, a) - Phi(s),
        así que Q_sin_shaping = Q_crudo + Phi(s).
        """
        with torch.no_grad():
            obs = torch.as_tensor(self.locals["new_obs"], device=self.model.device)
            self.ultimo_qmax = float(self.model.q_net(obs).max())
        phi = self.training_env.env_method("potencial_actual")[0]
        self.ultimo_qmax_sin_shaping = self.ultimo_qmax + phi
        return self.ultimo_qmax_sin_shaping, self.ultimo_qmax

    def _resumen_vivo(self) -> str:
        eps_info = list(self.model.ep_info_buffer)[-20:]
        ret = np.mean([e["r"] for e in eps_info]) if eps_info else float("nan")
        exitos = self.model.ep_success_buffer
        exito_t = 100.0 * np.mean(exitos) if len(exitos) else float("nan")
        return (f"eps={self.model.exploration_rate:.2f} ret={ret:.2f} exito_t={exito_t:.0f}% "
                f"Qmax={self._qmax()[0]:.2f} eval={self.ultima_eval:.0f}%")

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.update(self.num_timesteps - self.pbar.n)
            self.pbar.close()


def cargar_escenarios(ruta=RUTA_ESCENARIOS) -> dict:
    """
    Lee escenarios.yaml y devuelve {nombre: {"descripcion": str, "config": dict}}.
    Avisa con un error claro si un escenario tiene un parámetro que no existe (por ejemplo, un typo).
    """
    with open(ruta, encoding="utf-8") as f:
        crudo = yaml.safe_load(f) or {}
    escenarios = {}
    for nombre, valores in crudo.items():
        valores = dict(valores or {})
        descripcion = str(valores.pop("descripcion", ""))
        desconocidos = set(valores) - set(DEFAULTS)
        if desconocidos:
            raise ValueError(
                f"escenarios.yaml, escenario '{nombre}': parámetro(s) desconocido(s) {sorted(desconocidos)}. "
                f"Los válidos son: {sorted(DEFAULTS)}"
            )
        escenarios[nombre] = {"descripcion": descripcion, "config": valores}
    return escenarios


def kwargs_entorno(params: dict) -> dict:
    """Parámetros de PastoreoEnv a partir de los argumentos del script (o de DEFAULTS + un escenario)."""
    return dict(
        n_ovejas=params["n_ovejas"],
        recompensa=params["recompensa"],
        gamma=params["gamma"],  # el mismo gamma que el agente, para que el shaping sea correcto
        max_pasos=params["max_pasos"],
        cerco=params["cerco"],
        dist_inicio_pastor=params["dist_inicio_pastor"],
        dist_inicio_rebano=params["dist_inicio_rebano"],
        repetir_accion=params["repetir_accion"],
        angulo_pastor=params["angulo_pastor"],
    )


def leer_argumentos():
    """
    Combina, en este orden (cada uno pisa al anterior): DEFAULTS, el escenario elegido y lo que se
    pasó por la terminal. Devuelve (args, nombre del escenario o None, dict de cambios respecto del escenario).
    """
    ESCENARIOS = cargar_escenarios()
    lista = "\n".join(f"  {n:<16} {e['descripcion']}" for n, e in ESCENARIOS.items())
    p = argparse.ArgumentParser(
        description="Entrena el DQN de SB3 en el entorno de pastoreo.",
        epilog=f"escenarios disponibles (se editan en escenarios.yaml):\n{lista}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    S = argparse.SUPPRESS  # así solo aparecen en args los argumentos que se pasaron de verdad
    d = DEFAULTS
    p.add_argument("escenario", nargs="?", default=None, choices=list(ESCENARIOS),
                   help="escenario con nombre de escenarios.yaml (opcional)")
    p.add_argument("--n-ovejas", type=int, default=S, help=f"cantidad de ovejas (default: {d['n_ovejas']})")
    p.add_argument("--recompensa", choices=list(RECOMPENSAS), default=S,
                   help=f"función de recompensa (default: {d['recompensa']})")
    p.add_argument("--dist-inicio-rebano", type=float, default=S,
                   help="metros entre el centro del rebaño y la meta al arrancar (default: como el paper, ~150 m)")
    p.add_argument("--dist-inicio-pastor", type=float, default=S,
                   help="el pastor arranca entre 5 m y esta distancia del borde del rebaño (default: como el paper)")
    p.add_argument("--angulo-pastor", type=float, default=S,
                   help="grados de desvío del pastor respecto de 'detrás del rebaño'; 0 = alineado "
                        "(requiere --dist-inicio-pastor; default: cualquier dirección)")
    p.add_argument("--repetir-accion", type=int, default=S,
                   help=f"pasos de simulación por decisión (default: {d['repetir_accion']})")
    p.add_argument("--cerco", action="store_true", default=S, help="mundo con bordes en [-30, 180]")
    p.add_argument("--max-pasos", type=int, default=S,
                   help=f"pasos de simulación por episodio (default: {d['max_pasos']})")
    p.add_argument("--pasos", type=int, default=S, help=f"decisiones de entrenamiento en total (default: {d['pasos']})")
    p.add_argument("--seed", type=int, default=S, help=f"semilla (default: {d['seed']})")
    p.add_argument("--gamma", type=float, default=S, help=f"descuento, por decisión (default: {d['gamma']})")
    p.add_argument("--eval-cada", type=int, default=S, help=f"evaluar cada N decisiones (default: {d['eval_cada']})")
    p.add_argument("--eval-episodios", type=int, default=S, help=f"mapas por evaluación (default: {d['eval_episodios']})")
    p.add_argument("--lr", type=float, default=S,
                   help=f"tasa de aprendizaje (default: {d['lr']:g})")
    p.add_argument("--double", action="store_true", default=S,
                   help="usar Double DQN en vez de DQN (ver double_dqn.py)")
    p.add_argument("--nombre", default=S, help="nombre de la corrida (default: se arma con los argumentos)")
    p.add_argument("--sin-barra", action="store_true", default=S,
                   help="sin barra de progreso, solo una línea por evaluación (para correr varias en paralelo)")

    terminal = vars(p.parse_args())
    escenario = terminal.pop("escenario")
    del_escenario = ESCENARIOS[escenario]["config"] if escenario else {}
    config = {**DEFAULTS, **del_escenario, **terminal}
    # "cambios": lo que difiere del escenario (o de DEFAULTS si no hay escenario), para el registro
    base = {**DEFAULTS, **del_escenario}
    cambios = {k: v for k, v in terminal.items() if base.get(k) != v and k not in ("sin_barra",)}
    return argparse.Namespace(**config), escenario, cambios


def agregar_al_registro(fila: dict):
    nuevo = not os.path.exists(RUTA_REGISTRO)
    with open(RUTA_REGISTRO, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_REGISTRO)
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def main():
    args, escenario, cambios = leer_argumentos()

    torch.set_num_threads(1)  # redes chicas: un hilo es más rápido (ver PLAN_DQN.md, sección 6)

    env_kwargs = kwargs_entorno(vars(args))
    nombre = args.nombre or nombre_por_defecto(args)
    carpeta = os.path.join("resultados", "entrenamientos")
    os.makedirs(carpeta, exist_ok=True)
    # Si ya hay una corrida con ese nombre (por ejemplo, la misma config con más decisiones), no se pisa:
    # se agrega _v2, _v3, ... El registro guarda qué cambió en cada una.
    if not args.nombre and os.path.exists(os.path.join(carpeta, f"{nombre}.csv")):
        base, n = nombre, 2
        while os.path.exists(os.path.join(carpeta, f"{base}_v{n}.csv")):
            n += 1
        nombre = f"{base}_v{n}"
        print(f"Ya existía una corrida '{base}'; esta se guarda como '{nombre}'.")
    ruta_csv = os.path.join(carpeta, f"{nombre}.csv")
    with open(ruta_csv, "w", newline="") as f:
        csv.writer(f).writerow(["paso", "exito_pct", "pasos_media", "segundos", "qmax", "qmax_sin_shaping"])

    # Configuración completa de la corrida, para poder reproducirla (el nombre no lo dice todo)
    fecha = datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(carpeta, f"{nombre}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "nombre": nombre,
            "fecha": fecha,
            "escenario": escenario,
            "cambios_sobre_escenario": cambios,
            "argumentos": vars(args),
            "entorno": env_kwargs,
            "hiperparametros_dqn": {**HIPERPARAMETROS, "learning_rate": args.lr},
        }, f, indent=2, ensure_ascii=False)

    algoritmo = DoubleDQN if args.double else DQN
    model = algoritmo(
        "MlpPolicy",
        PastoreoEnv(**env_kwargs),
        gamma=args.gamma,
        seed=args.seed,
        verbose=0,
        **{**HIPERPARAMETROS, "learning_rate": args.lr},
    )

    desde = f"escenario '{escenario}'" + (f" con cambios {cambios}" if cambios else "") if escenario else "argumentos"
    print(f"[{nombre}] ({desde}) entrenando {args.pasos:,} decisiones ...", flush=True)
    t0 = time.time()
    os.makedirs("modelos", exist_ok=True)
    callback = EvalPeriodica(env_kwargs, args.eval_cada, ruta_csv, args.eval_episodios, barra=not args.sin_barra,
                             ruta_mejor=os.path.join("modelos", f"{nombre}_mejor"))
    interrumpido = False
    try:
        model.learn(args.pasos, callback=callback)
    except KeyboardInterrupt:
        # Ctrl+C: se guarda igual lo que se entrenó hasta ahora
        interrumpido = True
        if callback.pbar is not None:
            callback.pbar.close()
        print(f"\n[{nombre}] interrumpido en la decisión {model.num_timesteps:,}. Guardando el modelo...")

    model.save(os.path.join("modelos", nombre))
    estado = "interrumpido" if interrumpido else "listo"
    mejor = callback.mejor

    agregar_al_registro({
        "fecha": fecha,
        "nombre": nombre,
        "escenario": escenario or "",
        "cambios": " ".join(f"{k}={v}" for k, v in cambios.items()),
        "decisiones": model.num_timesteps,
        "eval_final": callback.evals[-1] if callback.evals else "",
        "eval_max": max(callback.evals) if callback.evals else "",
        "eval_max_decision": mejor[2] if mejor else "",
        "qmax_final": round(callback.ultimo_qmax, 3),
        "qmax_sin_shaping_final": round(callback.ultimo_qmax_sin_shaping, 3),
        "minutos": round((time.time() - t0) / 60, 1),
        "estado": estado,
    })
    print(f"[{nombre}] {estado}. Curva en {ruta_csv}, modelo en modelos/{nombre}.zip, "
          f"fila agregada a {RUTA_REGISTRO}")
    if mejor:
        print(f"[{nombre}] mejor modelo: {mejor[0]:.0f}% en {mejor[1]:.0f} pasos, decisión {mejor[2]:,} "
              f"-> modelos/{nombre}_mejor.zip")


if __name__ == "__main__":
    main()
