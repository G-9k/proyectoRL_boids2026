"""
Animación de un agente entrenado, con el mismo dibujo que Pastoreo_flechitas.py.

Sirve para cualquier corrida de entrenar_sb3.py: lee el entorno exacto con el que se entrenó de
resultados/entrenamientos/<nombre>.json, así que funciona igual con 1 o 30 ovejas, con o sin cerco,
con cualquier recompensa o ángulo.

    python ver_agente.py <nombre>                      # el mejor checkpoint (modelos/<nombre>_mejor.zip)
    python ver_agente.py <nombre> --final              # el modelo del final del entrenamiento
    python ver_agente.py <nombre> --episodios 5        # varios episodios seguidos
    python ver_agente.py <nombre> --semilla 20003      # otro mapa (20000-20049 son los de la evaluación)
    python ver_agente.py <nombre> --strombom           # el pastor de Strömbom en los mismos mapas, para comparar
    python ver_agente.py <nombre> --guardar videos/<nombre>.gif  # guarda un GIF en vez de abrir la ventana
                                                                 # (videos/ está en .gitignore)
    python ver_agente.py <nombre> --campo-completo     # sin zoom: el mundo entero de -30 a 180 m

Primero se simulan los episodios completos y después se reproducen, así que la animación muestra
cada paso de simulación aunque el agente decida cada k pasos (repetir_accion). El zoom se ajusta
a lo que recorren el pastor y las ovejas en cada episodio.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from stable_baselines3 import DQN

from Pastoreo_flechitas import L, TARGET, WORLD_MAX, WORLD_MIN, r_a, r_s
from pastoreo_env import PastoreoEnv
from strombom_politica import StrombomDiscreto

NOMBRES_ACCIONES = ["quieto", "N", "S", "O", "E", "NO", "NE", "SO", "SE"]
PAUSA_FINAL = 30  # cuadros que se queda quieto el final de cada episodio


# =====================================================================
# 1. Simular: grabar un cuadro por paso de simulación
# =====================================================================
def cargar_corrida(nombre: str, final: bool):
    ruta_json = os.path.join("resultados", "entrenamientos", f"{nombre}.json")
    if not os.path.exists(ruta_json):
        raise SystemExit(f"No existe {ruta_json}. ¿Está bien escrito el nombre de la corrida?")
    config = json.load(open(ruta_json, encoding="utf-8"))
    # Las corridas anteriores al 20/11/2026 no guardaban "observacion": se entrenaron con la original
    config["entorno"].setdefault("observacion", "original")

    ruta_mejor = os.path.join("modelos", f"{nombre}_mejor.zip")
    ruta_final = os.path.join("modelos", f"{nombre}.zip")
    if not final and os.path.exists(ruta_mejor):
        ruta, cual = ruta_mejor, "mejor checkpoint"
    else:
        if not final:
            print(f"Esta corrida no tiene mejor checkpoint (es anterior a que existiera); uso el modelo final.")
        ruta, cual = ruta_final, "modelo final"
    if not os.path.exists(ruta):
        raise SystemExit(f"No existe {ruta}. Los modelos no están en git: hay que entrenar la corrida en esta máquina.")
    return config, DQN.load(ruta, device="cpu"), cual


def simular(env: PastoreoEnv, politica, semilla: int) -> tuple[list[dict], bool]:
    """
    Corre un episodio y devuelve un cuadro por paso de simulación. `politica(env, obs)` devuelve
    (obs, terminated, truncated) y puede dejar en env.decision_actual qué decidió, para mostrarlo.
    """
    sim = env.sim
    cuadros = []

    def foto():
        cuadros.append({
            "A": sim.A.copy(), "H": sim.H.copy(), "S": sim.S.copy(), "gcm": sim.gcm.copy(),
            "far": sim.A[sim.far_sheep_idx].copy(), "d_max": sim.d_max, "dist_goal": sim.dist_goal,
            "completo": sim.task_completed, "paso": len(cuadros), **env.decision_actual,  # cuadro k = paso k
        })

    # Engancha la foto al final de cada paso de simulación (update_metrics se llama una vez por paso)
    original = sim.update_metrics

    def update_metrics_con_foto():
        original()
        if grabando:
            foto()

    grabando = False
    sim.update_metrics = update_metrics_con_foto
    obs, _ = env.reset(seed=semilla)
    env.decision_actual = {"accion": "", "q": float("nan")}
    foto()
    grabando = True
    terminado = truncado = False
    while not (terminado or truncado):
        obs, terminado, truncado = politica(env, obs)
    sim.update_metrics = original
    return cuadros, terminado


def politica_dqn(model):
    def paso(env, obs):
        with th.no_grad():
            q = model.q_net(th.as_tensor(obs, dtype=th.float32).unsqueeze(0))[0].numpy()
        accion = int(np.argmax(q))
        # Q sin shaping (Q + Phi): con cualquier recompensa nuestra, el retorno real máximo es 1
        env.decision_actual = {"accion": NOMBRES_ACCIONES[accion], "q": float(q[accion]) + env.potencial_actual()}
        obs, _, terminado, truncado, _ = env.step(accion)
        return obs, terminado, truncado
    return paso


def politica_strombom(semilla: int):
    pastor = StrombomDiscreto(seed=semilla)

    def paso(env, obs):
        env.decision_actual = {"accion": "Strömbom", "q": float("nan")}
        obs, _, terminado, truncado, _ = pastor.paso(env)
        return obs, terminado, truncado
    return paso


# =====================================================================
# 2. Dibujar: mismo estilo que Pastoreo_flechitas.py
# =====================================================================
def limites(cuadros: list[dict], radio_meta: float, margen: float = 15.0, minimo: float = 60.0):
    """Cuadrado que contiene la meta y todo lo que recorren pastor y ovejas en el episodio."""
    puntos = np.vstack([np.vstack([c["A"], c["S"][None]]) for c in cuadros] + [TARGET[None]])
    lo = puntos.min(axis=0) - margen
    hi = puntos.max(axis=0) + margen
    lo = np.minimum(lo, TARGET - radio_meta - 5)
    hi = np.maximum(hi, TARGET + radio_meta + 5)
    centro, lado = (lo + hi) / 2, max(minimo, float(np.max(hi - lo)))
    return (centro[0] - lado / 2, centro[0] + lado / 2), (centro[1] - lado / 2, centro[1] + lado / 2)


def animar(episodios, titulo, env, campo_completo, intervalo):
    # Lista plana de (episodio, cuadro), con una pausa al final de cada episodio
    secuencia = []
    for i, (cuadros, _, _) in enumerate(episodios):
        secuencia += [(i, j) for j in range(len(cuadros))] + [(i, len(cuadros) - 1)] * PAUSA_FINAL

    sim = env.sim
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("#f4f6f0")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(titulo, fontsize=10, fontweight="bold")

    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.6,
            label=f"Campo inicial ({int(L)}x{int(L)} m)")
    if sim.bounds is not None:
        lo, hi = sim.bounds
        ax.plot([lo, hi, hi, lo, lo], [lo, lo, hi, hi, lo], color="#6d4c41", linewidth=2.5, alpha=0.8, label="Cerco")
    ax.add_patch(Circle(TARGET, sim.target_radius, color="#2ecc71", alpha=0.35, label="Meta"))
    ax.plot(*TARGET, "g+", markersize=10, markeredgewidth=2)

    rs_circle = Circle((0, 0), r_s, color="#e74c3c", fill=False, linestyle=":", linewidth=1.5, alpha=0.7,
                       label=f"Detección oveja ($r_s={int(r_s)}$ m)")
    stop_circle = Circle((0, 0), 3.0 * r_a, color="#c0392b", fill=True, alpha=0.15,
                         label=f"Distancia crítica ($3r_a={3 * r_a:.0f}$ m)")
    fn_circle = Circle((0, 0), sim.f_N, color="#3498db", fill=False, linestyle="--", linewidth=1.2, alpha=0.7,
                       label=f"Cohesión $f(N)={sim.f_N:.1f}$ m")
    for c in (rs_circle, stop_circle, fn_circle):
        ax.add_patch(c)

    primero = episodios[0][0][0]
    ovejas = ax.quiver(primero["A"][:, 0], primero["A"][:, 1], primero["H"][:, 0], primero["H"][:, 1],
                       color="#2c3e50", scale=30, width=0.0045, headwidth=4, label="Ovejas")
    rastro_pastor, = ax.plot([], [], "-", color="#e74c3c", linewidth=1.2, alpha=0.5, label="Recorrido del pastor")
    rastro_gcm, = ax.plot([], [], "-", color="#2980b9", linewidth=1.2, alpha=0.5, label="Recorrido del GCM")
    lejana, = ax.plot([], [], "o", color="#e67e22", markersize=10, fillstyle="none", markeredgewidth=2,
                      label="Oveja más lejana")
    gcm, = ax.plot([], [], "bx", markersize=8, markeredgewidth=2, label="Centro de masa (GCM)")
    pastor, = ax.plot([], [], "ro", markersize=10, markeredgecolor="black", label="Pastor")
    texto = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top", fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#bdc3c7"))
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

    estado = {"episodio": -1}

    def update(k):
        i, j = secuencia[k]
        cuadros, exito, semilla = episodios[i]
        c = cuadros[j]
        if i != estado["episodio"]:  # nuevo episodio: zoom
            estado["episodio"] = i
            if campo_completo:
                ax.set_xlim(WORLD_MIN, WORLD_MAX)
                ax.set_ylim(WORLD_MIN, WORLD_MAX)
            else:
                xl, yl = limites(cuadros, sim.target_radius)
                ax.set_xlim(*xl)
                ax.set_ylim(*yl)

        ovejas.set_offsets(c["A"])
        ovejas.set_UVC(c["H"][:, 0], c["H"][:, 1])
        ovejas.set_color(np.where(np.linalg.norm(c["A"] - c["S"], axis=1) <= r_s, "#2c3e50", "#95a5a6"))
        pastor.set_data([c["S"][0]], [c["S"][1]])
        rs_circle.center = stop_circle.center = tuple(c["S"])
        gcm.set_data([c["gcm"][0]], [c["gcm"][1]])
        fn_circle.center = tuple(c["gcm"])
        lejana.set_data([c["far"][0]], [c["far"][1]])
        hasta = cuadros[: j + 1]
        rastro_pastor.set_data([q["S"][0] for q in hasta], [q["S"][1] for q in hasta])
        rastro_gcm.set_data([q["gcm"][0] for q in hasta], [q["gcm"][1] for q in hasta])

        if j == len(cuadros) - 1:
            resultado = "ÉXITO" if exito else "NO LLEGÓ (se acabó el tiempo)"
        else:
            resultado = "en curso"
        q = "" if np.isnan(c["q"]) else f"{c['q']:6.2f}"
        texto.set_text(
            f"Episodio       : {i + 1}/{len(episodios)}  (mapa {semilla})\n"
            f"Paso           : {c['paso']:4d} / {env.max_pasos}\n"
            f"Acción         : {c['accion']}\n"
            + (f"Q sin shaping  : {q}\n" if q else "")
            + f"Dist. GCM meta : {c['dist_goal']:5.1f} m\n"
            f"Rebaño compacto: {'SÍ' if c['d_max'] <= sim.f_N else 'NO'} ({c['d_max']:.1f} / {sim.f_N:.1f} m)\n"
            f"Resultado      : {resultado}"
        )

    ani = FuncAnimation(fig, update, frames=len(secuencia), interval=intervalo, blit=False, repeat=True)
    plt.tight_layout()
    return fig, ani


# =====================================================================
def main():
    p = argparse.ArgumentParser(description="Animación de un agente entrenado con entrenar_sb3.py")
    p.add_argument("nombre", help="nombre de la corrida (como en resultados/registro.csv)")
    p.add_argument("--final", action="store_true", help="usar el modelo final en vez del mejor checkpoint")
    p.add_argument("--strombom", action="store_true", help="mostrar el pastor de Strömbom en el mismo entorno")
    p.add_argument("--episodios", type=int, default=3)
    p.add_argument("--semilla", type=int, default=20000, help="mapa del primer episodio (los siguientes, +1)")
    p.add_argument("--guardar", metavar="ARCHIVO.gif", help="guardar un GIF en vez de abrir la ventana")
    p.add_argument("--campo-completo", action="store_true", help="sin zoom: mostrar el mundo entero")
    p.add_argument("--intervalo", type=int, default=30, help="milisegundos por paso de simulación")
    args = p.parse_args()

    config, model, cual = cargar_corrida(args.nombre, args.final)
    env = PastoreoEnv(**config["entorno"])
    if args.strombom:
        politica, quien = politica_strombom(args.semilla), "Strömbom (9 acciones)"
    else:
        politica, quien = politica_dqn(model), cual

    episodios = []
    for i in range(args.episodios):
        semilla = args.semilla + i
        cuadros, exito = simular(env, politica, semilla)
        episodios.append((cuadros, exito, semilla))
        print(f"  mapa {semilla}: {'éxito' if exito else 'no llegó'} en {cuadros[-1]['paso']} pasos")

    titulo = f"{args.nombre}\n{quien}"
    fig, ani = animar(episodios, titulo, env, args.campo_completo, args.intervalo)
    if args.guardar:
        print(f"Guardando {args.guardar} ...")
        ani.save(args.guardar, writer=PillowWriter(fps=max(1, 1000 // args.intervalo)))
        print("Listo.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
