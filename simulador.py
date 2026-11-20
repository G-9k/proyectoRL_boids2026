"""
Simulador interactivo: ver en vivo cómo juega un modelo entrenado, en cualquier escenario.

    python simulador.py                                  # el mejor checkpoint más reciente
    python simulador.py --modelo <nombre>                # un modelo en particular (nombre de la corrida)
    python simulador.py --modelo <nombre> --final        # su modelo final en vez del mejor checkpoint
    python simulador.py --escenario angulo_180           # escenario con el que arranca

En la ventana:
  - Escenario: cualquiera de escenarios.yaml, con su descripción. El modelo se puede probar en
    escenarios distintos del que se entrenó (por ejemplo, uno de 180° en el trivial).
  - Pastor: el modelo (DQN), la heurística de Strömbom o vos con las flechas.
  - Click izquierdo en el campo: mueve al pastor ahí. Click derecho: mueve el rebaño ahí.
    Sirve para armar a mano la situación que quieras (también con la simulación pausada).
  - Teclas: espacio = play/pausa, s = un paso, r = reiniciar el mismo mapa, n = mapa nuevo,
    z = zoom / campo completo, flechas = mover al pastor en modo manual.
  - Velocidad: pasos de simulación por segundo (tiempo real). Hasta ~40 pasos/s se dibuja cada paso;
    más rápido, la pantalla va a ~50 cuadros/s y cada cuadro avanza un par de pasos.

Con la simulación pausada, el panel muestra el valor Q (sin shaping) de las 9 acciones en el estado
actual: qué haría el agente y con cuánta convicción.
"""
import argparse
import glob
import os
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.patches import Circle
from matplotlib.transforms import offset_copy
from matplotlib.widgets import Button, RadioButtons, Slider

from entrenar_sb3 import DEFAULTS, cargar_escenarios, kwargs_entorno
from Pastoreo_flechitas import L, TARGET, WORLD_MAX, WORLD_MIN, delta_shepherd, r_a, r_s
from pastoreo_env import PastoreoEnv, medir
from strombom_politica import strombom_direccion
from ver_agente import NOMBRES_ACCIONES, cargar_corrida

FLECHAS = ["·", "↑", "↓", "←", "→", "↖", "↗", "↙", "↘"]
CONTROLES = ["DQN (modelo)", "Strömbom", "Manual (flechas)"]
TECLAS_MATPLOTLIB = ("keymap.back", "keymap.forward", "keymap.home", "keymap.save", "keymap.fullscreen",
                     "keymap.pan", "keymap.zoom", "keymap.xscale", "keymap.yscale", "keymap.grid",
                     "keymap.grid_minor", "keymap.quit")
MS_POR_CUADRO = 1           # el reloj consulta seguido; los pasos los marca el tiempo real, no el reloj
MAX_PASOS_POR_CUADRO = 10   # si la máquina no da abasto, mejor ir más lento que saltar de a muchos pasos
SEG_ENTRE_PANELES = 0.25    # el panel de números se refresca 4 veces por segundo (el texto es lo caro de dibujar)
ANCHO_ETIQUETA = 17         # caracteres de la columna de etiquetas del panel ("Decide de nuevo: ")


def mejor_checkpoint_mas_reciente() -> str:
    candidatos = glob.glob(os.path.join("modelos", "*_mejor.zip"))
    if not candidatos:
        raise SystemExit("No hay ningún modelos/*_mejor.zip. Entrená algo o pasá --modelo <nombre>.")
    return os.path.basename(max(candidatos, key=os.path.getmtime))[: -len("_mejor.zip")]


class Simulador:
    def __init__(self, nombre_modelo, final, escenario, semilla):
        self.escenarios = cargar_escenarios()
        if escenario not in self.escenarios:
            raise SystemExit(f"No existe el escenario '{escenario}'. Opciones: {list(self.escenarios)}")
        self.nombre_modelo = nombre_modelo
        self.cfg_modelo, self.model, self.cual = cargar_corrida(nombre_modelo, final)
        # El modelo decide cada k pasos, como cuando se entrenó (no como diga el escenario)
        self.k_modelo = int(self.cfg_modelo["entorno"]["repetir_accion"])

        self.escenario = escenario
        self.semilla = semilla
        self.control = CONTROLES[0]
        self.jugando = False
        self.velocidad = 30         # pasos de simulación por segundo
        self.pendientes = 0.0       # fracción de paso acumulada entre cuadros
        self.t_cuadro = self.t_panel = time.perf_counter()
        self.fondo_campo = self.fondo_info = None   # lo que no se mueve, para el blitting
        self.zoom = True
        self.rng_strombom = np.random.default_rng(0)

        self._armar_ventana()
        self.cambiar_escenario(escenario)

    # -----------------------------------------------------------------
    # Simulación
    # -----------------------------------------------------------------
    def cambiar_escenario(self, nombre):
        self.escenario = nombre
        params = {**DEFAULTS, **self.escenarios[nombre]["config"]}
        kwargs = kwargs_entorno(params)
        # Recompensa y gamma del modelo, para que "Q sin shaping" se calcule con el mismo Phi
        kwargs.update(recompensa=self.cfg_modelo["entorno"]["recompensa"], gamma=self.cfg_modelo["entorno"]["gamma"])
        self.env = PastoreoEnv(**kwargs)
        self.params = params
        self.reiniciar()
        self._actualizar_descripcion()

    def reiniciar(self, semilla=None):
        if semilla is not None:
            self.semilla = semilla
        self.env.reset(seed=self.semilla)
        self.restantes = 0          # pasos que le quedan a la decisión actual del DQN
        self.accion = None
        self.q = None
        self.terminado = False
        self.rastro_S, self.rastro_gcm = [self.env.sim.S.copy()], [self.env.sim.gcm.copy()]
        self.jugando = False
        self._q_estado_actual()
        self._nuevo_rebano_grafico()
        self._ajustar_vista()
        self.dibujar()

    def paso(self):
        """Un paso de simulación con el controlador elegido."""
        if self.terminado:
            return
        env, sim = self.env, self.env.sim
        if self.control == CONTROLES[0]:
            if self.restantes == 0:
                self._q_estado_actual()
                self.accion = int(np.argmax(self.q))
                self.restantes = self.k_modelo
            self.restantes -= 1
            accion = self.accion
            mover = lambda: sim.step_shepherd_action(accion)
        elif self.control == CONTROLES[1]:
            direccion, _ = strombom_direccion(sim, self.rng_strombom)
            mover = lambda: setattr(sim, "S", sim.S + delta_shepherd * np.asarray(direccion, dtype=float))
        else:
            mover = sim.step_shepherd_from_keys

        terminado = env.paso_simulacion(mover)
        env._medidas = medir(sim)
        self.rastro_S.append(sim.S.copy())
        self.rastro_gcm.append(sim.gcm.copy())
        if self.control != CONTROLES[0] or self.restantes == 0:
            self._q_estado_actual()   # para que el panel muestre los Q del estado nuevo
        if terminado:
            self.terminado = True
            self.jugando = False

    def _q_estado_actual(self):
        obs = th.as_tensor(self.env._obs(), dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            q = self.model.q_net(obs)[0].numpy()
        self.q_crudo = q
        self.q = q + self.env.potencial_actual()   # sin shaping

    def mover_pastor_a(self, x, y):
        self.env.sim.S = np.array([x, y], dtype=float)
        self._tras_edicion()

    def mover_rebano_a(self, x, y):
        sim = self.env.sim
        sim.A += np.array([x, y]) - np.mean(sim.A, axis=0)
        sim.task_completed = False
        self._tras_edicion()

    def _tras_edicion(self):
        sim = self.env.sim
        sim._apply_bounds()
        sim.update_metrics()
        self.env._medidas = medir(sim)
        self.terminado = bool(sim.task_completed) or self.env.pasos >= self.env.max_pasos
        self.restantes = 0          # que el DQN vuelva a decidir ya
        self.rastro_S, self.rastro_gcm = [sim.S.copy()], [sim.gcm.copy()]
        self._q_estado_actual()
        self.dibujar()

    # -----------------------------------------------------------------
    # Ventana
    # -----------------------------------------------------------------
    def _armar_ventana(self):
        for k in TECLAS_MATPLOTLIB:
            if k in plt.rcParams:
                plt.rcParams[k] = []
        self.fig = plt.figure(figsize=(15, 8.6))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Simulador de pastoreo")

        ax = self.ax = self.fig.add_axes([0.02, 0.04, 0.55, 0.92])
        ax.set_aspect("equal")
        ax.set_facecolor("#f4f6f0")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color="#7f8c8d", linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"Campo inicial ({int(L)}x{int(L)} m)")
        self.cerco, = ax.plot([], [], color="#6d4c41", linewidth=2.5, alpha=0.8, label="Cerco")
        self.meta = Circle(TARGET, 10, color="#2ecc71", alpha=0.35, label="Meta")
        ax.add_patch(self.meta)
        ax.plot(*TARGET, "g+", markersize=10, markeredgewidth=2)
        self.rs_circle = Circle((0, 0), r_s, color="#e74c3c", fill=False, linestyle=":", linewidth=1.5, alpha=0.7,
                                label=f"La oveja siente al pastor ($r_s={int(r_s)}$ m)")
        self.stop_circle = Circle((0, 0), 3 * r_a, color="#c0392b", alpha=0.15, label=f"$3r_a={3 * r_a:.0f}$ m")
        self.fn_circle = Circle((0, 0), 1, color="#3498db", fill=False, linestyle="--", linewidth=1.2, alpha=0.7,
                                label="Cohesión $f(N)$")
        for c in (self.rs_circle, self.stop_circle, self.fn_circle):
            ax.add_patch(c)
        self.rastro_pastor, = ax.plot([], [], "-", color="#e74c3c", linewidth=1.2, alpha=0.5, label="Recorrido del pastor")
        self.rastro_centro, = ax.plot([], [], "-", color="#2980b9", linewidth=1.2, alpha=0.5, label="Recorrido del GCM")
        self.lejana, = ax.plot([], [], "o", color="#e67e22", markersize=10, fillstyle="none", markeredgewidth=2,
                               label="Oveja más lejana")
        self.gcm, = ax.plot([], [], "bx", markersize=8, markeredgewidth=2, label="Centro del rebaño (GCM)")
        self.pastor, = ax.plot([], [], "ro", markersize=10, markeredgecolor="black", label="Pastor", zorder=5)
        self.ovejas = None
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        ax.set_title(f"Modelo: {self.nombre_modelo}  ({self.cual})", fontsize=9, fontweight="bold")

        # Panel derecho
        f = self.fig
        ax_esc = f.add_axes([0.59, 0.55, 0.18, 0.40])
        ax_esc.set_title("Escenario", fontsize=10, fontweight="bold", loc="left")
        nombres = list(self.escenarios)
        self.radio_esc = RadioButtons(ax_esc, nombres, active=nombres.index(self.escenario))
        self.radio_esc.on_clicked(self.cambiar_escenario)

        ax_ctrl = f.add_axes([0.79, 0.80, 0.19, 0.15])
        ax_ctrl.set_title("Pastor", fontsize=10, fontweight="bold", loc="left")
        self.radio_ctrl = RadioButtons(ax_ctrl, CONTROLES, active=0)
        self.radio_ctrl.on_clicked(self._cambiar_control)

        self.ax_desc = f.add_axes([0.59, 0.20, 0.18, 0.33])
        self.ax_desc.axis("off")
        self.texto_desc = self.ax_desc.text(0, 1, "", va="top", fontsize=8.5, wrap=False)

        # Panel de números en dos columnas: las etiquetas casi no cambian (van en el fondo) y los valores
        # se redibujan en cada refresco. Dibujar texto cuesta ~50 µs por carácter, así que conviene
        # redibujar solo los valores. Fuente monoespaciada: cada carácter mide 0,602 del tamaño.
        self.ax_info = f.add_axes([0.79, 0.20, 0.19, 0.58])
        self.ax_info.axis("off")
        self.texto_etiquetas = self.ax_info.text(0, 1, "", va="top", fontsize=9, family="monospace")
        desplazamiento = offset_copy(self.ax_info.transAxes, fig=f, x=ANCHO_ETIQUETA * 0.602 * 9, units="points")
        self.texto_info = self.ax_info.text(0, 1, "", va="top", fontsize=9, family="monospace", transform=desplazamiento)

        botones = [("Play / Pausa", self._play_pausa), ("Un paso", lambda e: self._un_paso()),
                   ("Reiniciar", lambda e: self.reiniciar()), ("Mapa nuevo", lambda e: self.reiniciar(self.semilla + 1))]
        self.botones = []
        for i, (texto, fn) in enumerate(botones):
            b = Button(f.add_axes([0.59 + i * 0.1, 0.11, 0.09, 0.05]), texto)
            b.on_clicked(fn)
            self.botones.append(b)
        self.slider = Slider(f.add_axes([0.66, 0.065, 0.25, 0.025]), "Pasos/s", 5, 300, valinit=self.velocidad, valstep=5)
        self.slider.on_changed(lambda v: setattr(self, "velocidad", int(v)))
        f.text(0.59, 0.012, "Teclas: espacio = play/pausa · s = un paso · r = reiniciar · n = mapa nuevo · z = zoom\n"
                            "Mouse: click izquierdo = mover al pastor · click derecho = mover al rebaño",
               fontsize=8, color="#555")

        f.canvas.mpl_connect("key_press_event", self._tecla)
        f.canvas.mpl_connect("key_release_event", self._tecla_suelta)
        f.canvas.mpl_connect("button_press_event", self._click)

        # Blitting: lo que se mueve (pastor, ovejas, rastros, panel de números) se marca "animated" y
        # el dibujo completo lo saltea. Cada vez que se dibuja todo (al arrancar, al cambiar de
        # escenario, al tocar un botón) se guarda ese fondo fijo; en cada cuadro se pega el fondo y se
        # dibuja encima solo lo que se mueve. Redibujar todo tarda ~0,3 s (el texto es caro), esto ~ms.
        for a in self._animados_campo() + [self.texto_info]:
            a.set_animated(True)
        f.canvas.mpl_connect("draw_event", self._al_dibujar_todo)
        self.reloj = f.canvas.new_timer(interval=MS_POR_CUADRO)
        self.reloj.add_callback(self._cuadro)
        self.reloj.start()
        f.canvas.mpl_connect("close_event", lambda e: self.reloj.stop())

    def _animados_campo(self):
        animados = [self.rs_circle, self.stop_circle, self.fn_circle, self.rastro_pastor, self.rastro_centro,
                    self.lejana, self.gcm, self.pastor]
        return animados + ([self.ovejas] if self.ovejas is not None else [])

    def _al_dibujar_todo(self, _):
        c = self.fig.canvas
        self.fondo_campo = c.copy_from_bbox(self.ax.bbox)
        self.fondo_info = c.copy_from_bbox(self.ax_info.bbox)
        for a in self._animados_campo():
            self.ax.draw_artist(a)
        self.ax_info.draw_artist(self.texto_info)

    def _nuevo_rebano_grafico(self):
        sim = self.env.sim
        if self.ovejas is not None:
            self.ovejas.remove()
        self.ovejas = self.ax.quiver(sim.A[:, 0], sim.A[:, 1], sim.H[:, 0], sim.H[:, 1], color="#2c3e50",
                                     scale=30, width=0.0045, headwidth=4, zorder=4, animated=True)
        self.fn_circle.set_radius(sim.f_N)
        self.meta.set_radius(sim.target_radius)
        if sim.bounds is not None:
            lo, hi = sim.bounds
            self.cerco.set_data([lo, hi, hi, lo, lo], [lo, lo, hi, hi, lo])
        else:
            self.cerco.set_data([], [])

    def _ajustar_vista(self):
        if not self.zoom:
            self.ax.set_xlim(WORLD_MIN, WORLD_MAX)
            self.ax.set_ylim(WORLD_MIN, WORLD_MAX)
            return
        sim = self.env.sim
        puntos = np.vstack([sim.A, sim.S[None], TARGET[None]])
        lo, hi = puntos.min(axis=0) - 25, puntos.max(axis=0) + 25
        centro, lado = (lo + hi) / 2, max(80.0, float(np.max(hi - lo)))
        self.ax.set_xlim(centro[0] - lado / 2, centro[0] + lado / 2)
        self.ax.set_ylim(centro[1] - lado / 2, centro[1] + lado / 2)

    def _asegurar_visible(self) -> bool:
        """Si el pastor o alguna oveja se salen de la vista con zoom, la agranda para incluirlos.
        Devuelve True si cambió la vista (entonces hay que redibujar todo, no alcanza con el blitting)."""
        if not self.zoom:
            return False
        sim = self.env.sim
        puntos = np.vstack([sim.A, sim.S[None]])
        (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
        if (puntos[:, 0].min() > x0 + 3 and puntos[:, 0].max() < x1 - 3
                and puntos[:, 1].min() > y0 + 3 and puntos[:, 1].max() < y1 - 3):
            return False
        lo = np.minimum([x0, y0], puntos.min(axis=0) - 60)  # margen amplio: cada agrandada es un redibujo completo (~0,3 s)
        hi = np.maximum([x1, y1], puntos.max(axis=0) + 60)
        centro, lado = (lo + hi) / 2, float(np.max(hi - lo))
        self.ax.set_xlim(centro[0] - lado / 2, centro[0] + lado / 2)
        self.ax.set_ylim(centro[1] - lado / 2, centro[1] + lado / 2)
        return True

    def _actualizar_descripcion(self):
        p, e = self.params, self.escenarios[self.escenario]
        envolver = lambda s: "\n".join(textwrap.wrap(s, 40))
        nada = "como el paper"
        lineas = [
            f"{self.escenario}", "",
            envolver(e["descripcion"]), "",
            f"ovejas: {p['n_ovejas']}",
            f"rebaño a la meta: {p['dist_inicio_rebano'] or nada}" + (" m" if p["dist_inicio_rebano"] else ""),
            f"pastor al rebaño: {('5-' + str(p['dist_inicio_pastor']) + ' m') if p['dist_inicio_pastor'] else nada}",
            f"ángulo del pastor: {'±' + str(p['angulo_pastor']) + '°' if p['angulo_pastor'] is not None else 'cualquiera'}",
            f"cerco: {'sí' if p['cerco'] else 'no'}    máx. pasos: {p['max_pasos']}", "",
            envolver(f"Modelo entrenado en: {self.cfg_modelo.get('escenario') or 'argumentos'} "
                     f"(decide cada {self.k_modelo} pasos)"),
        ]
        self.texto_desc.set_text("\n".join(lineas))

    def dibujar(self):
        """Redibuja la ventana entera (lento). Para cuando cambia algo fuera del campo o al pausar."""
        self._actualizar_artistas(panel=True)
        self.fig.canvas.draw_idle()

    def _dibujar_rapido(self, panel: bool):
        """Redibuja solo lo que se mueve, sobre el fondo guardado (blitting). panel=False deja los números como estaban."""
        vista_cambio = self._actualizar_artistas(panel)
        if vista_cambio or self.fondo_campo is None:
            self.fig.canvas.draw_idle()
            return
        c = self.fig.canvas
        c.restore_region(self.fondo_campo)
        for a in self._animados_campo():
            self.ax.draw_artist(a)
        c.blit(self.ax.bbox)
        if panel:
            c.restore_region(self.fondo_info)
            self.ax_info.draw_artist(self.texto_info)
            c.blit(self.ax_info.bbox)

    def _actualizar_artistas(self, panel: bool) -> bool:
        """Pone los datos actuales en los dibujos. Devuelve True si cambió la vista del campo."""
        sim, env = self.env.sim, self.env
        self.ovejas.set_offsets(sim.A)
        self.ovejas.set_UVC(sim.H[:, 0], sim.H[:, 1])
        self.ovejas.set_color(np.where(np.linalg.norm(sim.A - sim.S, axis=1) <= r_s, "#2c3e50", "#95a5a6"))
        self.pastor.set_data([sim.S[0]], [sim.S[1]])
        self.rs_circle.center = self.stop_circle.center = tuple(sim.S)
        self.gcm.set_data([sim.gcm[0]], [sim.gcm[1]])
        self.fn_circle.center = tuple(sim.gcm)
        far = sim.A[sim.far_sheep_idx]
        self.lejana.set_data([far[0]], [far[1]])
        rs, rg = np.array(self.rastro_S), np.array(self.rastro_gcm)
        self.rastro_pastor.set_data(rs[:, 0], rs[:, 1])
        self.rastro_centro.set_data(rg[:, 0], rg[:, 1])
        vista_cambio = self._asegurar_visible()
        if panel:
            self._actualizar_panel()
        return vista_cambio

    def _actualizar_panel(self):
        """Arma el panel como pares (etiqueta, valor). Las etiquetas solo dependen del control elegido
        y de si terminó el episodio, y en esos dos casos se redibuja todo, así que pueden ir en el fondo."""
        sim, env = self.env.sim, self.env
        if self.terminado:
            estado = f"ÉXITO en {env.pasos} pasos" if sim.task_completed else "se acabó el tiempo"
        else:
            estado = "jugando" if self.jugando else "en pausa"
        filas = [
            ("Estado", estado),
            ("Mapa", str(self.semilla)),
            ("Paso", f"{env.pasos} / {env.max_pasos}"),
            ("GCM a la meta", f"{sim.dist_goal:6.1f} m"),
            ("Pastor a oveja", f"{sim.min_dist_sheep:6.1f} m"),
            ("Dispersión", f"{sim.d_max:6.1f} m"),
            ("Cohesión f(N)", f"{sim.f_N:6.1f} m"),
            ("Compacto", "sí" if sim.d_max <= sim.f_N else "no"),
            ("", ""),
            ("Q sin shaping por acción", ""),
            ("(lo que el modelo espera ganar;", ""),
            (" el máximo real es 1)", ""),
        ]
        mejor = int(np.argmax(self.q))
        for i, (nombre, flecha) in enumerate(zip(NOMBRES_ACCIONES, FLECHAS)):
            filas.append((f" {flecha} {nombre}", f"{self.q[i]:6.3f}" + ("  ◀ elige" if i == mejor else "")))
        if self.control == CONTROLES[0] and not self.terminado:
            filas += [("", ""), ("Decide de nuevo", f"en {self.restantes} paso(s)" if self.restantes else "ahora")]
        elif self.control != CONTROLES[0]:
            filas += [("", ""), (f"(manda {self.control.split()[0]}; los Q son", ""), (" solo informativos)", "")]
        self.texto_etiquetas.set_text("\n".join(e if not v else f"{e:<{ANCHO_ETIQUETA - 2}}:" for e, v in filas))
        self.texto_info.set_text("\n".join(v for _, v in filas))

    # -----------------------------------------------------------------
    # Eventos
    # -----------------------------------------------------------------
    def _cuadro(self):
        """Lo llama el reloj todo el tiempo. Avanza los pasos que tocan según el tiempo real que pasó:
        a 30 pasos/s, un paso por cuadro; a 120 pasos/s (la pantalla da ~50 cuadros/s), dos o tres por cuadro."""
        ahora = time.perf_counter()
        dt, self.t_cuadro = ahora - self.t_cuadro, ahora
        if not self.jugando or not plt.fignum_exists(self.fig.number):
            return
        self.pendientes = min(self.pendientes + dt * self.velocidad, MAX_PASOS_POR_CUADRO)
        n = int(self.pendientes)
        if n == 0:
            return
        self.pendientes -= n
        for _ in range(n):
            self.paso()
            if self.terminado:
                break
        if self.terminado:
            self.dibujar()
            return
        panel = ahora - self.t_panel >= SEG_ENTRE_PANELES
        if panel:
            self.t_panel = ahora
        self._dibujar_rapido(panel)

    def _play_pausa(self, _=None):
        if self.terminado:  # al terminar, play arranca un mapa nuevo
            self.reiniciar(self.semilla + 1)
        self.jugando = not self.jugando
        self.pendientes = 0.0
        self.t_cuadro = time.perf_counter()
        self.dibujar()

    def _un_paso(self):
        self.jugando = False
        self.paso()
        self.dibujar()

    def _cambiar_control(self, etiqueta):
        self.control = etiqueta
        self.restantes = 0
        self.dibujar()

    def _tecla(self, ev):
        if ev.key in ("up", "down", "left", "right"):
            self.env.sim.keys_pressed.add(ev.key)
        elif ev.key == " ":
            self._play_pausa()
        elif ev.key == "s":
            self._un_paso()
        elif ev.key == "r":
            self.reiniciar()
        elif ev.key == "n":
            self.reiniciar(self.semilla + 1)
        elif ev.key == "z":
            self.zoom = not self.zoom
            self._ajustar_vista()
            self.dibujar()

    def _tecla_suelta(self, ev):
        self.env.sim.keys_pressed.discard(ev.key)

    def _click(self, ev):
        if ev.inaxes is not self.ax or ev.xdata is None:
            return
        barra = self.fig.canvas.toolbar
        if barra is not None and barra.mode:  # usando la lupa o la mano de matplotlib
            return
        if ev.button == 1:
            self.mover_pastor_a(ev.xdata, ev.ydata)
        elif ev.button == 3:
            self.mover_rebano_a(ev.xdata, ev.ydata)


def main():
    p = argparse.ArgumentParser(description="Simulador interactivo de pastoreo con un modelo entrenado")
    p.add_argument("--modelo", help="nombre de la corrida (por defecto, el mejor checkpoint más reciente)")
    p.add_argument("--final", action="store_true", help="usar el modelo final en vez del mejor checkpoint")
    p.add_argument("--escenario", default="trivial_case", help="escenario con el que arranca")
    p.add_argument("--semilla", type=int, default=20000, help="mapa con el que arranca")
    args = p.parse_args()

    nombre = args.modelo or mejor_checkpoint_mas_reciente()
    print(f"Modelo: {nombre}")
    simulador = Simulador(nombre, args.final, args.escenario, args.semilla)  # hay que guardarlo: si no, se pierden los eventos
    plt.show()
    return simulador


if __name__ == "__main__":
    main()
