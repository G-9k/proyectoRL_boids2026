"""
Entorno de pastoreo con la interfaz de Gymnasium.

Envuelve la física de `ManualStrombomEnv` (Pastoreo_flechitas.py) sin cambiarla y agrega lo que
necesita un agente de RL:
  - reset(seed=...) reproducible,
  - terminated (el rebaño llegó a la meta) separado de truncated (se acabaron los pasos),
  - la recompensa como una función intercambiable, para poder iterarla sin tocar la física,
  - parámetros para el currículum: cantidad de ovejas, dónde arranca el pastor, radio de la meta,
    máximo de pasos y cerco opcional.

Uso rápido:
    env = PastoreoEnv(n_ovejas=1, recompensa="potencial_gcm", dist_inicio_pastor=20)
    obs, info = env.reset(seed=0)
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())

Para chequear el entorno:  python pastoreo_env.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Pastoreo_flechitas import (
    ManualStrombomEnv,
    L,
    TARGET,
    TARGET_RADIUS,
    WORLD_MIN,
    WORLD_MAX,
    delta_shepherd,
    r_s,
)

# Cómo se arma la observación (el vector de 8 números que ve la red):
#   "normalizada" (desde el 20/11/2026): la distancia pastor-oveja más cercana se divide por r_s = 65 m
#                 (el radio en el que la oveja siente al pastor). Vale 1 justo en ese borde y ~2 a 130 m.
#   "original":   la de get_dqn_state() en Pastoreo_flechitas.py, que la divide por 3·r_a = 6 m. Con el
#                 pastor a 80 m valía ~13, mientras las otras 7 entradas rondan entre -1 y 1; una entrada
#                 mucho más grande que el resto le complica el aprendizaje a la red.
# Los modelos entrenados antes del cambio usan "original" (ver_agente.cargar_corrida lo completa solo).
OBSERVACIONES = ("normalizada", "original")


# =====================================================================
# MEDIDAS DEL ESTADO (lo que pueden usar las funciones de recompensa)
# =====================================================================
def medir(sim: ManualStrombomEnv) -> dict:
    """Foto de las cantidades relevantes para la recompensa, en metros."""
    dist_ovejas_meta = np.linalg.norm(sim.A - TARGET, axis=1)
    pastor_rel = sim.S - sim.gcm        # del centro del rebaño al pastor
    detras = sim.gcm - TARGET           # de la meta al centro del rebaño: "detrás" es seguir en esa dirección
    norma = np.linalg.norm(pastor_rel) * np.linalg.norm(detras)
    return {
        "dist_gcm_meta": sim.dist_goal,                  # centro del rebaño -> meta
        "dist_max_meta": float(dist_ovejas_meta.max()),  # oveja más lejana a la meta -> meta
        "dispersion_max": sim.d_max,                     # oveja más lejana al centro del rebaño
        "dist_pastor_oveja": sim.min_dist_sheep,         # pastor -> oveja más cercana
        "dist_pastor_gcm": float(np.linalg.norm(pastor_rel)),
        # cos del ángulo entre (pastor - centro) y (centro - meta): 1 = detrás, -1 = entre el rebaño y la meta
        "cos_detras": float(np.dot(pastor_rel, detras) / norma) if norma > 1e-9 else 0.0,
    }


# =====================================================================
# FUNCIONES DE RECOMPENSA
# Firma: f(antes, despues, terminated, gamma) -> float
#   antes / despues: dicts de medir() antes y después del paso.
#   terminated: True si este paso completó la tarea.
#   gamma: el mismo descuento que usa el agente (hace falta para el shaping por potencial).
# Para agregar una variante: escribir la función y sumarla a RECOMPENSAS.
# =====================================================================
def recompensa_esparsa(antes, despues, terminated, gamma):
    """+1 solo al llegar a la meta. Sin ninguna ayuda."""
    return 1.0 if terminated else 0.0


def _shaping(phi_antes, phi_despues, terminated, gamma):
    """F = gamma * Phi(s') - Phi(s), con Phi(terminal) = 0 (Ng, Harada y Russell, 1999)."""
    if terminated:
        phi_despues = 0.0
    return gamma * phi_despues - phi_antes


def phi_gcm(medidas):
    """Potencial: -distancia(centro del rebaño, meta) / L. Más cerca de la meta, mayor."""
    return -medidas["dist_gcm_meta"] / L


def recompensa_potencial_gcm(antes, despues, terminated, gamma):
    """Esparsa + shaping con Phi = phi_gcm."""
    return recompensa_esparsa(antes, despues, terminated, gamma) + _shaping(
        phi_gcm(antes), phi_gcm(despues), terminated, gamma
    )


# Cada recompensa con shaping declara su potencial. Con shaping, el valor Q que aprende la red es
# Q_sin_shaping(s, a) - Phi(s); declarar Phi permite volver a Q_sin_shaping para diagnosticar
# (ver PastoreoEnv.potencial_actual y EvalPeriodica en entrenar_sb3.py).
recompensa_potencial_gcm.potencial = phi_gcm


PENALIZACION_TIEMPO = 0.01  # por decisión (no por paso de simulación)


def recompensa_potencial_gcm_tiempo(antes, despues, terminated, gamma):
    """
    potencial_gcm - PENALIZACION_TIEMPO en cada decisión.
    No es shaping basado en potencial: cambia la tarea a "llegar RÁPIDO". Como la única forma de
    terminar el episodio es ganar, no incentiva cortar el episodio de otra manera.
    Quedarse quieto para siempre vale -PENALIZACION_TIEMPO / (1 - gamma) (= -1 con gamma 0.99), así
    que los lugares donde "no pasa nada" (por ejemplo, una esquina) dejan de parecer atractivos.
    """
    return recompensa_potencial_gcm(antes, despues, terminated, gamma) - PENALIZACION_TIEMPO


recompensa_potencial_gcm_tiempo.potencial = phi_gcm


ESCALA_POSICION = 50.0  # metros: a esta distancia del centro del rebaño, el factor de cercanía vale 1/2


def phi_posicion(medidas):
    """
    Potencial de la posición del pastor respecto del rebaño y la meta, entre 0 y 1:
        (1 + cos θ) / 2  *  D / (D + d)
    θ = ángulo entre (pastor - centro del rebaño) y (centro del rebaño - meta); d = distancia pastor-centro.
    Vale 1 justo detrás y pegado al rebaño, 0 entre el rebaño y la meta, y crece al acercarse desde
    cualquier distancia (salvo justo delante del rebaño, donde vale 0 a cualquier distancia).
    Siempre >= 0: este término no agrega premio por esperar lejos (ver GUIA_CONCEPTOS.md, 4.5).
    """
    angulo = (1.0 + medidas["cos_detras"]) / 2.0
    cercania = ESCALA_POSICION / (ESCALA_POSICION + medidas["dist_pastor_gcm"])
    return angulo * cercania


def hacer_recompensa(peso_posicion: float, penalizacion: float):
    """
    Esparsa + shaping con Phi = phi_gcm + peso_posicion * phi_posicion, menos `penalizacion` por decisión.
    Todo lo que es shaping está en Phi (basado en potencial), así que no se puede farmear.
    """
    def phi(medidas):
        return phi_gcm(medidas) + peso_posicion * phi_posicion(medidas)

    def recompensa(antes, despues, terminated, gamma):
        return (recompensa_esparsa(antes, despues, terminated, gamma)
                + _shaping(phi(antes), phi(despues), terminated, gamma)
                - penalizacion)

    recompensa.potencial = phi
    return recompensa


RECOMPENSAS = {
    "esparsa": recompensa_esparsa,
    "potencial_gcm": recompensa_potencial_gcm,
    "potencial_gcm_tiempo": recompensa_potencial_gcm_tiempo,
    # Penalización doble: lejos de la meta, (gamma - 1) * phi_gcm ~ +0.01 anula la de 0.01
    "potencial_gcm_tiempo_fuerte": hacer_recompensa(peso_posicion=0.0, penalizacion=2 * PENALIZACION_TIEMPO),
    # Con la posición del pastor (detrás del rebaño, cerca), con dos pesos
    "potencial_gcm_pos05_tiempo": hacer_recompensa(peso_posicion=0.5, penalizacion=PENALIZACION_TIEMPO),
    "potencial_gcm_pos1_tiempo": hacer_recompensa(peso_posicion=1.0, penalizacion=PENALIZACION_TIEMPO),
}


# =====================================================================
# ENTORNO
# =====================================================================
class PastoreoEnv(gym.Env):
    """
    Observación: los 8 valores de ManualStrombomEnv.get_dqn_state() (float32), con la última entrada
    (distancia a la oveja más cercana) dividida por r_s si observacion="normalizada" (ver OBSERVACIONES).
    Acciones: Discrete(9). 0 = quieto, 1..8 = las 8 direcciones (ver ACTION_VECTORS).
    terminated: el centro del rebaño quedó a menos de `radio_meta` de la meta.
    truncated: se llegó a `max_pasos` sin completar la tarea.

    Parámetros:
      n_ovejas: cantidad de ovejas.
      recompensa: nombre en RECOMPENSAS o una función con la misma firma.
      gamma: descuento del agente; tiene que coincidir con el del DQN para que el shaping sea correcto.
      max_pasos: límite de pasos por episodio.
      radio_meta: radio de éxito alrededor de la meta (el paper usa 10 m).
      cerco: si es True, ovejas y pastor no pueden salir de [WORLD_MIN, WORLD_MAX]^2
             (el mismo cuadrado que muestra la cámara).
      dist_inicio_pastor: None = el pastor arranca como en el paper (cuarto inferior izquierdo).
             Un número = el pastor arranca en una dirección al azar, entre 5 m y esta distancia
             por fuera del borde del rebaño. Sirve para el currículum ("pastor cerca").
      angulo_pastor: solo se usa junto con dist_inicio_pastor. None = la dirección del pastor
             alrededor del rebaño es al azar (cualquier lado). Un número a (en grados) = el pastor
             arranca DETRÁS del rebaño respecto de la meta (rebaño entre el pastor y la meta),
             desviado al azar entre -a y +a de esa línea. 0 = perfectamente alineado;
             180 = equivale a cualquier dirección.
      dist_inicio_rebano: None = el rebaño arranca como en el paper (cuarto superior derecho,
             ~150 m de la meta). Un número = el rebaño se traslada (sin cambiar su forma) para que
             su centro quede a esa distancia de la meta, en un ángulo al azar entre 0 y 90 grados.
             Sirve para el currículum ("rebaño cerca de la meta").
      repetir_accion: cuántos pasos de simulación dura cada acción elegida (frame skip). Con k > 1
             el agente decide cada k pasos; `max_pasos` y info["pasos"] siguen contando pasos de
             simulación. El gamma es por decisión, no por paso de simulación.
      observacion: "normalizada" (por defecto) u "original"; ver OBSERVACIONES. Un modelo se tiene
             que usar con la misma observación con la que se entrenó.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_ovejas: int = 30,
        recompensa="potencial_gcm",
        gamma: float = 0.99,
        max_pasos: int = 2000,
        radio_meta: float = TARGET_RADIUS,
        cerco: bool = False,
        dist_inicio_pastor: float | None = None,
        dist_inicio_rebano: float | None = None,
        repetir_accion: int = 1,
        angulo_pastor: float | None = None,
        observacion: str = "normalizada",
    ):
        super().__init__()
        if observacion not in OBSERVACIONES:
            raise ValueError(f"Observación desconocida '{observacion}'. Opciones: {OBSERVACIONES}")
        self.observacion = observacion
        if isinstance(recompensa, str):
            if recompensa not in RECOMPENSAS:
                raise ValueError(f"Recompensa desconocida '{recompensa}'. Opciones: {list(RECOMPENSAS)}")
            self.nombre_recompensa = recompensa
            self.recompensa_fn = RECOMPENSAS[recompensa]
        else:
            self.nombre_recompensa = getattr(recompensa, "__name__", "custom")
            self.recompensa_fn = recompensa

        self.gamma = float(gamma)
        self.max_pasos = int(max_pasos)
        self.dist_inicio_pastor = dist_inicio_pastor
        self.dist_inicio_rebano = dist_inicio_rebano
        self.repetir_accion = int(repetir_accion)
        self.angulo_pastor = angulo_pastor
        if angulo_pastor is not None and dist_inicio_pastor is None:
            raise ValueError("angulo_pastor necesita dist_inicio_pastor")

        self.sim = ManualStrombomEnv(
            n_sheep=n_ovejas,
            target_radius=radio_meta,
            bounds=(WORLD_MIN, WORLD_MAX) if cerco else None,
        )

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ManualStrombomEnv.ACTION_VECTORS))

        self.pasos = 0
        self._medidas = medir(self.sim)

    # -----------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # La simulación usa el generador de Gymnasium: misma semilla -> mismo episodio.
        self.sim.rng = self.np_random
        self.sim.reset()

        if self.dist_inicio_rebano is not None:
            self._ubicar_rebano()
        if self.dist_inicio_pastor is not None:
            self._ubicar_pastor_cerca()

        self.pasos = 0
        self._medidas = medir(self.sim)
        return self._obs(), self._info(exito=False)

    def step(self, action):
        return self._avanzar(lambda: self.sim.step_shepherd_action(int(action)))

    def step_continuo(self, direccion):
        """
        Como step(), pero el pastor se mueve en una dirección cualquiera (vector de norma <= 1,
        multiplicado por la velocidad del pastor). No es parte de la interfaz de Gym: lo usa la
        heurística de Strömbom original, que no está limitada a 9 direcciones.
        """
        def mover():
            self.sim.S = self.sim.S + delta_shepherd * np.asarray(direccion, dtype=float)
        return self._avanzar(mover)

    def _avanzar(self, mover_pastor):
        """
        Una decisión del agente: repite la acción `repetir_accion` pasos de simulación (o menos, si el
        episodio termina antes). En cada paso se mueve el pastor y después las ovejas.
        La recompensa se calcula una sola vez, entre el estado antes y después de la decisión.
        """
        antes = self._medidas

        for _ in range(self.repetir_accion):
            if self.paso_simulacion(mover_pastor):
                break

        despues = medir(self.sim)
        terminated = bool(self.sim.task_completed)
        truncated = (not terminated) and self.pasos >= self.max_pasos
        reward = float(self.recompensa_fn(antes, despues, terminated, self.gamma))

        self._medidas = despues
        return self._obs(), reward, terminated, truncated, self._info(exito=terminated)

    def paso_simulacion(self, mover_pastor) -> bool:
        """
        Un solo paso de simulación: se mueve el pastor, después las ovejas. Devuelve True si el
        episodio terminó (éxito o max_pasos). No calcula recompensa: eso lo hace _avanzar por decisión.
        Lo usa también simulador.py para avanzar paso a paso.
        """
        mover_pastor()
        self.sim.step_sheep()
        self.sim._apply_bounds()
        self.sim.update_metrics()
        self.pasos += 1
        return bool(self.sim.task_completed) or self.pasos >= self.max_pasos

    # -----------------------------------------------------------------
    def _ubicar_rebano(self):
        angulo = self.np_random.uniform(0.0, np.pi / 2.0)
        centro = TARGET + float(self.dist_inicio_rebano) * np.array([np.cos(angulo), np.sin(angulo)])
        self.sim.A += centro - np.mean(self.sim.A, axis=0)
        self.sim._apply_bounds()
        self.sim.update_metrics()

    def _ubicar_pastor_cerca(self):
        rng = self.np_random
        if self.angulo_pastor is None:
            angulo = rng.uniform(0.0, 2.0 * np.pi)
        else:
            # Dirección "detrás del rebaño": de la meta hacia el centro del rebaño, y más allá
            detras = self.sim.gcm - TARGET
            desvio = np.deg2rad(float(self.angulo_pastor))
            angulo = np.arctan2(detras[1], detras[0]) + rng.uniform(-desvio, desvio)
        distancia = self.sim.d_max + rng.uniform(5.0, max(5.0, float(self.dist_inicio_pastor)))
        self.sim.S = self.sim.gcm + distancia * np.array([np.cos(angulo), np.sin(angulo)])
        self.sim._apply_bounds()
        self.sim.update_metrics()

    def potencial_actual(self) -> float:
        """Phi del estado actual según la recompensa en uso (0 si la recompensa no tiene shaping)."""
        phi = getattr(self.recompensa_fn, "potencial", None)
        return float(phi(self._medidas)) if phi is not None else 0.0

    def _obs(self):
        obs = self.sim.get_dqn_state()
        if self.observacion == "normalizada":
            obs[7] = self.sim.min_dist_sheep / r_s
        return obs.astype(np.float32)

    def _info(self, exito: bool):
        # "is_success" lo leen los callbacks de evaluación de SB3 para calcular la tasa de éxito.
        return {"is_success": exito, "pasos": self.pasos, **self._medidas}


# =====================================================================
# CHEQUEOS: python pastoreo_env.py
# =====================================================================
if __name__ == "__main__":
    import time

    from stable_baselines3.common.env_checker import check_env

    # 1. Interfaz de Gymnasium (incluye el chequeo de SB3)
    for kwargs in [dict(), dict(n_ovejas=1, dist_inicio_pastor=20), dict(n_ovejas=3, cerco=True)]:
        check_env(PastoreoEnv(**kwargs), warn=True)
    print("check_env: OK")

    # 2. Reproducibilidad: misma semilla + mismas acciones -> misma trayectoria
    def trayectoria(seed, n=200):
        env = PastoreoEnv(n_ovejas=10)
        obs, _ = env.reset(seed=seed)
        acciones = np.random.default_rng(123).integers(9, size=n)
        obs_list = [obs]
        for a in acciones:
            obs, *_ = env.step(a)
            obs_list.append(obs)
        return np.array(obs_list)

    assert np.array_equal(trayectoria(7), trayectoria(7)), "La misma semilla dio trayectorias distintas"
    assert not np.array_equal(trayectoria(7), trayectoria(8)), "Semillas distintas dieron lo mismo"
    print("Reproducibilidad: OK")

    # 3. Velocidad y comportamiento con acciones al azar
    for n in (1, 3, 10, 30):
        env = PastoreoEnv(n_ovejas=n, max_pasos=2000)
        env.reset(seed=0)
        t0, pasos, episodios, exitos = time.time(), 0, 0, 0
        while time.time() - t0 < 3.0:
            _, _, term, trunc, info = env.step(env.action_space.sample())
            pasos += 1
            if term or trunc:
                episodios += 1
                exitos += int(info["is_success"])
                env.reset()
        print(f"N={n:2d}: {pasos / (time.time() - t0):6.0f} pasos/s | al azar: {exitos}/{episodios} episodios exitosos")
