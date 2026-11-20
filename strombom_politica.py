"""
Pastor de Strömbom et al. (2014), modo global, como política que maneja al pastor dentro de PastoreoEnv.

Es la misma regla que compute_strombom_heuristic() en Pastoreo_clásico.py:
  - si la oveja más lejana al centro del rebaño (GCM) está a más de f(N): RECOLECTAR,
    ir a un punto r_a detrás de esa oveja (del lado opuesto al GCM);
  - si no: ARREAR, ir a un punto r_a*sqrt(N) detrás del GCM (del lado opuesto a la meta);
  - si está a menos de 3*r_a de alguna oveja: quedarse quieto;
  - al moverse, tiene el mismo ruido que las ovejas (e = 0.3).

Dos versiones:
  - StrombomContinuo: se mueve en cualquier dirección (como en el paper).
  - StrombomDiscreto: elige la más parecida de las 9 acciones del DQN.
"""

import numpy as np

from Pastoreo_flechitas import ManualStrombomEnv, TARGET, r_a, e_noise

normalize = ManualStrombomEnv.normalize

# Direcciones de las acciones 1..8 (la 0 es quedarse quieto), como matriz (8, 2)
_DIRECCIONES = np.array([ManualStrombomEnv.ACTION_VECTORS[a] for a in range(1, 9)])


def strombom_direccion(sim: ManualStrombomEnv, rng: np.random.Generator):
    """Devuelve (vector de dirección unitario o cero, modo) según la regla de Strömbom."""
    A, S, n = sim.A, sim.S, sim.N

    # Regla de proximidad: a menos de 3*r_a de cualquier oveja, el pastor se frena
    if np.min(np.linalg.norm(A - S, axis=1)) < 3.0 * r_a:
        return np.zeros(2), "quieto"

    gcm = np.mean(A, axis=0)
    dists_gcm = np.linalg.norm(A - gcm, axis=1)
    i_lejana = int(np.argmax(dists_gcm))
    f_n = r_a * n ** (2.0 / 3.0)

    if dists_gcm[i_lejana] <= f_n:
        modo = "arrear"
        objetivo = gcm + r_a * np.sqrt(n) * normalize(gcm - TARGET)
    else:
        modo = "recolectar"
        A_lejana = A[i_lejana]
        objetivo = A_lejana + r_a * normalize(A_lejana - gcm)

    angulo = rng.uniform(0.0, 2.0 * np.pi)
    ruido = np.array([np.cos(angulo), np.sin(angulo)])
    return normalize(normalize(objetivo - S) + e_noise * ruido), modo


def direccion_a_accion(direccion) -> int:
    """La acción (0..8) cuya dirección se parece más a `direccion` (0 si es el vector cero)."""
    if np.linalg.norm(direccion) < 1e-12:
        return 0
    return int(np.argmax(_DIRECCIONES @ direccion)) + 1


class StrombomContinuo:
    nombre = "strombom_continuo"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def paso(self, env):
        direccion, _ = strombom_direccion(env.sim, self.rng)
        return env.step_continuo(direccion)


class StrombomDiscreto:
    nombre = "strombom_9_acciones"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def paso(self, env):
        direccion, _ = strombom_direccion(env.sim, self.rng)
        return env.step(direccion_a_accion(direccion))
