import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# =====================================================================
# PARÁMETROS EXACTOS DE LA TABLA 1 DE STRÖMBOM ET AL. (2014)
# =====================================================================
L = 150.0               # Lado del campo cuadrado inicial L x L (150 m)
N = 30                  # Número de ovejas (en el paper evalúan N = 1..201)
n_topo = 20             # Vecinos más cercanos (n > 0.53*N asegura cohesión del rebaño)

# Radios de interacción (en metros)
r_a = 2.0               # Radio de repulsión oveja-oveja (r_a = 2 m)
r_s = 65.0              # Radio de detección del pastor (r_s = 65 m)

# Pesos relativos de los vectores unitarios de las ovejas (Tabla 1: rho_a > c > rho_s > h)
c_weight = 1.05         # Atracción a los n vecinos más cercanos (c = 1.05)
rho_a = 2.0             # Repulsión de otras ovejas (\rho_a = 2.0)
rho_s = 1.0             # Repulsión del pastor (\rho_s = 1.0)
h_inertia = 0.5         # Inercia de la dirección previa (h = 0.5)
e_noise = 0.3           # Fuerza relativa del ruido angular (e = 0.3)

# Desplazamientos por paso de tiempo (\Delta t = 1) y pastoreo
delta_sheep = 1.0       # Desplazamiento de la oveja por paso (\delta = 1 m ts^-1)
delta_shepherd = 1.5    # Desplazamiento máximo del pastor por paso (\delta_s = 1.5 m ts^-1)
p_grazing_move = 0.05   # Probabilidad de moverse por paso mientras pastorea (p = 0.05)

# Umbral geométrico de cohesión de referencia f(N) = r_a * N^(2/3)
f_N = r_a * (N ** (2.0 / 3.0))

# Posición de la Meta (Esquina inferior izquierda / Origen, Fig. 2 y Sección 4.1)
TARGET = np.array([0.0, 0.0])
TARGET_RADIUS = 10.0    # A menos de 10 unidades de (0,0) la tarea se completa (Fig. 2)
WORLD_MIN = -30.0
WORLD_MAX = 180.0


class ManualStrombomEnv:
    r"""
    Entorno con la física exacta de las ovejas de Strömbom et al. (2014),
    sin las reglas heurísticas del pastor (para usar con flechitas
    o para conectar con DQN).
    """
    # Espacio de 9 acciones discretas (8 rumbos + 0: quedarse quieto) pensado para DQN
    ACTION_VECTORS = {
        0: np.array([0.0, 0.0]),
        1: np.array([0.0, 1.0]),                        # Arriba (N)
        2: np.array([0.0, -1.0]),                       # Abajo (S)
        3: np.array([-1.0, 0.0]),                       # Izquierda (O)
        4: np.array([1.0, 0.0]),                        # Derecha (E)
        5: np.array([-1.0, 1.0]) / np.sqrt(2.0),        # Noroeste (NO)
        6: np.array([1.0, 1.0]) / np.sqrt(2.0),         # Noreste (NE)
        7: np.array([-1.0, -1.0]) / np.sqrt(2.0),       # Sudoeste (SO)
        8: np.array([1.0, -1.0]) / np.sqrt(2.0),        # Sudeste (SE)
    }

    def __init__(self, n_sheep=N, target_radius=TARGET_RADIUS, bounds=None, seed=None):
        r"""
        n_sheep: cantidad de ovejas (por defecto N = 30, como en el paper).
        target_radius: el episodio se completa cuando el GCM está a menos de esto de la meta.
        bounds: None = mundo sin bordes (original). Una tupla (min, max) = cerco cuadrado:
                ovejas y pastor no pueden salir de [min, max] x [min, max].
        seed: semilla del generador aleatorio (None = no reproducible).
        """
        self.N = int(n_sheep)
        self.f_N = r_a * (self.N ** (2.0 / 3.0))
        self.target_radius = float(target_radius)
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        self.keys_pressed = set()
        self.reset()

    def reset(self):
        # Sección 4.1: N agentes posicionados aleatoriamente en el cuarto superior derecho
        # de un cuadrado L x L ([L/2, L] x [L/2, L], es decir, [75, 150] x [75, 150])
        self.A = self.rng.uniform(L / 2.0, L, size=(self.N, 2))

        # Direcciones iniciales unitarias \hat{H}_i
        angles = self.rng.uniform(0, 2 * np.pi, size=self.N)
        self.H = np.column_stack((np.cos(angles), np.sin(angles)))

        # Sección 4.1: Pastor liberado en el cuarto inferior izquierdo ([0, L/2] x [0, L/2])
        self.S = self.rng.uniform(0.0, L / 2.0, size=2)

        self.task_completed = False
        self.update_metrics()
        return self.get_dqn_state()

    @staticmethod
    def normalize(V):
        r"""Operador de vector unitario \hat{v} = v / ||v|| (devuelve 0 si ||v|| == 0)."""
        if V.ndim == 1:
            norm = np.linalg.norm(V)
            return V / norm if norm > 1e-12 else np.zeros_like(V)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return np.divide(V, norms, out=np.zeros_like(V), where=norms > 1e-12)

    def update_metrics(self):
        r"""Calcula las métricas espaciales del rebaño (GCM, dispersión, oveja más lejana)."""
        self.gcm = np.mean(self.A, axis=0)
        dists_to_gcm = np.linalg.norm(self.A - self.gcm, axis=1)
        self.far_sheep_idx = int(np.argmax(dists_to_gcm))
        self.d_max = float(dists_to_gcm[self.far_sheep_idx])
        self.dist_goal = float(np.linalg.norm(self.gcm - TARGET))
        self.min_dist_sheep = float(np.min(np.linalg.norm(self.A - self.S, axis=1)))
        if self.dist_goal <= self.target_radius:
            self.task_completed = True

    def _apply_bounds(self):
        r"""Cerco opcional: mantiene ovejas y pastor dentro de [min, max] x [min, max]."""
        if self.bounds is None:
            return
        lo, hi = self.bounds
        np.clip(self.A, lo, hi, out=self.A)
        self.S = np.clip(self.S, lo, hi)

    def get_dqn_state(self):
        r"""
        Vector de observación invariante a permutaciones ideal para alimentar una red DQN (8 valores):
        [ (GCM - S)/L, (A_far - GCM)/f(N), (TARGET - GCM)/L, d_max/f(N), min_dist_sheep/(3*r_a) ]
        """
        A_far = self.A[self.far_sheep_idx]
        return np.concatenate([
            (self.gcm - self.S) / L,
            (A_far - self.gcm) / self.f_N,
            (TARGET - self.gcm) / L,
            [self.d_max / self.f_N, self.min_dist_sheep / (3.0 * r_a)]
        ])

    def step_shepherd_from_keys(self):
        r"""Mueve al pastor únicamente según las teclas de dirección presionadas."""
        move = np.zeros(2)
        if 'up' in self.keys_pressed:
            move[1] += 1.0
        if 'down' in self.keys_pressed:
            move[1] -= 1.0
        if 'left' in self.keys_pressed:
            move[0] -= 1.0
        if 'right' in self.keys_pressed:
            move[0] += 1.0

        if np.linalg.norm(move) > 0:
            self.S = self.S + delta_shepherd * self.normalize(move)

    def step_shepherd_action(self, action_idx):
        r"""Mueve al pastor usando una acción discreta (0..8) para entrenamiento con DQN."""
        direction = self.ACTION_VECTORS.get(int(action_idx), np.zeros(2))
        self.S = self.S + delta_shepherd * direction

    def step_sheep(self):
        r"""
        Implementación exacta de las ecuaciones (4.1), (4.2) y (4.3) de las ovejas
        según Strömbom et al. (2014).
        """
        # 1. Distancias entre todas las ovejas: diff[i, j] = A_i - A_j
        diff = self.A[:, np.newaxis, :] - self.A[np.newaxis, :, :]  # (N, N, 2)
        dist_matrix = np.linalg.norm(diff, axis=2)                  # (N, N)
        np.fill_diagonal(dist_matrix, np.inf)

        # 2. Vector unitario de repulsión local oveja-oveja: \hat{R}_i^a (Ecuación 4.1)
        in_ra = dist_matrix < r_a                                   # (N, N)
        has_ra_neighbor = np.any(in_ra, axis=1)                     # (N,) booleano

        unit_diff = np.divide(
            diff, dist_matrix[:, :, np.newaxis],
            out=np.zeros_like(diff),
            where=(dist_matrix[:, :, np.newaxis] > 1e-12) & (dist_matrix[:, :, np.newaxis] < np.inf)
        )
        R_ia = self.normalize(np.sum(unit_diff * in_ra[:, :, np.newaxis], axis=1))

        # 3. Detección del pastor (||A_i - S|| <= r_s) y repulsión \hat{R}_i^s
        vec_from_S = self.A - self.S                                # (N, 2)
        dist_to_S = np.linalg.norm(vec_from_S, axis=1)              # (N,)
        threatened = dist_to_S <= r_s                               # (N,) booleano

        R_is = np.zeros_like(self.A)
        R_is[threatened] = self.normalize(vec_from_S[threatened])

        # 4. Atracción al Centro de Masa Local (LCM_i) de sus n vecinos más cercanos: \hat{C}_i
        # Activa únicamente cuando la oveja está dentro de r_s del pastor
        # Con N <= n_topo se usan las N-1 restantes (sin contarse a sí misma); con N = 1 no hay atracción.
        C_i = np.zeros_like(self.A)
        k_neighbors = min(n_topo, self.N - 1)
        if k_neighbors > 0:
            nearest_idx = np.argsort(dist_matrix, axis=1)[:, :k_neighbors]  # (N, k)
            LCM = np.mean(self.A[nearest_idx], axis=1)                       # (N, 2)
            C_i[threatened] = self.normalize(LCM[threatened] - self.A[threatened])

        # 5. Ruido angular unitario \hat{\varepsilon}_i
        angles_eps = self.rng.uniform(0, 2 * np.pi, size=self.N)
        eps_i = np.column_stack((np.cos(angles_eps), np.sin(angles_eps)))

        # 6. Regla de activación del movimiento (Pastoreo vs Interacción):
        active_interaction = threatened | has_ra_neighbor
        random_jiggle = (~active_interaction) & (self.rng.random(self.N) < p_grazing_move)
        moving_mask = active_interaction | random_jiggle

        # 7. Nueva dirección \hat{H}_i' (Ecuación 4.2)
        H_new_raw = (
            h_inertia * self.H +
            c_weight * C_i +
            rho_a * R_ia +
            rho_s * R_is +
            e_noise * eps_i
        )
        H_updated = self.normalize(H_new_raw)
        self.H[moving_mask] = H_updated[moving_mask]

        # 8. Nueva posición A_i' = A_i + \delta \hat{H}_i' (Ecuación 4.3)
        self.A[moving_mask] += delta_sheep * self.H[moving_mask]

    def step(self, action_idx=None):
        r"""
        Ejecuta un paso de simulación:
        - Si action_idx es None, mueve al pastor con las flechas del teclado.
        - Si action_idx es un entero (0..8), aplica esa acción discreta (útil para DQN).
        """
        if action_idx is None:
            self.step_shepherd_from_keys()
        else:
            self.step_shepherd_action(action_idx)

        self.step_sheep()
        self._apply_bounds()
        self.update_metrics()


# =====================================================================
# VISUALIZACIÓN INTERACTIVA (CONTROL CON FLECHAS)
# =====================================================================
def main():
    sim = ManualStrombomEnv()

    # Desactivar atajos de teclado de Matplotlib de forma segura en cualquier versión
    for key in ('keymap.save', 'keymap.fullscreen', 'keymap.pan', 'keymap.zoom', 'keymap.xscale', 'keymap.yscale'):
        if key in plt.rcParams:
            plt.rcParams[key] = []

    fig, ax = plt.subplots(figsize=(9, 9))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Entorno Strömbom (2014) - Control Manual del Pastor / Base DQN")
    ax.set_xlim(WORLD_MIN, WORLD_MAX)
    ax.set_ylim(WORLD_MIN, WORLD_MAX)
    ax.set_aspect('equal')
    ax.set_facecolor('#f4f6f0')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Borde del campo inicial L x L (150 x 150 m)
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color='#7f8c8d', linestyle='--', linewidth=1.0, alpha=0.6, label=f'Campo inicial ({int(L)}x{int(L)} m)')

    # Meta en el origen (0, 0)
    target_circle = Circle(TARGET, sim.target_radius, color='#2ecc71', alpha=0.35, label='Meta (Origen 0,0)')
    ax.add_patch(target_circle)
    ax.plot(TARGET[0], TARGET[1], 'g+', markersize=10, markeredgewidth=2)

    # Radio de amenaza del pastor (r_s = 65 m)
    rs_circle = Circle(sim.S, r_s, color='#e74c3c', fill=False, linestyle=':', linewidth=1.5, alpha=0.7, label=f'Detección oveja ($r_s={int(r_s)}$ m)')
    ax.add_patch(rs_circle)

    # Radio de proximidad crítica (3 * r_a = 6 m)
    stop_circle = Circle(sim.S, 3.0 * r_a, color='#c0392b', fill=True, alpha=0.15, label=f'Distancia crítica ($3r_a={3*r_a:.0f}$ m)')
    ax.add_patch(stop_circle)

    # Umbral de cohesión f(N) alrededor del GCM
    fn_circle = Circle(sim.gcm, sim.f_N, color='#3498db', fill=False, linestyle='--', linewidth=1.2, alpha=0.7, label=f'Cohesión $f(N)={sim.f_N:.1f}$ m')
    ax.add_patch(fn_circle)

    # Gráficos dinámicos
    sheep_quiver = ax.quiver(
        sim.A[:, 0], sim.A[:, 1], sim.H[:, 0], sim.H[:, 1],
        color='#2c3e50', scale=30, width=0.0045, headwidth=4, label='Ovejas ($A_i$)'
    )
    far_sheep_plot, = ax.plot([], [], 'o', color='#e67e22', markersize=10, fillstyle='none', markeredgewidth=2, label='Oveja más lejana ($A_f$)')
    gcm_plot, = ax.plot([], [], 'bx', markersize=8, markeredgewidth=2, label='Centro de Masa (GCM)')
    shepherd_plot, = ax.plot([], [], 'ro', markersize=10, markeredgecolor='black', label='Pastor ($S$)')

    status_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
        fontsize=9.0, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
    )

    ax.legend(loc='lower right', fontsize=8.0, framealpha=0.9)
    ax.set_title("Flechas: Mover Pastor Manualmente | R: Reiniciar Episodio", fontsize=10.5, fontweight='bold')

    def on_key_press(event):
        if event.key in ['up', 'down', 'left', 'right']:
            sim.keys_pressed.add(event.key)
        elif event.key in ['r', 'R']:
            sim.reset()

    def on_key_release(event):
        if event.key in sim.keys_pressed:
            sim.keys_pressed.remove(event.key)

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('key_release_event', on_key_release)

    def update(frame):
        sim.step()

        sheep_quiver.set_offsets(sim.A)
        sheep_quiver.set_UVC(sim.H[:, 0], sim.H[:, 1])

        # Oscuras si sienten al pastor (<= r_s), grises si pastorean tranquilas (> r_s)
        dists_to_S = np.linalg.norm(sim.A - sim.S, axis=1)
        colors = np.where(dists_to_S <= r_s, '#2c3e50', '#95a5a6')
        sheep_quiver.set_color(colors)

        shepherd_plot.set_data([sim.S[0]], [sim.S[1]])
        rs_circle.center = (sim.S[0], sim.S[1])
        stop_circle.center = (sim.S[0], sim.S[1])

        gcm_plot.set_data([sim.gcm[0]], [sim.gcm[1]])
        fn_circle.center = (sim.gcm[0], sim.gcm[1])

        far_pos = sim.A[sim.far_sheep_idx]
        far_sheep_plot.set_data([far_pos[0]], [far_pos[1]])

        cohesive_str = "SÍ (Compacto)" if sim.d_max <= sim.f_N else "NO (Disperso)"
        goal_str = "¡COMPLETADO!" if sim.task_completed else f"{sim.dist_goal:5.1f} m"

        status_text.set_text(
            f"Control Pastor : MANUAL (Flechas)\n"
            f"Rebaño compacto: {cohesive_str}\n"
            f"Dispersión máx : {sim.d_max:5.1f} / f(N)={sim.f_N:.1f} m\n"
            f"Dist. GCM Meta : {goal_str}\n"
            f"Dist. oveja mín: {sim.min_dist_sheep:5.1f} m"
        )

        return (sheep_quiver, shepherd_plot, rs_circle, stop_circle,
                gcm_plot, fn_circle, far_sheep_plot, status_text)

    ani = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
