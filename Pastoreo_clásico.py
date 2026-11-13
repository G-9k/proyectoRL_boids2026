import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

# =====================================================================
# PARÁMETROS EXACTOS DE LA TABLA 1 DE STRÖMBOM ET AL. (2014)
# =====================================================================
L = 150.0               # Lado del campo cuadrado inicial L x L (150 m)
N = 30                  # Número de ovejas (en el paper evalúan N = 1..201)
n_topo = 20             # Vecinos más cercanos (n > 0.53*N asegura éxito en modo global)

# Radios de interacción (en metros)
r_a = 2.0               # Radio de repulsión oveja-oveja (r_a = 2 m)
r_s = 65.0              # Radio de detección del pastor (r_s = 65 m)

# Pesos relativos de los vectores unitarios (Tabla 1: rho_a > c > rho_s > h)
c_weight = 1.05         # Atracción a los n vecinos más cercanos (c = 1.05)
rho_a = 2.0             # Repulsión de otras ovejas (\rho_a = 2.0)
rho_s = 1.0             # Repulsión del pastor (\rho_s = 1.0)
h_inertia = 0.5         # Inercia de la dirección previa (h = 0.5)
e_noise = 0.3           # Fuerza relativa del ruido angular (e = 0.3, tanto ovejas como pastor)

# Desplazamientos por paso de tiempo (\Delta t = 1) y pastoreo
delta_sheep = 1.0       # Desplazamiento de la oveja por paso (\delta = 1 m ts^-1)
delta_shepherd = 1.5    # Desplazamiento del pastor por paso (\delta_s = 1.5 m ts^-1)
p_grazing_move = 0.05   # Probabilidad de moverse por paso mientras pastorea (p = 0.05)

# Parámetros geométricos de la heurística del pastor (Ecuaciones de Strömbom)
p_c = r_a               # Posición de recolección: r_a metros detrás de la oveja más lejana
# Nota: f(N) = r_a * N^(2/3) y p_d = r_a * sqrt(N) se calculan dinámicamente
# según el tamaño del grupo percibido (N en modo global, o tamaño del subgrupo en modo local).

# Parámetros para el Pastor Local (Sección 4.2 y Tabla 1)
n_s = 20                # Número de agentes más cercanos sobre los que opera el pastor local
beta_blind = np.pi / 2  # Ángulo ciego detrás del pastor (\beta = \pi / 2)

# Posición de la Meta (Esquina inferior izquierda / Origen, Fig. 2 y Sección 4.1)
TARGET = np.array([0.0, 0.0])
TARGET_RADIUS = 10.0    # A menos de 10 unidades de (0,0) la tarea se completa (Fig. 2)
WORLD_MIN = -30.0
WORLD_MAX = 180.0


class CanonicalStrombomEnv:
    def __init__(self, local_shepherd=False):
        self.rng = np.random.default_rng()
        self.local_shepherd = local_shepherd
        self.reset()

    def reset(self):
        # Sección 4.1: N agentes posicionados aleatoriamente en el cuarto superior derecho
        # de un cuadrado L x L ([L/2, L] x [L/2, L], es decir, [75, 150] x [75, 150])
        self.A = self.rng.uniform(L / 2.0, L, size=(N, 2))

        # Direcciones iniciales unitarias \hat{H}_i
        angles = self.rng.uniform(0, 2 * np.pi, size=N)
        self.H = np.column_stack((np.cos(angles), np.sin(angles)))

        # Sección 4.1: Pastor liberado en el cuarto inferior izquierdo ([0, L/2] x [0, L/2])
        self.S = self.rng.uniform(0.0, L / 2.0, size=2)

        # Controles interactivos
        self.keys_pressed = set()
        self.auto_mode = True
        self.task_completed = False

        # Estado de diagnóstico y lo que "ve" el perro
        self.perceived_indices = np.arange(N)
        self.cm = np.mean(self.A, axis=0)
        self.f_group = r_a * (N ** (2.0 / 3.0))
        self.heuristic_target = self.S.copy()
        self.mode_str = "COLLECTING (Recolección)"
        self.far_sheep_idx = 0

    @staticmethod
    def normalize(V):
        r"""Operador de vector unitario \hat{v} = v / ||v|| (devuelve 0 si ||v|| == 0)."""
        if V.ndim == 1:
            norm = np.linalg.norm(V)
            return V / norm if norm > 1e-12 else np.zeros_like(V)
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        return np.divide(V, norms, out=np.zeros_like(V), where=norms > 1e-12)

    def get_shepherd_perceived_flock(self):
        r"""
        Determina qué ovejas ve y considera el perro pastor:
        - Modo Global (Sección 4.1): Considera las N ovejas del rebaño (GCM).
        - Modo Local (Sección 4.2): Considera los min(N_t, n_s) agentes más cercanos
          que aún no fueron entregados y que están dentro de su rango visual
          (excluyendo la zona ciega de ángulo \beta detrás del pastor respecto al LCM_t).
        """
        if not self.local_shepherd:
            return np.arange(N)

        # Agentes que aún no han llegado a la meta (N_t)
        dists_to_target = np.linalg.norm(self.A - TARGET, axis=1)
        not_delivered = np.where(dists_to_target > TARGET_RADIUS)[0]
        if len(not_delivered) == 0:
            return np.arange(N)

        # Paso 1: Detectar el LCM_t inicial de los k = min(N_t, n_s) más cercanos al pastor
        k = min(len(not_delivered), n_s)
        dists_to_S = np.linalg.norm(self.A[not_delivered] - self.S, axis=1)
        nearest_sub = not_delivered[np.argsort(dists_to_S)[:k]]
        lcm_t = np.mean(self.A[nearest_sub], axis=0)

        # Paso 2: Filtrar aquellos en el ángulo ciego \beta detrás del pastor respecto a LCM_t
        heading_to_lcm = self.normalize(lcm_t - self.S)
        if np.linalg.norm(heading_to_lcm) > 1e-12:
            vecs_to_agents = self.normalize(self.A[not_delivered] - self.S)
            cos_angles = np.dot(vecs_to_agents, heading_to_lcm)
            # La zona ciega trasera tiene apertura total \beta (\beta/2 a cada lado del vector opuesto)
            in_visual_range = cos_angles >= -np.cos(beta_blind / 2.0)
            visible_candidates = not_delivered[in_visual_range]
            if len(visible_candidates) > 0:
                not_delivered = visible_candidates

        k = min(len(not_delivered), n_s)
        dists_to_S = np.linalg.norm(self.A[not_delivered] - self.S, axis=1)
        perceived = not_delivered[np.argsort(dists_to_S)[:k]]
        return perceived

    def compute_strombom_heuristic(self):
        r"""Calcula el centro de masa percibido (GCM o LCM_t) y el punto objetivo P_d o P_c."""
        self.perceived_indices = self.get_shepherd_perceived_flock()
        perceived_A = self.A[self.perceived_indices]
        n_group = len(self.perceived_indices)

        # Centro de masa (GCM en modo global, LCM_t en modo local)
        self.cm = np.mean(perceived_A, axis=0)

        # Radio de cohesión f(N) y distancia de arreo P_d según el grupo percibido
        self.f_group = r_a * (n_group ** (2.0 / 3.0))
        p_d_group = r_a * np.sqrt(n_group)

        # Distancia de las ovejas percibidas al centro de masa
        dists_to_cm = np.linalg.norm(perceived_A - self.cm, axis=1)
        local_far_idx = int(np.argmax(dists_to_cm))
        self.far_sheep_idx = int(self.perceived_indices[local_far_idx])
        d_max = dists_to_cm[local_far_idx]

        # Verificar si el GCM global ya está en la meta
        if np.linalg.norm(np.mean(self.A, axis=0) - TARGET) <= TARGET_RADIUS:
            self.task_completed = True
            self.mode_str = "COMPLETED (En Meta)"
            self.heuristic_target = self.S.copy()
            return self.heuristic_target

        self.task_completed = False

        # Regla de decisión del pastor (Figura 1b y Sección 4.1)
        if d_max <= self.f_group:
            self.mode_str = "DRIVING (Arreo)"
            dir_from_target = self.normalize(self.cm - TARGET)
            self.heuristic_target = self.cm + p_d_group * dir_from_target
        else:
            self.mode_str = "COLLECTING (Recolección)"
            A_far = self.A[self.far_sheep_idx]
            dir_from_cm = self.normalize(A_far - self.cm)
            self.heuristic_target = A_far + p_c * dir_from_cm

        return self.heuristic_target

    def step_shepherd(self):
        r"""Actualiza al pastor (con las flechas o siguiendo el algoritmo de Strömbom)."""
        P_target = self.compute_strombom_heuristic()

        if self.auto_mode:
            if self.task_completed:
                return

            # Regla de proximidad exacta de Strömbom (Fig. 1b y Sección 4.1):
            # "If the shepherd is within 3*r_a from any flocking agent, its speed delta_s is set to 0."
            min_dist_sheep = np.min(np.linalg.norm(self.A - self.S, axis=1))
            if min_dist_sheep < 3.0 * r_a:
                step_size = 0.0
                self.mode_str = "STOPPED (< 3 r_a)"
            else:
                step_size = delta_shepherd

            if step_size > 0.0:
                dir_to_target = self.normalize(P_target - self.S)
                angle_eps = self.rng.uniform(0, 2 * np.pi)
                eps_s = np.array([np.cos(angle_eps), np.sin(angle_eps)])

                # Sección 4.1 y Tabla 1: "The shepherd experiences the same noise as the agents e*\hat{\epsilon}" (e = 0.3)
                H_s = self.normalize(dir_to_target + e_noise * eps_s)
                self.S = self.S + step_size * H_s
        else:
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
        # Suma de versores (A_i - A_j)/||A_i - A_j|| y normalización a \hat{R}_i^a
        R_ia = self.normalize(np.sum(unit_diff * in_ra[:, :, np.newaxis], axis=1))

        # 3. Detección del pastor (||A_i - S|| <= r_s) y repulsión \hat{R}_i^s
        vec_from_S = self.A - self.S                                # (N, 2)
        dist_to_S = np.linalg.norm(vec_from_S, axis=1)              # (N,)
        threatened = dist_to_S <= r_s                               # (N,) booleano

        R_is = np.zeros_like(self.A)
        R_is[threatened] = self.normalize(vec_from_S[threatened])

        # 4. Atracción al Centro de Masa Local (LCM_i) de sus n vecinos más cercanos: \hat{C}_i
        # Activa únicamente cuando la oveja está dentro de r_s del pastor
        nearest_idx = np.argsort(dist_matrix, axis=1)[:, :n_topo]   # (N, n_topo)
        LCM = np.mean(self.A[nearest_idx], axis=1)                  # (N, 2)

        C_i = np.zeros_like(self.A)
        C_i[threatened] = self.normalize(LCM[threatened] - self.A[threatened])

        # 5. Ruido angular unitario \hat{\varepsilon}_i
        angles_eps = self.rng.uniform(0, 2 * np.pi, size=N)
        eps_i = np.column_stack((np.cos(angles_eps), np.sin(angles_eps)))

        # 6. Regla de activación del movimiento (Pastoreo vs Interacción):
        # Si está a más de r_s del pastor y sin vecinos dentro de r_a, pastorea
        # (permanece estacionaria salvo con probabilidad p = 0.05 por paso).
        active_interaction = threatened | has_ra_neighbor
        random_jiggle = (~active_interaction) & (self.rng.random(N) < p_grazing_move)
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

    def step(self):
        # Evaluamos las ovejas y el pastor y actualizamos el diagnóstico para que la gráfica esté sincronizada
        self.step_shepherd()
        self.step_sheep()
        self.compute_strombom_heuristic()


# =====================================================================
# VISUALIZACIÓN INTERACTIVA
# =====================================================================
def main():
    sim = CanonicalStrombomEnv(local_shepherd=False)

    # Desactivar atajos de teclado de Matplotlib de forma segura en cualquier versión
    for key in ('keymap.save', 'keymap.fullscreen', 'keymap.pan', 'keymap.zoom', 'keymap.xscale', 'keymap.yscale'):
        if key in plt.rcParams:
            plt.rcParams[key] = []

    fig, ax = plt.subplots(figsize=(9, 9))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Modelo Perrito")
    ax.set_xlim(WORLD_MIN, WORLD_MAX)
    ax.set_ylim(WORLD_MIN, WORLD_MAX)
    ax.set_aspect('equal')
    ax.set_facecolor('#f4f6f0')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Dibujar el borde del campo inicial L x L (150 x 150 m)
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color='#7f8c8d', linestyle='--', linewidth=1.0, alpha=0.6, label=f'Campo inicial ({int(L)}x{int(L)} m)')

    # Meta en el origen (0, 0)
    target_circle = Circle(TARGET, TARGET_RADIUS, color='#2ecc71', alpha=0.35, label='Meta (Origen 0,0)')
    ax.add_patch(target_circle)
    ax.plot(TARGET[0], TARGET[1], 'g+', markersize=10, markeredgewidth=2)

    # Radio de amenaza del pastor (r_s = 65 m)
    rs_circle = Circle(sim.S, r_s, color='#e74c3c', fill=False, linestyle=':', linewidth=1.5, alpha=0.7, label=f'Detección oveja ($r_s={int(r_s)}$ m)')
    ax.add_patch(rs_circle)

    # Radio de freno del pastor (3 * r_a = 6 m)
    stop_circle = Circle(sim.S, 3.0 * r_a, color='#c0392b', fill=True, alpha=0.15, label=f'Zona freno pastor ($3r_a={3*r_a:.0f}$ m)')
    ax.add_patch(stop_circle)

    # Umbral de cohesión f(N) alrededor del centro de masa
    fn_circle = Circle(sim.cm, sim.f_group, color='#3498db', fill=False, linestyle='--', linewidth=1.2, alpha=0.7, label=f'Cohesión $f(N)={sim.f_group:.1f}$ m')
    ax.add_patch(fn_circle)

    # Gráficos dinámicos
    sheep_quiver = ax.quiver(
        sim.A[:, 0], sim.A[:, 1], sim.H[:, 0], sim.H[:, 1],
        color='#2c3e50', scale=30, width=0.0045, headwidth=4, label='Ovejas ($A_i$)'
    )
    perceived_plot, = ax.plot([], [], 'o', color='#2980b9', markersize=6, fillstyle='none', alpha=0.6, label='Vistas por el pastor')
    far_sheep_plot, = ax.plot([], [], 'o', color='#e67e22', markersize=10, fillstyle='none', markeredgewidth=2, label='Oveja más lejana ($A_f$)')
    cm_plot, = ax.plot([], [], 'bx', markersize=8, markeredgewidth=2, label='Centro de Masa (GCM/LCM)')
    heuristic_plot, = ax.plot([], [], 'P', color='#f39c12', markersize=10, markeredgecolor='black', label='Punto objetivo ($P_d / P_c$)')
    shepherd_plot, = ax.plot([], [], 'ro', markersize=10, markeredgecolor='black', label='Pastor ($S$)')

    status_text = ax.text(
        0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
        fontsize=9.0, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
    )

    ax.legend(loc='lower right', fontsize=8.0, framealpha=0.9)
    ax.set_title("ESPACIO: Auto/Manual | L: Pastor Global/Local | R: Reiniciar | Flechas: Mover", fontsize=10.0, fontweight='bold')

    def on_key_press(event):
        if event.key in ['up', 'down', 'left', 'right']:
            sim.keys_pressed.add(event.key)
        elif event.key == ' ':
            sim.auto_mode = not sim.auto_mode
            sim.keys_pressed.clear()
        elif event.key in ['l', 'L']:
            sim.local_shepherd = not sim.local_shepherd
            sim.compute_strombom_heuristic()
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

        cm_plot.set_data([sim.cm[0]], [sim.cm[1]])
        fn_circle.center = (sim.cm[0], sim.cm[1])
        fn_circle.set_radius(sim.f_group)

        perceived_A = sim.A[sim.perceived_indices]
        perceived_plot.set_data(perceived_A[:, 0], perceived_A[:, 1])

        far_pos = sim.A[sim.far_sheep_idx]
        far_sheep_plot.set_data([far_pos[0]], [far_pos[1]])
        heuristic_plot.set_data([sim.heuristic_target[0]], [sim.heuristic_target[1]])

        d_max = np.linalg.norm(far_pos - sim.cm)
        dist_goal = np.linalg.norm(np.mean(sim.A, axis=0) - TARGET)
        ctrl_label = "AUTO (Strömbom 2014)" if sim.auto_mode else "MANUAL (Flechas)"
        vision_label = f"LOCAL (n_s={n_s}, beta=90°)" if sim.local_shepherd else f"GLOBAL (N={N})"

        status_text.set_text(
            f"Control Pastor : {ctrl_label}\n"
            f"Visión Pastor  : {vision_label}\n"
            f"Fase Strömbom  : {sim.mode_str}\n"
            f"Dispersión máx : {d_max:5.1f} / f={sim.f_group:.1f} m\n"
            f"Dist. GCM Meta : {dist_goal:5.1f} m"
        )

        return (sheep_quiver, perceived_plot, shepherd_plot, rs_circle, stop_circle,
                cm_plot, fn_circle, far_sheep_plot, heuristic_plot, status_text)

    ani = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
