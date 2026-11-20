"""
Paso 2: línea de base. Mide al pastor de Strömbom (continuo y con 9 acciones) en PastoreoEnv,
para 1, 3, 10 y 30 ovejas, con y sin cerco, sobre las mismas semillas.

Uso:  python medir_strombom.py [--episodios 100] [--max-pasos 6000]
Guarda los resultados en resultados/strombom_baseline.csv e imprime una tabla.
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from pastoreo_env import PastoreoEnv
from strombom_politica import StrombomContinuo, StrombomDiscreto

POLITICAS = {p.nombre: p for p in (StrombomContinuo, StrombomDiscreto)}
N_OVEJAS = [1, 3, 10, 30]
CERCO = [False, True]
SEED_BASE = 10_000  # semillas de los mapas: SEED_BASE + episodio (las mismas para todas las configuraciones)


def correr_config(args):
    politica_nombre, n, cerco, episodios, max_pasos = args
    env = PastoreoEnv(n_ovejas=n, cerco=cerco, max_pasos=max_pasos, recompensa="esparsa")
    politica = POLITICAS[politica_nombre](seed=0)

    exitos, pasos_exito = 0, []
    t0 = time.time()
    for ep in range(episodios):
        env.reset(seed=SEED_BASE + ep)
        terminado = truncado = False
        while not (terminado or truncado):
            _, _, terminado, truncado, info = politica.paso(env)
        if terminado:
            exitos += 1
            pasos_exito.append(info["pasos"])

    pasos = np.array(pasos_exito) if pasos_exito else np.array([np.nan])
    return {
        "politica": politica_nombre,
        "n_ovejas": n,
        "cerco": cerco,
        "episodios": episodios,
        "exito_pct": 100.0 * exitos / episodios,
        "pasos_media": float(np.nanmean(pasos)),
        "pasos_p50": float(np.nanpercentile(pasos, 50)),
        "pasos_p95": float(np.nanpercentile(pasos, 95)),
        "pasos_max": float(np.nanmax(pasos)),
        "segundos": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodios", type=int, default=100)
    parser.add_argument("--max-pasos", type=int, default=6000)
    args = parser.parse_args()

    configs = [
        (p, n, c, args.episodios, args.max_pasos)
        for p, n, c in product(POLITICAS, N_OVEJAS, CERCO)
    ]
    # Las más largas primero, para repartir mejor entre procesos
    configs.sort(key=lambda c: -c[1])

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(configs), os.cpu_count() or 1)) as pool:
        filas = list(pool.map(correr_config, configs))
    filas.sort(key=lambda f: (f["politica"], f["n_ovejas"], f["cerco"]))

    os.makedirs("resultados", exist_ok=True)
    ruta = os.path.join("resultados", "strombom_baseline.csv")
    with open(ruta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    print(f"\n{'política':<20} {'N':>3} {'cerco':>6} {'éxito':>7} {'pasos media':>12} {'p50':>6} {'p95':>6} {'máx':>6}")
    for f in filas:
        print(
            f"{f['politica']:<20} {f['n_ovejas']:>3} {str(f['cerco']):>6} {f['exito_pct']:>6.0f}% "
            f"{f['pasos_media']:>12.0f} {f['pasos_p50']:>6.0f} {f['pasos_p95']:>6.0f} {f['pasos_max']:>6.0f}"
        )
    print(f"\nTotal: {time.time() - t0:.0f} s. Guardado en {ruta}")


if __name__ == "__main__":
    main()
