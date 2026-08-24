"""Cargar un run entrenado y evaluarlo: δ(a, Ω_m0) y métricas de error.

No hardcodea nada del run — todo lo necesario para reconstruir la solución
(tipo de condición, si es bundle o no) se lee de `config.json`, que guardó
`training.py`. Así `validate.py` y `benchmark.py` sirven para cualquier run,
en vez de apuntar a una carpeta fija.
"""

import json
from pathlib import Path

import numpy as np
import torch
from neurodiffeq.conditions import IVP, BundleIVP
from neurodiffeq.solvers import BundleSolution1D, Solution1D

from cosmopinn.equation import N0_ABS, NP0, X_INIT, XP_INIT, to_physical


def load_run(run_dir):
    """Devuelve `(solution, config)` para un run guardado por `training.py`."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)

    nets = torch.load(run_dir / "nets.pth", map_location="cpu", weights_only=False)

    if config["kind"] == "bundle":
        condition = [BundleIVP(NP0, X_INIT), BundleIVP(NP0, XP_INIT)]
        solution = BundleSolution1D(nets, condition)
    else:
        condition = [IVP(NP0, X_INIT), IVP(NP0, XP_INIT)]
        solution = Solution1D(nets, condition)

    return solution, config


def delta_pinn(solution, config, a, Om_m0=None):
    """Evalúa la red en el factor de escala `a`, devuelve (δ, dδ/da).

    `Om_m0` es obligatorio para un run bundle, e ignorado para uno single
    (el run ya está entrenado para su propio Ω_m0 fijo, en `config["Om_m0"]`).
    """
    a = np.asarray(a, dtype=float)
    N_p = np.log(a) / N0_ABS

    if config["kind"] == "bundle":
        if Om_m0 is None:
            raise ValueError("Om_m0 es obligatorio para evaluar un run bundle")
        Om_vec = np.full_like(N_p, Om_m0)
        x, y = solution(N_p, Om_vec, to_numpy=True)
    else:
        x, y = solution(N_p, to_numpy=True)

    return to_physical(x, y, a)


def relative_error_percent(pred, true):
    """Error relativo punto a punto, en porcentaje."""
    return np.abs(pred - true) / np.abs(true) * 100


def error_summary(err):
    """Mediana / percentil 95 / máximo de un array de errores."""
    return {
        "median": float(np.median(err)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
    }
