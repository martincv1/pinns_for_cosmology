"""Arma la red, el optimizador y el solver de neurodiffeq, entrena, y guarda el run.

Dos funciones, una por script (`scripts/train_single.py`, `scripts/train_bundle.py`):

- `train_single`: Ω_m0 fijo. Usa `Solver1D` + `IVP`.
- `train_bundle`: Ω_m0 variable en un rango. Usa `BundleSolver1D` + `BundleIVP`,
  con el truco de neurodiffeq de tomar el producto de dos generadores (`^`)
  para samplear el dominio (N', Ω_m0) conjunto.

Ninguna hace nada específico de ΛCDM: la física está toda en `equation.py`.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from neurodiffeq.conditions import IVP, BundleIVP
from neurodiffeq.generators import Generator1D
from neurodiffeq.networks import FCNN
from neurodiffeq.solvers import BundleSolver1D, Solver1D

from cosmopinn.equation import NF, NP0, NPF, X_INIT, XP_INIT, residual


def mse_loss(res, x, t):
    """Pérdida estándar de una PINN: el residuo de la ODE debería ser cero en
    todo el dominio, así que se minimiza su cuadrado medio. No hay datos de
    entrenamiento — la ecuación misma es la función de costo."""
    return (res**2).mean()


def _save_run(out_dir, solver, config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loss = np.array(solver.metrics_history["train_loss"])
    np.save(out_dir / "loss.npy", loss)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(loss)
    plt.yscale("log")
    plt.xlabel("iteración")
    plt.ylabel("loss (train)")
    plt.title("Pérdida durante el entrenamiento")
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=150)
    plt.close()

    best_nets = solver._get_internal_variables()["best_nets"]
    torch.save(best_nets, out_dir / "nets.pth")

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def train_single(
    Om_m0,
    out_dir,
    hidden_units=(32, 32),
    learning_rate=1e-3,
    batch_size=32,
    iterations=5000,
    seed=42,
):
    """Entrena la red no-bundle: aprende δ(N') para un Ω_m0 fijo."""
    torch.manual_seed(seed)

    nets = [FCNN(n_input_units=1, hidden_units=hidden_units) for _ in range(2)]
    optimizer = torch.optim.Adam(set(p for net in nets for p in net.parameters()), lr=learning_rate)

    condition = [IVP(NP0, X_INIT), IVP(NP0, XP_INIT)]
    train_gen = Generator1D(batch_size, NP0, NPF)
    valid_gen = Generator1D(batch_size, NP0, NPF)

    def ode_system(x, x_prime, N_p):
        return residual(x, x_prime, N_p, Om_m0)

    solver = Solver1D(
        ode_system=ode_system,
        conditions=condition,
        t_min=NP0,
        t_max=NPF,
        nets=nets,
        optimizer=optimizer,
        train_generator=train_gen,
        valid_generator=valid_gen,
        loss_fn=mse_loss,
    )
    t0 = time.perf_counter()
    solver.fit(iterations)
    train_seconds = time.perf_counter() - t0

    config = {
        "kind": "single",
        "Om_m0": Om_m0,
        "hidden_units": list(hidden_units),
        "activation": "Tanh",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "iterations": iterations,
        "seed": seed,
        "N0": NP0,
        "Nf": NF,
        "train_seconds": train_seconds,
    }
    _save_run(out_dir, solver, config)
    return solver, config


def train_bundle(
    Om_m0_min,
    Om_m0_max,
    out_dir,
    hidden_units=(64, 64),
    learning_rate=1e-3,
    batch_size=32,
    iterations=20000,
    seed=42,
):
    """Entrena la red bundle: aprende δ(N', Ω_m0) en todo el rango [Om_m0_min, Om_m0_max]."""
    torch.manual_seed(seed)

    nets = [FCNN(n_input_units=2, hidden_units=hidden_units, actv=torch.nn.SiLU) for _ in range(2)]
    optimizer = torch.optim.Adam(set(p for net in nets for p in net.parameters()), lr=learning_rate)

    condition = [BundleIVP(NP0, X_INIT), BundleIVP(NP0, XP_INIT)]

    train_gen = Generator1D(batch_size, NP0, NPF) ^ Generator1D(batch_size, Om_m0_min, Om_m0_max)
    valid_gen = Generator1D(batch_size, NP0, NPF) ^ Generator1D(batch_size, Om_m0_min, Om_m0_max)

    solver = BundleSolver1D(
        ode_system=residual,
        nets=nets,
        conditions=condition,
        t_min=NP0,
        t_max=NPF,
        theta_min=Om_m0_min,
        theta_max=Om_m0_max,
        eq_param_index=(0,),
        optimizer=optimizer,
        train_generator=train_gen,
        valid_generator=valid_gen,
        loss_fn=mse_loss,
    )
    t0 = time.perf_counter()
    solver.fit(iterations)
    train_seconds = time.perf_counter() - t0

    config = {
        "kind": "bundle",
        "Om_m0_range": [Om_m0_min, Om_m0_max],
        "hidden_units": list(hidden_units),
        "activation": "SiLU",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "iterations": iterations,
        "seed": seed,
        "N0": NP0,
        "Nf": NF,
        "train_seconds": train_seconds,
    }
    _save_run(out_dir, solver, config)
    return solver, config
