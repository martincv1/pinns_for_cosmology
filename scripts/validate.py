"""Compara un run entrenado contra la solución de referencia (`scipy.solve_ivp`).

Genera tres figuras en `figures/` y una tabla de error en stdout:

- `comparison.png`: δ(a) y dδ/da, red vs referencia, para un Ω_m0.
- `error_curve.png`: error relativo (%) vs a, para ese mismo Ω_m0.
- `error_heatmap.png` (solo si el run es bundle): error relativo (%) en el
  plano (a, Ω_m0).

Uso:
    python scripts/validate.py --run models/single_om030
    python scripts/validate.py --run models/bundle_om_010_050
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cosmopinn.evaluate import delta_pinn, error_summary, load_run, relative_error_percent  # noqa: E402
from cosmopinn.reference import solve  # noqa: E402


def compare_one(solution, config, om_m0, figures_dir):
    a, delta_ref, ddelta_ref, delta_interp, ddelta_interp = solve(om_m0)
    a_test = np.logspace(np.log10(a.min()), np.log10(a.max()), 1000)

    delta_nn, ddelta_nn = delta_pinn(solution, config, a_test, om_m0)
    delta_ref_test = delta_interp(a_test)
    ddelta_ref_test = ddelta_interp(a_test)

    err_delta = relative_error_percent(delta_nn, delta_ref_test)
    err_ddelta = relative_error_percent(ddelta_nn, ddelta_ref_test)

    fig, ax = plt.subplots()
    ax.plot(a_test, delta_ref_test, label=r"$\delta_m$ (RK45)")
    ax.plot(a_test, delta_nn, "--", label=r"$\delta_m$ (PINN)")
    ax.plot(a_test, ddelta_ref_test, label=r"$\delta_m'$ (RK45)")
    ax.plot(a_test, ddelta_nn, "--", label=r"$\delta_m'$ (PINN)")
    ax.set_xscale("log")
    ax.set_xlabel("a")
    ax.set_ylabel(r"$\delta_m$, $\delta_m'$")
    ax.set_title(rf"$\Omega_{{m0}}$ = {om_m0}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(a_test, err_delta, label=r"error relativo $\delta_m$")
    ax.plot(a_test, err_ddelta, label=r"error relativo $\delta_m'$")
    ax.set_xscale("log")
    ax.set_xlabel("a")
    ax.set_ylabel("error relativo (%)")
    ax.set_title(rf"$\Omega_{{m0}}$ = {om_m0}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "error_curve.png", dpi=150)
    plt.close(fig)

    return err_delta, err_ddelta


def sweep_heatmap(solution, config, figures_dir, num_omegas=40, num_points=200):
    om_min, om_max = config["Om_m0_range"]
    omegas = np.linspace(om_min, om_max, num_omegas)

    err_grid = np.zeros((num_omegas, num_points))
    a_ref = None
    for i, om in enumerate(omegas):
        a, _, _, delta_interp, _ = solve(om, num_points=1000)
        if a_ref is None:
            a_ref = np.logspace(np.log10(a.min()), np.log10(a.max()), num_points)
        delta_ref = delta_interp(a_ref)
        delta_nn, _ = delta_pinn(solution, config, a_ref, om)
        err_grid[i] = relative_error_percent(delta_nn, delta_ref)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(a_ref, omegas, err_grid, cmap="viridis", shading="auto")
    ax.set_xscale("log")
    ax.set_xlabel("a")
    ax.set_ylabel(r"$\Omega_{m0}$")
    ax.set_title(r"error relativo (%) de $\delta_m$")
    fig.colorbar(mesh, ax=ax, label="error (%)")
    fig.tight_layout()
    fig.savefig(figures_dir / "error_heatmap.png", dpi=150)
    plt.close(fig)

    return err_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, required=True, help="carpeta del run, p.ej. models/bundle_om_010_050")
    parser.add_argument("--om-m0", type=float, default=None, help="Ω_m0 para las figuras de comparación puntual")
    parser.add_argument("--skip-heatmap", action="store_true", help="no barrer todo el rango de Ω_m0 (más rápido)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    solution, config = load_run(run_dir)

    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    if config["kind"] == "bundle":
        om_min, om_max = config["Om_m0_range"]
        om_m0 = args.om_m0 if args.om_m0 is not None else 0.5 * (om_min + om_max)
    else:
        om_m0 = config["Om_m0"]
        if args.om_m0 is not None and abs(args.om_m0 - om_m0) > 1e-9:
            print(f"Aviso: este run es no-bundle, entrenado para Om_m0={om_m0}; se ignora --om-m0={args.om_m0}.")

    err_delta, err_ddelta = compare_one(solution, config, om_m0, figures_dir)

    print(f"Run: {run_dir}  ({config['kind']})")
    print(f"Om_m0 evaluado: {om_m0}")
    print(f"Error relativo delta   -> {error_summary(err_delta)}")
    print(f"Error relativo ddelta  -> {error_summary(err_ddelta)}")

    if config["kind"] == "bundle" and not args.skip_heatmap:
        print("\nBarriendo todo el rango de Om_m0 para el heatmap...")
        err_grid = sweep_heatmap(solution, config, figures_dir)
        print(f"Error relativo delta en todo el rango -> {error_summary(err_grid)}")

    print(f"\nFiguras guardadas en {figures_dir}")


if __name__ == "__main__":
    main()
