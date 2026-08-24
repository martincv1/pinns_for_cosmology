"""¿Cuándo conviene entrenar la red en vez de integrar la ODE cada vez?

Entrenar el bundle tiene un costo fijo (una sola vez). Después, evaluarlo
para un Ω_m0 nuevo es un forward pass; integrar con `scipy.solve_ivp` hay
que volver a hacerlo por completo cada vez. Este script mide ambos costos
por separado y calcula el punto de equilibrio: a partir de cuántas
evaluaciones distintas de Ω_m0 la inversión de entrenar se paga sola.

Uso:
    python scripts/benchmark.py
    python scripts/benchmark.py --run models/bundle_om_010_050 --n-params 200
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cosmopinn.evaluate import delta_pinn, error_summary, load_run, relative_error_percent  # noqa: E402
from cosmopinn.reference import solve  # noqa: E402


def time_reference(om_values, num_points):
    times = []
    for om in om_values:
        t0 = time.perf_counter()
        solve(om, num_points=num_points)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def time_pinn_per_call(solution, config, om_values, a_grid):
    times = []
    for om in om_values:
        t0 = time.perf_counter()
        delta_pinn(solution, config, a_grid, om)
        times.append(time.perf_counter() - t0)
    return np.array(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=str, default=str(ROOT / "models" / "bundle_om_010_050"))
    parser.add_argument("--n-params", type=int, default=200, help="cuántos Ω_m0 distintos evaluar")
    parser.add_argument("--num-points", type=int, default=200, help="puntos de la curva δ(a) por evaluación")
    args = parser.parse_args()

    run_dir = Path(args.run)
    solution, config = load_run(run_dir)
    if config["kind"] != "bundle":
        raise SystemExit("benchmark.py necesita un run bundle (para evaluar Ω_m0 sin reentrenar)")

    train_seconds = config["train_seconds"]
    om_min, om_max = config["Om_m0_range"]

    rng = np.random.default_rng(0)
    om_values = rng.uniform(om_min, om_max, args.n_params)
    a_grid = np.logspace(np.log10(1 / 1001), 0, args.num_points)

    print(f"Run: {run_dir}")
    print(f"Costo de entrenamiento (una vez): {train_seconds:.1f} s")
    print(f"Evaluando {args.n_params} valores de Ω_m0 distintos, {args.num_points} puntos cada uno...\n")

    t_ref = time_reference(om_values, args.num_points)
    t_pinn = time_pinn_per_call(solution, config, om_values, a_grid)

    # Además del costo por-llamada, el mejor caso para la red: evaluar todos
    # los Ω_m0 juntos en un único forward pass batcheado.
    t0 = time.perf_counter()
    a_batch = np.tile(a_grid, len(om_values))
    om_batch = np.repeat(om_values, len(a_grid))
    delta_pinn(solution, config, a_batch, om_batch)
    t_pinn_batched = time.perf_counter() - t0

    # Error de paso, para dejar claro que la velocidad no viene gratis.
    errs = []
    for om in om_values[:20]:
        a_ref, _, _, delta_interp, _ = solve(om, num_points=1000)
        a_test = np.logspace(np.log10(a_ref.min()), np.log10(a_ref.max()), 200)
        delta_ref = delta_interp(a_test)
        delta_nn, _ = delta_pinn(solution, config, a_test, om)
        errs.append(relative_error_percent(delta_nn, delta_ref))
    err_summary = error_summary(np.concatenate(errs))

    print(f"scipy.solve_ivp  : {t_ref.mean()*1e3:.2f} ms/evaluación (± {t_ref.std()*1e3:.2f} ms)")
    print(f"PINN (uno a uno) : {t_pinn.mean()*1e3:.2f} ms/evaluación (± {t_pinn.std()*1e3:.2f} ms)")
    print(f"PINN (batcheado) : {t_pinn_batched/len(om_values)*1e3:.4f} ms/evaluación (todo en un forward pass)")
    print(f"speedup (uno a uno)  : {t_ref.mean()/t_pinn.mean():.1f}x")
    print(f"speedup (batcheado)  : {t_ref.mean()/(t_pinn_batched/len(om_values)):.0f}x")
    print(f"error relativo de delta en la muestra: {err_summary}")

    # Punto de equilibrio: costo_scipy(M) = train_seconds + costo_pinn(M)
    #   M * t_ref = train_seconds + M * t_pinn  =>  M* = train_seconds / (t_ref - t_pinn)
    t_ref_mean, t_pinn_mean = t_ref.mean(), t_pinn.mean()
    if t_ref_mean > t_pinn_mean:
        break_even = train_seconds / (t_ref_mean - t_pinn_mean)
    else:
        break_even = float("inf")
    print(f"\nPunto de equilibrio: ~{break_even:.0f} evaluaciones de Ω_m0")

    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)

    m = np.arange(1, max(int(break_even * 2), args.n_params * 2) + 1)
    cost_ref = m * t_ref_mean
    cost_pinn = train_seconds + m * t_pinn_mean

    fig, ax = plt.subplots()
    ax.plot(m, cost_ref, label="resolver con scipy cada vez")
    ax.plot(m, cost_pinn, label="entrenar una vez + evaluar la red")
    if np.isfinite(break_even):
        ax.axvline(break_even, color="gray", linestyle="--", label=f"equilibrio (~{break_even:.0f} evaluaciones)")
    ax.set_xlabel("cantidad de valores de Ω_m0 evaluados")
    ax.set_ylabel("tiempo acumulado (s)")
    ax.set_title("Costo acumulado: integrar vs. entrenar + evaluar")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "benchmark.png", dpi=150)
    plt.close(fig)
    print(f"\nFigura guardada en {figures_dir / 'benchmark.png'}")


if __name__ == "__main__":
    main()
