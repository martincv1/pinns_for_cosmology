import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cosmopinn.reference import solve  # noqa: E402


def test_matter_domination_growth():
    """Cerca de a0=1e-3 el crecimiento arranca en δ = a exactamente (es la
    condición inicial impuesta). Todavía hay ~30% de radiación ahí (a0 está
    a solo ~3 veces a_eq), así que δ/a se aparta un poco de 1 apenas se avanza
    — no es ruido numérico, es la física de la aproximación "matter regime"
    usada como condición inicial. Solo se pide que no se aparte mucho."""
    a, delta, ddelta_da, _, _ = solve(0.3, num_points=2000)
    early = a < 3e-3
    assert np.allclose(delta[early] / a[early], 1.0, atol=0.06)


def test_growth_suppressed_by_dark_energy():
    """Con más Ω_m0 (menos energía oscura) el crecimiento hoy (a=1) relativo a
    su valor en el régimen de materia es mayor: la energía oscura frena el
    crecimiento de estructura en tiempos recientes."""
    a_low, delta_low, _, _, _ = solve(0.1)
    a_high, delta_high, _, _, _ = solve(0.5)

    growth_low = delta_low[-1] / a_low[0]
    growth_high = delta_high[-1] / a_high[0]

    assert growth_high > growth_low


def test_solution_is_monotonic():
    """δ crece monótonamente con a en este régimen (no hay oscilaciones)."""
    _, delta, _, _, _ = solve(0.3)
    assert np.all(np.diff(delta) > 0)
