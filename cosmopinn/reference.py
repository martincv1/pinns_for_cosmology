"""Solución de referencia: la misma ecuación, integrada con `scipy.solve_ivp`.

Sirve para validar la red — no para reemplazarla. Resolver así es exacto
(dentro de la tolerancia numérica) pero hay que integrar una vez por cada
valor de Ω_m0.

Trabaja directamente en la variable física a (sin el cambio de variables de
`equation.py`), y usa las mismas `E2`/`dlnH_dN` de `cosmology.py` que el
residuo de la red — así ambos caminos parten de la misma física.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from cosmopinn.cosmology import E2, dlnH_dN

Z0 = 1001.0  # controla a_min = 1/Z0, bien adentro del régimen de materia


def _rhs(a, y, Om_m0):
    delta, ddelta = y
    term1 = (dlnH_dN(a, Om_m0) + 3) / a
    term2 = 1.5 * Om_m0 / (E2(a, Om_m0) * a**5)
    return [ddelta, -term1 * ddelta + term2 * delta]


def solve(Om_m0, num_points=1000, z0=Z0):
    """Integra δ(a) y dδ/da entre a_min = 1/(1+z0) y a=1.

    Devuelve `(a, delta, ddelta_da, delta_interp, ddelta_interp)`, con los
    dos últimos interpoladores cúbicos para evaluar en cualquier grilla.
    """
    a_min = 1.0 / z0
    a_max = 1.0
    a_eval = np.logspace(np.log10(a_min), np.log10(a_max), num_points)
    # logspace puede desbordar por redondeo los extremos que le pedimos;
    # recortamos a_min/a_max al rango realmente cubierto por a_eval.
    a_eval = a_eval[(a_eval >= a_min) & (a_eval <= a_max)]
    a_min, a_max = a_eval.min(), a_eval.max()

    y0 = [a_min, 1.0]  # régimen de materia: δ ∝ a
    sol = solve_ivp(
        _rhs, [a_min, a_max], y0, args=(Om_m0,), t_eval=a_eval, method="RK45", atol=1e-12, rtol=1e-10
    )

    a, delta, ddelta_da = sol.t, sol.y[0], sol.y[1]
    delta_interp = interp1d(a, delta, kind="cubic", bounds_error=True)
    ddelta_interp = interp1d(a, ddelta_da, kind="cubic", bounds_error=True)
    return a, delta, ddelta_da, delta_interp, ddelta_interp
