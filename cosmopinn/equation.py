"""La ecuación que resuelve la red, y el cambio de variables detrás de ella.

Ecuación física (ver `reference.py` para la misma ecuación sin cambiar de
variables), con `'` = derivada respecto de a:

    δ'' + (H'/H + 3/a) δ' - (3/2) Ω_m0 (H0/H)² / a^5 · δ = 0

Entrenar directamente en (a, δ) es numéricamente incómodo: δ crece varios
órdenes de magnitud entre a=10⁻³ y a=1, y el dominio de a es muy chico
comparado con el rango donde la red tiene que ajustar. Se hacen dos cambios
de variable:

1.  Variable independiente: N = ln(a), reescalada a N' = N / n0 con
    n0 = |ln(a0)|, de modo que el dominio de entrenamiento sea [-1, 0]
    en vez de [a0, 1].
2.  Variable dependiente: x = ln(δ), y = dx/dN'. Como δ ~ a en el régimen
    de materia, x crece de forma suave y acotada en vez de exponencial.

Con esos cambios, la ecuación queda:

    y' + y² + n0·(2 + dlnH/dN)·y - (3/2)·n0²·Ω_m0 / (E(a)² a³) = 0
    x' = y

donde `dlnH/dN` y `E(a)²` vienen de `cosmology.py` — la misma física que usa
el solver de referencia. La derivación y su verificación numérica contra la
parametrización original de la tesis están en `notes/derivation.md`.
"""

import numpy as np
import torch
from neurodiffeq import diff

from cosmopinn.cosmology import E2, dlnH_dN

# Dominio de entrenamiento en el factor de escala a.
A0 = 1e-3
AF = 1.0

N0 = np.log(A0)
NF = np.log(AF)
N0_ABS = np.abs(N0)  # n0, el factor de reescala de N

# Dominio de la variable independiente reescalada N' = N / n0.
NP0 = N0 / N0_ABS  # -1.0
NPF = NF / N0_ABS  # 0.0

# Condición inicial: régimen de dominación de materia, δ(a) ∝ a.
# x = ln(δ) = ln(a) = N = n0 · N'  ⇒  x(N'=-1) = -n0.
# x' = dx/dN' = d(ln a)/dN' = n0 · d(ln a)/dN = n0 ⇒ x'(N'=-1) = n0.
X_INIT = -N0_ABS
XP_INIT = N0_ABS


def residual(x, x_prime, N_p, Om_m0):
    """Residuo de la ODE, listo para `Solver1D`/`BundleSolver1D` de neurodiffeq.

    `Om_m0` puede ser un escalar (caso no-bundle) o un tensor del tamaño del
    batch (caso bundle) — la misma función sirve para los dos scripts de
    entrenamiento.
    """
    N = N0_ABS * N_p
    a = torch.exp(N)

    factor = N0_ABS * (2 + dlnH_dN(a, Om_m0))
    source = 1.5 * N0_ABS**2 * Om_m0 / (E2(a, Om_m0) * a**3)

    res_x = diff(x, N_p) - x_prime
    res_y = diff(x_prime, N_p) + x_prime**2 + factor * x_prime - source

    return [res_x, res_y]


def to_physical(x, y, a):
    """Deshace el cambio de variables: (x, y) → (δ, dδ/da).

    δ = exp(x). dδ/da = δ · dx/da = δ · (dx/dN') · (dN'/dN) · (dN/da)
             = δ · y · (1/n0) · (1/a).
    """
    delta = np.exp(x)
    ddelta_da = delta * y / (N0_ABS * a)
    return delta, ddelta_da
