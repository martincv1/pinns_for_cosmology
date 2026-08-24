import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cosmopinn.cosmology import E2, OM_R0, dlnH_dN  # noqa: E402


def test_E2_today_is_one():
    """Por definición E(a=1)^2 = 1 (H(a=1) = H0), para cualquier Ω_m0."""
    for om_m0 in [0.1, 0.3, 0.5]:
        assert np.isclose(E2(1.0, om_m0), 1.0)


def test_E2_matter_domination_limit():
    """Bien adentro del régimen de materia (a=0.01, muchas veces la a de
    igualdad materia-radiación a_eq=Ω_r0/Ω_m0≈3e-4, y muy por debajo de donde
    empieza a pesar la energía oscura), E(a)^2 ≈ Ω_m0 / a^3."""
    a = 0.01
    om_m0 = 0.3
    assert np.isclose(E2(a, om_m0), om_m0 / a**3, rtol=0.05)


def test_dlnH_dN_matter_domination_limit():
    """En dominación de materia, H ~ a^(-3/2) ⇒ d(ln H)/dN → -3/2.

    Nota: a=1e-3 (el a0 de entrenamiento) todavía no sirve para este límite
    — ahí a_eq/a ≈ 0.3, la radiación sigue siendo una fracción apreciable de
    la densidad. Por eso el ansatz "δ ~ a" usado como condición inicial es
    una aproximación, no un límite exacto; a=0.01 sí está limpio.
    """
    a = 0.01
    om_m0 = 0.3
    assert np.isclose(dlnH_dN(a, om_m0), -1.5, atol=0.05)


def test_numpy_and_torch_agree():
    """E2 y dlnH_dN son puramente algebraicas: deben dar lo mismo con
    numpy que con torch, que es justamente lo que permite reusarlas en el
    residuo de la red (torch) y en el solver de referencia (numpy)."""
    a_vals = np.array([1e-3, 1e-2, 0.1, 0.5, 1.0])
    om_m0 = 0.3

    e2_np = E2(a_vals, om_m0)
    dln_np = dlnH_dN(a_vals, om_m0)

    a_t = torch.tensor(a_vals)
    e2_t = E2(a_t, om_m0)
    dln_t = dlnH_dN(a_t, om_m0)

    assert np.allclose(e2_np, e2_t.numpy())
    assert np.allclose(dln_np, dln_t.numpy())


def test_om_r0_is_small():
    """Sanity check de la constante: la radiación hoy es una fracción chica."""
    assert 0 < OM_R0 < 1e-3
