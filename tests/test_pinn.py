"""Tests de regresión numérica: cargan los pesos ya entrenados y commiteados
en `models/` y verifican que sigan de acuerdo con el solver de referencia.

Estos son los tests que respaldan los números de error que reporta el
README — si alguien cambia la ecuación o reentrena sin revisar, fallan acá.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from cosmopinn.evaluate import delta_pinn, error_summary, load_run, relative_error_percent  # noqa: E402
from cosmopinn.reference import solve  # noqa: E402

SINGLE_RUN = ROOT / "models" / "single_om030"
BUNDLE_RUN = ROOT / "models" / "bundle_om_010_050"

pytestmark = pytest.mark.skipif(
    not (SINGLE_RUN / "nets.pth").exists() or not (BUNDLE_RUN / "nets.pth").exists(),
    reason="pesos entrenados no encontrados en models/ (correr scripts/train_*.py primero)",
)


def _median_error(run_dir, om_m0):
    solution, config = load_run(run_dir)
    a, _, _, delta_interp, _ = solve(om_m0)
    a_test = np.logspace(np.log10(a.min()), np.log10(a.max()), 500)
    delta_ref = delta_interp(a_test)
    delta_nn, _ = delta_pinn(solution, config, a_test, om_m0)
    err = relative_error_percent(delta_nn, delta_ref)
    return error_summary(err)


def test_single_run_matches_reference():
    summary = _median_error(SINGLE_RUN, 0.3)
    assert summary["median"] < 2.0
    assert summary["max"] < 5.0


@pytest.mark.parametrize("om_m0", [0.15, 0.30, 0.45])
def test_bundle_run_matches_reference_across_range(om_m0):
    summary = _median_error(BUNDLE_RUN, om_m0)
    assert summary["median"] < 3.0
    assert summary["max"] < 8.0
