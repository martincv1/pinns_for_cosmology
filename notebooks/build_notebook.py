"""Genera notebooks/perturbaciones_pinn.ipynb. Correr una sola vez para
regenerar el notebook si cambia la narrativa; no forma parte del repo final
tal cual (se puede borrar), pero se deja documentado por transparencia."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
    """# Una red que aprende a resolver una ecuación diferencial

Este notebook cuenta la historia completa, de punta a punta, con los mismos
módulos que usan los scripts de `scripts/`. La idea de fondo:

En cosmología, la forma en que se agrupa la materia en el universo —de dónde
salen las galaxias— sigue una ecuación diferencial. Para explorar qué modelo
cosmológico describe mejor las observaciones hay que resolver esa ecuación
una y otra vez, una vez por cada combinación de parámetros que se quiere
probar. En vez de eso: ¿se puede entrenar una red que aprenda **la familia
completa de soluciones** de una sola vez, y después evaluarla al costo de un
forward pass?

Esa es la idea de una *bundle solution*. Este notebook la construye y la
valida paso a paso."""
)

md(
    """## 1. El problema: perturbaciones de materia en ΛCDM

La ecuación que describe cómo crece una perturbación de densidad $\\delta_m$
en el modelo estándar ΛCDM es

$$
\\delta_m''(a) + \\left(\\frac{H'(a)}{H(a)} + \\frac{3}{a}\\right)\\delta_m'(a)
- \\frac{3}{2}\\frac{\\Omega_{m0} H_0^2}{H(a)^2\\, a^5}\\,\\delta_m(a) = 0,
$$

donde $a$ es el factor de escala del universo ($a=1$ hoy) y $\\Omega_{m0}$ es
la fracción de materia en el universo hoy — el parámetro que se quiere
explorar. $H(a)$ es la tasa de expansión, que en ΛCDM tiene una forma
cerrada: toda la física del fondo está en `cosmopinn/cosmology.py`."""
)

code(
    """import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from cosmopinn.cosmology import E2, dlnH_dN
from cosmopinn.reference import solve
from cosmopinn.evaluate import load_run, delta_pinn, relative_error_percent, error_summary

a = np.logspace(-3, 0, 200)
plt.figure()
for om in [0.1, 0.3, 0.5]:
    plt.plot(a, np.sqrt(E2(a, om)), label=fr"$\\Omega_{{m0}}$={om}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("a"); plt.ylabel("H(a) / H0")
plt.legend(); plt.title("Tasa de expansión para distintos Ω_m0")
plt.show()"""
)

md(
    """## 2. La solución de referencia: integrarla, a fuerza bruta

Antes de entrenar nada, conviene tener una solución en la que confiar.
`cosmopinn/reference.py` integra la ecuación con `scipy.solve_ivp` (Runge-Kutta
de orden 4-5), partiendo de la condición inicial del régimen de dominación de
materia: ahí $\\delta_m \\propto a$."""
)

code(
    """a_ref, delta_ref, ddelta_ref, delta_interp, ddelta_interp = solve(0.3)

plt.figure()
plt.plot(a_ref, delta_ref, label=r"$\\delta_m$")
plt.plot(a_ref, ddelta_ref, label=r"$\\delta_m'$")
plt.xscale("log")
plt.xlabel("a"); plt.title(r"Solución de referencia, $\\Omega_{m0}=0.3$")
plt.legend(); plt.show()"""
)

md(
    """## 3. Entrenar una red para un único Ω_m0

Antes de intentar el bundle, el caso más simple: una red que aprenda
$\\delta_m(a)$ para un $\\Omega_{m0}$ fijo. No hay datos de entrenamiento — la
función de costo es directamente el residuo al cuadrado de la ecuación,
promediado sobre puntos muestreados del dominio. Esto ya está entrenado y
guardado en `models/single_om030/`; acá solo se carga y se compara."""
)

code(
    """solution, config = load_run("../models/single_om030")

a_test = np.logspace(np.log10(a_ref.min()), np.log10(a_ref.max()), 1000)
delta_nn, ddelta_nn = delta_pinn(solution, config, a_test)
delta_true = delta_interp(a_test)

err = relative_error_percent(delta_nn, delta_true)
print("Error relativo de delta:", error_summary(err))

plt.figure()
plt.plot(a_test, delta_true, label=r"$\\delta_m$ (RK45)")
plt.plot(a_test, delta_nn, "--", label=r"$\\delta_m$ (PINN)")
plt.xscale("log"); plt.legend()
plt.title(r"Red entrenada para un único $\\Omega_{m0}=0.3$")
plt.show()"""
)

md(
    """## 4. El bundle: una red, todo un rango de Ω_m0

Ahora la parte interesante. En vez de fijar $\\Omega_{m0}$, se lo agrega como
una entrada más de la red: la red aprende $\\delta_m(a, \\Omega_{m0})$ para
$\\Omega_{m0} \\in [0.1, 0.5]$ de una sola vez (`scripts/train_bundle.py`,
`cosmopinn/training.py::train_bundle`). Layer del método: `neurodiffeq`
samplea el dominio $(a, \\Omega_{m0})$ como el producto de dos generadores
independientes durante el entrenamiento."""
)

code(
    """bundle, bundle_config = load_run("../models/bundle_om_010_050")

a_test = np.logspace(np.log10(a_ref.min()), np.log10(a_ref.max()), 500)

plt.figure()
for om in [0.15, 0.3, 0.45]:
    _, _, _, delta_interp_om, _ = solve(om)
    delta_nn, _ = delta_pinn(bundle, bundle_config, a_test, om)
    plt.plot(a_test, delta_interp_om(a_test), color="k", alpha=0.4)
    plt.plot(a_test, delta_nn, "--", label=fr"$\\Omega_{{m0}}$={om}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("a"); plt.title("Una sola red evaluada en tres Ω_m0 (líneas negras: RK45)")
plt.legend(); plt.show()"""
)

md(
    """## 5. ¿Qué tan bien generaliza dentro del rango?

Un barrido en $\\Omega_{m0}$ contra la solución de referencia, en todo el
dominio de entrenamiento."""
)

code(
    """omegas = np.linspace(0.1, 0.5, 25)
a_grid = np.logspace(np.log10(a_ref.min()), np.log10(a_ref.max()), 200)
err_grid = np.zeros((len(omegas), len(a_grid)))

for i, om in enumerate(omegas):
    _, _, _, delta_interp_om, _ = solve(om)
    delta_nn, _ = delta_pinn(bundle, bundle_config, a_grid, om)
    err_grid[i] = relative_error_percent(delta_nn, delta_interp_om(a_grid))

plt.figure()
plt.pcolormesh(a_grid, omegas, err_grid, cmap="viridis", shading="auto")
plt.xscale("log"); plt.colorbar(label="error relativo (%)")
plt.xlabel("a"); plt.ylabel(r"$\\Omega_{m0}$")
plt.title("Error de la red bundle vs. RK45")
plt.show()

print("Resumen del error en toda la grilla:", error_summary(err_grid))"""
)

md(
    """## Conclusión

Una red entrenada sin datos —minimizando el residuo de la ecuación— aprende
una familia continua de soluciones que se mantiene cerca del integrador de
referencia en todo el rango de $\\Omega_{m0}$ entrenado. El límite de esta
demo es el de la demo: un solo parámetro y ΛCDM. El trabajo de tesis del
que sale este extracto extiende la misma idea a gravedad modificada f(R) y
a un espacio de 4 parámetros — fuera del alcance de este repo, que busca
mostrar el método en su forma más simple y legible."""
)

nb["cells"] = cells

with open("perturbaciones_pinn.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook escrito.")
