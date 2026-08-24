"""Fondo cosmológico ΛCDM.

Este módulo es la única fuente de verdad sobre el fondo: todo lo demás
(la ecuación que entrena la red y el solver numérico de referencia) llama a
`E2` y `dlnH_dN` en vez de reescribir las fórmulas. Así ambos caminos usan
exactamente la misma física, y un cambio acá se propaga a los dos.

Las dos funciones son puramente algebraicas (sumas, productos, potencias),
por lo que funcionan sin cambios tanto con arrays de numpy como con tensores
de torch — es lo que permite reusarlas dentro del residuo de la PINN
(`equation.py`, que necesita autograd de torch) y dentro del integrador de
referencia (`reference.py`, que usa scipy con numpy).
"""

# Densidad de radiación hoy, en unidades de la densidad crítica (Ω_r0).
# Fijo: no es un parámetro de entrenamiento, ΛCDM lo trae dado.
OM_R0 = 9.04e-5


def E2(a, Om_m0):
    """E(a)^2 = (H(a)/H0)^2 para ΛCDM.

    E(a)^2 = Ω_r0/a^4 + Ω_m0/a^3 + Ω_Λ0, con Ω_Λ0 = 1 - Ω_m0 - Ω_r0.
    """
    Om_L0 = 1 - Om_m0 - OM_R0
    return OM_R0 / a**4 + Om_m0 / a**3 + Om_L0


def dlnH_dN(a, Om_m0):
    """d(ln H)/dN, con N = ln(a).

    Se obtiene de ln H = ½ ln E(a)^2 + const., usando d/dN = a·d/da:

        d(ln H)/dN = ½ · (a · dE²/da) / E²
    """
    Om_L0 = 1 - Om_m0 - OM_R0
    E2_val = OM_R0 / a**4 + Om_m0 / a**3 + Om_L0
    a_dE2_da = -4 * OM_R0 / a**4 - 3 * Om_m0 / a**3
    return 0.5 * a_dE2_da / E2_val
