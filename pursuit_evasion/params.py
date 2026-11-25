"""Game parameters (report Table 1 and Sections III, VI).

The protected zone is centred at the origin. k_v is the velocity-damping gain of the
modified pursuer law in Section VI; it sets Re(lambda) = -k_v / 2 = -0.5.
"""
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Params:
    a_P: float = 1.5        # pursuer max acceleration [m/s^2]
    v_P: float = 3.0        # pursuer max speed [m/s]
    a_E: float = 1.5        # evader max acceleration [m/s^2]
    v_E: float = 3.0        # evader max speed [m/s]
    rho_star: float = 8.0   # chase/defend crossover range [m]
    k_alpha: float = 0.5    # chase/defend blend sharpness
    d_star: float = 8.0     # dodge crossover range [m]
    k_beta: float = 0.5     # dodge sharpness
    R_Z: float = 3.5        # protected-zone radius [m]
    r_c: float = 0.5        # capture radius [m]
    k_v: float = 1.0        # damping gain of the modified pursuer law [1/s]
    x_Z: np.ndarray = field(default_factory=lambda: np.zeros(2))  # zone centre [m]

    def with_(self, **kw):
        return replace(self, **kw)


# Initial condition used for Picard iteration and the continuous-dependence tests (Table 1)
X0 = np.array([-4.0, 3.0, 1.0, -2.0,   # pursuer  x, y, vx, vy
               5.0, -5.0, 1.0, 2.0])   # evader   x, y, vx, vy
T_PICARD = 3.5  # [s]

# Equilibrium of the reduced pursuer subsystem (Section VI): evader frozen at x_E*
X_P_STAR = np.array([4.5, 0.0])
X_E_STAR = np.array([12.5, 0.0])
