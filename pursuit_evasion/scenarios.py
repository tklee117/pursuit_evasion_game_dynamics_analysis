"""Example games for the trajectory figure and the animation."""
import numpy as np

SCENARIOS = {
    "Pursuer faster": dict(params=dict(a_P=2.0, a_E=1.2), T=30.0,
                           X0=np.array([-7.37, 0.46, 0.28, 0.71, -0.98, -10.5, 0.19, -0.48])),
    "Evader faster": dict(params=dict(a_P=1.2, a_E=2.0), T=30.0,
                          X0=np.array([4.33, -1.21, -0.28, -0.28, 10.88, 9.08, -0.03, 0.92])),
}
