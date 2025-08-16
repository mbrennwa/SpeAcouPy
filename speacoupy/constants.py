PROGRAMNAME = 'SpeAcouPy'

import numpy as np
TWOPI = 2.0 * np.pi

# Physical constants (room temp)
RHO0 = 1.2041      # air density [kg/m^3]
C0   = 343.0       # speed of sound [m/s]
P0   = 20e-6       # reference pressure [Pa]

# Far-field SPL reference distance (meters)
FARFIELD_DIST_M = 1.0
