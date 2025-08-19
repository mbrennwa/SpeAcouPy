# import things so they will be available:
# import speacoupy as sp
# sp.Horn(...)

from .domains import Domain, Element, Net, Series, Parallel
from .electrical import Electrical, Re, Le, Ce
from .mechanical import Mechanical, Rms, Mms, Cms

from .acoustic import (
    Acoustic, Ra, Ma, Ca,
    SealedBox, Port, VentedBox,
    RadiationPiston, RadiationSpace, Horn,
    piston_directivity,
)

from .transformers import AcToMech, MechToElec, ElecToMech
from .driver import DriverMechanicalBranch, Driver
from .response import ResponseSolver, ResponseResult, omega_logspace
from .plotting import plot_spl, plot_impedance, plot_spl_multi

# import ALL public constants directly (no indirect re-export via acoustic)
from .constants import PROGRAMNAME, TWOPI, RHO0, C0, P0, FARFIELD_DIST_M

# utilities / helpers exposed at top level
from .radiators import collect_radiators

__all__ = [
    # domains
    "Domain", "Element", "Net", "Series", "Parallel",
    # electrical
    "Electrical", "Re", "Le", "Ce",
    # mechanical
    "Mechanical", "Rms", "Mms", "Cms",
    # acoustic
    "Acoustic", "Ra", "Ma", "Ca", "SealedBox", "Port", "VentedBox",
    "RadiationPiston", "RadiationSpace", "Horn", "piston_directivity",
    # transformers
    "AcToMech", "MechToElec", "ElecToMech",
    # driver
    "DriverMechanicalBranch", "Driver",
    # response
    "ResponseSolver", "ResponseResult", "omega_logspace",
    # plotting
    "plot_spl", "plot_impedance", "plot_spl_multi",
    # constants
    "PROGRAMNAME", "TWOPI", "RHO0", "C0", "P0", "FARFIELD_DIST_M",
    # utilities
    "collect_radiators",
]

