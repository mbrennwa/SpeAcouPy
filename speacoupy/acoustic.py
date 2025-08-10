
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from .domains import Element, Domain

# Physical constants (room temp)
RHO0 = 1.2041      # air density [kg/m^3]
C0   = 343.0       # speed of sound [m/s]
P0   = 20e-6       # reference pressure [Pa]

@dataclass
class Acoustic(Element):
    domain: ClassVar[Domain] = Domain.ACOUSTIC

@dataclass
class Ra(Acoustic):
    R: float = 0.0
    def impedance(self, omega): return np.broadcast_to(self.R + 0j, omega.shape)

@dataclass
class Ma(Acoustic):
    M: float = 0.0
    def impedance(self, omega): return 1j * omega * self.M

@dataclass
class Ca(Acoustic):
    C: float = 0.0
    def impedance(self, omega): return 1/(1j * omega * self.C)

@dataclass
class SealedBox(Acoustic):
    Vb: float  # m^3
    def impedance(self, omega):
        Cab = self.Vb / (RHO0 * C0**2)
        return 1/(1j * omega * Cab)

@dataclass
class Port(Acoustic):
    diameter: float
    length: float
    alpha_in: float = 0.85
    alpha_out: float = 0.61
    R_loss: float = 0.0
    def impedance(self, omega):
        r = 0.5 * self.diameter
        S = np.pi * r**2
        L_eff = self.length + (self.alpha_in + self.alpha_out) * r
        M_a = RHO0 * L_eff / S
        Z = 1j * omega * M_a
        if self.R_loss:
            Z = Z + self.R_loss
        return Z

@dataclass
class VentedBox(Acoustic):
    Vb: float
    port: Port
    def impedance(self, omega):
        Z_box = SealedBox(self.Vb).impedance(omega)
        Z_port = self.port.impedance(omega)
        Y = 1/Z_box + 1/Z_port
        return 1/Y
