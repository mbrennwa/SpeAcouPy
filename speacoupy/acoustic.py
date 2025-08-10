
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

@dataclass
class RadiationPistonLF(Acoustic):
    """Low-frequency baffled piston radiation with boundary loading.
    Approximation:
      eta ≈ (ka)^2 / 2       (radiation efficiency)
      X/(ρcS) ≈ (8/(3π))ka   (reactive part)
    Boundary loading factor k_b (image sources):
      4π -> 1, 2π -> 2, 1π -> 4, 1/2π -> 8
    We scale both R and X by k_b (simple LF approximation).
    """
    Sd: float
    loading: str = "4pi"

    def impedance(self, omega):
        S = self.Sd
        a = np.sqrt(S / np.pi)
        k = omega / C0
        ka = k * a
        eta = 0.5 * (ka**2)
        Xnorm = (8.0 / (3.0 * np.pi)) * ka
        Z0 = RHO0 * C0 * S * (eta + 1j * Xnorm)

        kb_map = {"4pi": 1.0, "2pi": 2.0, "1pi": 4.0, "1/2pi": 8.0, "0.5pi": 8.0}
        kb = kb_map.get((self.loading or "4pi").lower(), 1.0)
        return kb * Z0
