
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from .domains import Element, Domain

@dataclass
class Electrical(Element):
    domain: ClassVar[Domain] = Domain.ELECTRICAL

@dataclass
class Re(Electrical):
    R: float = 0.0
    def impedance(self, omega): return np.broadcast_to(self.R + 0j, omega.shape)

@dataclass
class Le(Electrical):
    L: float = 0.0
    def impedance(self, omega): return 1j * omega * self.L

@dataclass
class Ce(Electrical):
    C: float = 0.0
    def impedance(self, omega): return 1/(1j * omega * self.C)

@dataclass
class CeNonIdeal(Electrical):
    C: float
    ESR: float = 0.0
    ESL: float = 0.0
    R_leak: float = float('inf')
    def impedance(self, omega):
        Z_series = self.ESR + 1j*omega*self.ESL + 1/(1j*omega*self.C)
        if self.R_leak == float('inf'):
            return Z_series
        return 1 / (1/Z_series + 1/self.R_leak)
