
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
