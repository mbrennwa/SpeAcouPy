
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from .domains import Element, Domain

@dataclass
class Mechanical(Element):
	domain: ClassVar[Domain] = Domain.MECHANICAL

@dataclass
class Rms(Mechanical):
	R: float = 0.0
	def impedance(self, omega): return np.broadcast_to(self.R + 0j, omega.shape)

@dataclass
class Mms(Mechanical):
	M: float = 0.0
	def impedance(self, omega): return 1j * omega * self.M

@dataclass
class Cms(Mechanical):
	C: float = 0.0
	def impedance(self, omega): return 1/(1j * omega * self.C)
