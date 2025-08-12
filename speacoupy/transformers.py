
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from .domains import Element, Domain
from .mechanical import Mechanical
from .acoustic import Acoustic

@dataclass
class MechToElec(Element):
	load: Mechanical
	Bl: float
	domain: ClassVar[Domain] = Domain.ELECTRICAL
	def impedance(self, omega):
	    Zm = self.load.impedance(omega)
	    return (self.Bl**2) / Zm

@dataclass
class AcToMech(Element):
	load: Acoustic
	Sd: float
	domain: ClassVar[Domain] = Domain.MECHANICAL
	def impedance(self, omega):
	    Za = self.load.impedance(omega)
	    return (self.Sd**2) * Za
