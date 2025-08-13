
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from .domains import Element, Domain
from .electrical import Electrical
from .mechanical import Mechanical
from .acoustic import Acoustic

@dataclass
class MechToElec(Element):
# convert mechanical impedance to electrical domain
	load: Mechanical
	Bl: float
	domain: ClassVar[Domain] = Domain.ELECTRICAL
	def impedance(self, omega):
		Zm = self.load.impedance(omega)
		return (self.Bl**2) / Zm

@dataclass
class ElecToMech(Element):
# convert electrical impedance to mechanical domain
	load: Electrical
	Bl: float
	domain: ClassVar[Domain] = Domain.MECHANICAL
	def impedance(self, omega):
		Ze = self.load.impedance(omega)
		print('Hello from ElecToMech')
		return (self.Bl**2) / Ze

@dataclass
class AcToMech(Element):
# convert acoustic impedance to mechanical domain
	load: Acoustic
	Sd: float
	domain: ClassVar[Domain] = Domain.MECHANICAL
	def impedance(self, omega):
		Za = self.load.impedance(omega)
		return (self.Sd**2) * Za
