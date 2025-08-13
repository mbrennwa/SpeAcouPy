
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Optional
import numpy as np
from .domains import Element, Domain
from .transformers import AcToMech
from .acoustic import Acoustic

@dataclass
class DriverMechanicalBranch:
	Rms_val: float
	Mms_val: float
	Cms_val: float
	front_load: Optional[Acoustic] = None
	back_load: Optional[Acoustic] = None
	Sd: float = 1.0
	domain: ClassVar[Domain] = Domain.MECHANICAL
	
	def impedance(self, omega):
		Zm = (self.Rms_val + 0j) + 1j*omega*self.Mms_val + 1/(1j*omega*self.Cms_val)
		if self.front_load is None or self.back_load is None:
			raise ValueError("DriverMechanicalBranch requires both front_load and back_load to be specified.")
		Za_mech = AcToMech(self.front_load, self.Sd).impedance(omega) + AcToMech(self.back_load, self.Sd).impedance(omega)
		return Zm + Za_mech

@dataclass
class Driver(Element):
	Re_val: float
	k_semi: float
	alpha_semi: float
	Bl: float
	motional: DriverMechanicalBranch
	domain: ClassVar[Domain] = Domain.ELECTRICAL
	
	def impedance(self, omega):
		Zvc = self.impedance_voicecoil(omega)
		Ze_mot = (self.Bl**2) / self.motional.impedance(omega)
		return Zvc + Ze_mot
		
	def impedance_voicecoil(self, omega):
		# Semi-inductance (lossy inductor) model: Z = Re + k*(j*omega)^alpha
		return self.Re_val + self.k_semi * (1j*omega)**(self.alpha_semi)
		
	def Sd(self):
		return self.motional.Sd
