
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
    acoustic_load: Optional[Acoustic] = None
    Sd: float = 1.0
    domain: ClassVar[Domain] = Domain.MECHANICAL
    def impedance(self, omega):
        Zm = (self.Rms_val + 0j) + 1j*omega*self.Mms_val + 1/(1j*omega*self.Cms_val)
        if self.acoustic_load is not None:
            Zm = Zm + AcToMech(self.acoustic_load, self.Sd).impedance(omega)
        return Zm

@dataclass
class Driver(Element):
    Re_val: float
    Le_val: float
    Bl: float
    motional: DriverMechanicalBranch
    domain: ClassVar[Domain] = Domain.ELECTRICAL
    def impedance(self, omega):
        Zvc = self.Re_val + 1j*omega*self.Le_val
        Ze_mot = (self.Bl**2) / self.motional.impedance(omega)
        return Zvc + Ze_mot
