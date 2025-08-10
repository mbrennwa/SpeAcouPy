
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np

class Domain(str, Enum):
    ELECTRICAL = "electrical"
    MECHANICAL = "mechanical"
    ACOUSTIC   = "acoustic"

@dataclass
class Element(ABC):
    """Generic lumped element. Subclasses implement impedance()."""

    @abstractmethod
    def impedance(self, omega: np.ndarray) -> np.ndarray:
        """Return complex impedance Z(ω) in this element's domain."""
        ...

    def to(self, domain: Domain):
        if getattr(self, "domain", None) == domain:
            return self
        raise NotImplementedError(f"{self.__class__.__name__} cannot transform to {domain} directly.")

@dataclass
class Series(Element):
    parts: Sequence[Element]

    def __post_init__(self):
        if not self.parts:
            raise ValueError("Series() requires at least one element")
        d0 = getattr(self.parts[0], "domain", None)
        if any(getattr(p, "domain", None) != d0 for p in self.parts):
            raise ValueError("All elements in Series must share the same domain. Insert adapters/transformers.")
        self.domain = d0

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        Z = 0j
        for p in self.parts:
            Z = Z + p.impedance(omega)
        return Z

@dataclass
class Parallel(Element):
    parts: Sequence[Element]

    def __post_init__(self):
        if not self.parts:
            raise ValueError("Parallel() requires at least one element")
        d0 = getattr(self.parts[0], "domain", None)
        if any(getattr(p, "domain", None) != d0 for p in self.parts):
            raise ValueError("All elements in Parallel must share the same domain. Insert adapters/transformers.")
        self.domain = d0

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        Y = 0j
        for p in self.parts:
            Zp = p.impedance(omega)
            Y = Y + 1/Zp
        return 1/Y
