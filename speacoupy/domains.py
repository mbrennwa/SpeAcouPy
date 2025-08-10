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
class Net(Element):
    """
    Universal two-terminal network: op ∈ {'series','parallel'} with a list of parts.
    Enforces that all parts share the same domain (insert adapters to convert).
    """
    op: str
    parts: Sequence[Element]

    def __post_init__(self):
        if not self.parts:
            raise ValueError("Net() requires at least one element")
        op_norm = self.op.lower().strip()
        if op_norm not in ("series","parallel"):
            raise ValueError(f"Net.op must be 'series' or 'parallel', got: {self.op}")
        self.op = op_norm
        d0 = getattr(self.parts[0], "domain", None)
        if any(getattr(p, "domain", None) != d0 for p in self.parts):
            raise ValueError("All elements in a Net must share the same domain. Insert adapters/transformers.")
        self.domain = d0  # type: ignore[attr-defined]

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        if self.op == "series":
            Z = 0j
            for p in self.parts:
                Z = Z + p.impedance(omega)
            return Z
        # parallel
        Y = 0j
        for p in self.parts:
            Zp = p.impedance(omega)
            Y = Y + 1/Zp
        return 1/Y

# Backward-compatible convenience wrappers
@dataclass
class Series(Net):
    def __init__(self, parts: Sequence[Element]):
        super().__init__(op="series", parts=parts)

@dataclass
class Parallel(Net):
    def __init__(self, parts: Sequence[Element]):
        super().__init__(op="parallel", parts=parts)
