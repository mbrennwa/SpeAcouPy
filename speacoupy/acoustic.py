
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Optional, Iterable
import numpy as np
from scipy.special import j1, struve  # wideband piston needs Bessel/Struve
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
class RadiationPiston(Acoustic):
	"""Wideband baffled circular piston radiation impedance.
	Z = ρ0 c0 π a^2 * [ 1 - J1(2ka)/(k a) - j H1(2ka)/(k a) ]
	A boundary factor k_b ∈ {1,2,4,8} scales R and X for 4π,2π,1π,1/2π respectively.
	"""
	Sd: float                # diaphragm area [m^2]
	loading: str = "4pi"     # boundary loading: 4pi|2pi|1pi|1/2pi|0.5pi

	def impedance(self, omega):
		S = self.Sd
		a = np.sqrt(S / np.pi)
		k = omega / C0
		ka = k * a
		x = 2.0 * ka
		with np.errstate(divide='ignore', invalid='ignore'):
			J = j1(x)
			H = struve(1, x)
			denom = np.where(ka == 0, np.inf, ka)
			Rn = 1.0 - (J / denom)
			Xn = - (H / denom)
		Z0 = RHO0 * C0 * np.pi * a*a * (Rn + 1j * Xn)

		kb_map = {"4pi": 1.0, "2pi": 2.0, "1pi": 4.0, "1/2pi": 8.0, "0.5pi": 8.0}
		kb = kb_map.get((self.loading or "4pi").lower(), 1.0)
		return kb * Z0

def piston_directivity(Sd: float, omega: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
	a = np.sqrt(Sd / np.pi)
	k = omega / C0
	th = theta_rad.reshape((-1,1))
	x = (k.reshape((1,-1)) * a * np.maximum(1e-16, np.sin(th)))
	from scipy.special import j1 as J1
	num = 2.0 * J1(x)
	den = np.where(x==0, 1.0, x)
	D = num / den
	return D
