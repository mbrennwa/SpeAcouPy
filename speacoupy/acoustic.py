
from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Optional, Iterable
import numpy as np
from scipy.special import j1, struve  # wideband piston needs Bessel/Struve
from .domains import Element, Domain

# Physical constants (room temp)
from .constants import RHO0, C0, P0

@dataclass
class Acoustic(Element):
	def radiation_channels(self, omega, U_in=None):
		return []

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
	Rb: float  # Pa·s/m^3, acoustic loss shunt; must be > 0

	def __post_init__(self):
		if not (float(self.Rb) > 0.0):
			raise ValueError("SealedBox: Rb must be > 0 (Pa·s/m^3).")

	def _cab(self):
		return self.Vb / (RHO0 * C0**2)

	def impedance(self, omega):
		Cab = self._cab()
		# Parallel of Rb and Cab; Rb is guaranteed > 0
		return 1.0 / ((1.0 / float(self.Rb)) + 1j * omega * Cab)


@dataclass
class Port(Acoustic):
	diameter: float
	length: float
	Rp: float
	### mouth_load: Optional[Acoustic] = None
	mouth_load: Acoustic
	alpha_in: float = 0.85
	alpha_out: float = 0.61
	
	def __post_init__(self):
		if not (float(self.Rp) > 0.0):
			raise ValueError("Port: Rp must be > 0 Pa·s/m^3.")
		if not (float(self.diameter) > 0.0):
			raise ValueError("Port: diameter must be > 0 m.")
		if not (float(self.length) > 0.0):
			raise ValueError("Port: length must be > 0 m.")

	def impedance(self, omega):
		r = 0.5 * self.diameter
		S = np.pi * r**2
		L_eff = self.length + (self.alpha_in + self.alpha_out) * r
		M_a = RHO0 * L_eff / S
		Z = 1j * omega * M_a
		if self.Rp:
			Z = Z + self.Rp
		return Z

@dataclass
class VentedBox(Acoustic):
	Vb: float
	Rb: float # see SealedBox
	port: Port
	
	def __post_init__(self):
		if not (float(self.Rb) > 0.0):
			raise ValueError("VentedBox: Rb must be > 0 Pa·s/m^3.")

	def impedance(self, omega):
		Z_box = SealedBox(self.Vb, self.Rb).impedance(omega)
		Z_port = self.port.impedance(omega)
		Y = 1/Z_box + 1/Z_port
		return 1/Y

	def radiation_channels(self, omega, U_in=None):
		Z_box = SealedBox(self.Vb, self.Rb).impedance(omega)
		Z_port = self.port.impedance(omega)
		Yb = 1.0 / (Z_box + 0j)
		Yp = 1.0 / (Z_port + 0j)
		H = Yp / (Yp + Yb)
		U_port = H * (U_in if U_in is not None else np.zeros_like(omega, dtype=complex))
		label = getattr(self.port, "label", None)
		# Enforce label only when the port radiates into radiation_space
		if getattr(self, 'port_load', None) == 'radiation_space' and (label is None or (isinstance(label, str) and not label.strip())):
			raise ValueError("VentedBox port requires 'port_label' when port_load is 'radiation_space'.")
		return [ { "label": label, "U": U_port } ]


@dataclass
class RadiationPiston(Acoustic):
	"""Wideband baffled circular piston radiation impedance.
	Z = ρ0 c0 π a^2 * [ 1 - J1(2ka)/(k a) - j H1(2ka)/(k a) ]
	A boundary factor k_b ∈ {1,2,4,8} scales R and X for 4π,2π,1π,1/2π respectively.
	"""
	Sd: float        # diaphragm area [m^2]
	loading: str     # boundary loading: 4pi|2pi|1pi|1/2pi|0.5pi

	def radiation_channels(self, omega, U_in=None):
		# emit a single radiator channel with the incoming acoustic volume flow
		label = getattr(self, "label", None)
		U = U_in if U_in is not None else np.zeros_like(omega, dtype=complex)
		return [ { "label": label, "U": U } ]

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

class RadiationSpace:
	"""Internal helper holding a boundary (4pi/2pi/1pi/1/2pi).
	Not a YAML element. Used to treat 'radiation_space' like a normal element in code.
	"""
	def __init__(self, space: str = "4pi"):
		space = (space or "4pi").strip().lower()
		if space == "0.5pi":
			space = "1/2pi"
		if space not in {"4pi","2pi","1pi","1/2pi"}:
			raise ValueError(f"Invalid radiation_space '{space}'.")
		self.space = space

	def make_piston(self, Sd: float):
		return RadiationPiston(Sd=Sd, loading=self.space)

	def __repr__(self):
		return f"RadiationSpace({self.space})"

from dataclasses import dataclass

@dataclass
class Horn(Acoustic):
	L: float
	S_throat: float
	S_mouth: float
	profile: str  # "con" | "exp"
	label: str = ""
	mouth_load: str | None = None
	mouth_label: str | None = None
	throat_load: str | None = None
	throat_label: str | None = None

	_m: float | None = None  # derived for exp

	def __post_init__(self) -> None:
		self.profile = (self.profile or "con").lower()
		if self.profile not in ("con", "exp"):
			raise ValueError("Horn.profile must be 'con' or 'exp'")
		if self.L <= 0 or self.S_throat <= 0 or self.S_mouth <= 0:
			raise ValueError("Horn: L, S_throat, S_mouth must be > 0")
		if self.profile == "exp":
			self._m = np.log(self.S_mouth / self.S_throat) / self.L
		else:
			self._m = None
		if (self.mouth_load == "radiation_space") and not self.mouth_label:
			raise ValueError("Horn: mouth_label is required when mouth_load='radiation_space'")
		if (self.throat_load == "radiation_space") and not self.throat_label:
			raise ValueError("Horn: throat_label is required when throat_load='radiation_space'")

	def _area(self, x: np.ndarray) -> np.ndarray:
		if self.profile == "con":
			return self.S_throat + (self.S_mouth - self.S_throat) * (x / self.L)
		return self.S_throat * np.exp((self._m or 0.0) * x)

	def _abcd_chain(self, omega: np.ndarray, N: int = 64):
		F = omega.size
		dx = self.L / N
		xc = (np.arange(N) + 0.5) * dx
		S = self._area(xc)[None, :]
		gamma = 1j * omega / C0
		cosh_gdx = np.cosh(gamma * dx)
		sinh_gdx = np.sinh(gamma * dx)
		Zc = (RHO0 * C0) / S
		A = np.ones((F,), dtype=np.complex128)
		B = np.zeros((F,), dtype=np.complex128)
		C = np.zeros((F,), dtype=np.complex128)
		D = np.ones((F,), dtype=np.complex128)
		for i in range(N):
			Ai = cosh_gdx
			Bi = Zc[:, i] * sinh_gdx
			Ci = sinh_gdx / Zc[:, i]
			Di = cosh_gdx
			A, B, C, D = A*Ai + B*Ci, A*Bi + B*Di, C*Ai + D*Ci, C*Bi + D*Di
		return A, B, C, D

	def _radiation_impedance(self, omega: np.ndarray, S_ap: float) -> np.ndarray:
		Z0 = (RHO0 * C0) / S_ap
		a = np.sqrt(S_ap / np.pi)
		k = omega / C0
		ka = np.maximum(k * a, 1e-12)
		R = Z0 * (1.0 - np.sinc(ka / np.pi))
		X = Z0 * ((8.0 / (3.0 * np.pi)) * ka)
		return R + 1j * X

	def _load_impedance_at_mouth(self, omega: np.ndarray) -> np.ndarray:
		if self.mouth_load == "rigid":
			return np.full_like(omega, np.inf + 0j)
		if self.mouth_load == "radiation_space":
			return self._radiation_impedance(omega, self.S_mouth)
		if isinstance(self.mouth_load, str) and self.mouth_load not in ("", None):
			return np.full_like(omega, np.inf + 0j)
		return (RHO0 * C0) / self.S_mouth + 0j

	def _load_impedance_at_throat(self, omega: np.ndarray) -> np.ndarray | None:
		if self.throat_load == "rigid":
			return np.full_like(omega, np.inf + 0j)
		if self.throat_load == "radiation_space":
			return self._radiation_impedance(omega, self.S_throat)
		return None

	def impedance(self, omega: np.ndarray) -> np.ndarray:
		A, B, C, D = self._abcd_chain(omega)
		ZL = self._load_impedance_at_mouth(omega)
		Zin = (A * ZL + B) / (C * ZL + D)
		Zth = self._load_impedance_at_throat(omega)
		if Zth is not None:
			Zin = (Zin * Zth) / (Zin + Zth)
		return Zin

	def radiation_channels(self, omega: np.ndarray, U_in: np.ndarray | None = None):
		A, B, C, D = self._abcd_chain(omega)
		ZL = self._load_impedance_at_mouth(omega)
		den = (C * ZL + D)
		den = np.where(np.abs(den) < 1e-18, 1e-18 + 0j, den)
		H_umouth = 1.0 / den

		chs = []
		if self.mouth_load == "radiation_space" and self.mouth_label:
			if U_in is None:
				Ui = H_umouth      # transfer function
			else:
				Ui = H_umouth * U_in
			chs.append({
				"label": self.mouth_label,
				"U": Ui,
				"S": self.S_mouth,
			})

		if self.throat_load == "radiation_space" and self.throat_label:
			if U_in is None:
				Ui = np.ones_like(omega, dtype=complex)  # unit transfer
			else:
				Ui = U_in
			chs.append({
				"label": self.throat_label,
				"U": Ui,
				"S": self.S_throat,
			})

		return chs
