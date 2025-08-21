
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
	num = 2.0 * j1(x)
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

@dataclass
class Horn(Acoustic):
	"""
	Simplified horn element — **conical only** (exact algebraic solution).

	This step intentionally supports only conical flares. Other profiles will be
	added later.

	Notes
	-----
	• **No implicit default mouth termination.** Either set an explicit `mouth_load` that
	  resolves locally (e.g. "rigid" or "radiation_space"), or pass `ZL_external` to
	  `impedance(omega, ZL_external=...)`. If neither is provided, `impedance()` raises.
	"""
	L: float
	S_throat: float
	S_mouth: float
	profile: str
	R_throat: float = 0.0
	R_mouth: float = 0.0
	throat_load: Optional[str] = None
	throat_label: Optional[str] = None
	mouth_load: Optional[str] = None
	mouth_label: Optional[str] = None
	mouth_loading: str = "4pi"
	throat_loading: str = "4pi"
	label: str = ""

	# -------------------------- Validation / setup -------------------------- #
	def __post_init__(self) -> None:
		p = (self.profile or "").strip().lower()
		if p in ("con", "conical", "cone"):
			self.profile = "con"
		else:
			raise NotImplementedError("Horn: only conical profile is supported in this revision.")

		if not (self.L > 0.0 and self.S_throat > 0.0 and self.S_mouth > 0.0):
			raise ValueError("Horn: L, S_throat, S_mouth must be > 0.")
		if not (self.R_throat >= 0.0 and self.R_mouth >= 0.0):
			raise ValueError("Horn: R_throat and R_mouth must be ≥ 0.")

		if self.mouth_load == "radiation_space" and not self.mouth_label:
			raise ValueError("Horn: mouth_label is required when mouth_load='radiation_space'.")
		if self.throat_load == "radiation_space" and not self.throat_label:
			raise ValueError("Horn: throat_label is required when throat_load='radiation_space'.")

	# ------------------------------ Utilities ------------------------------ #
	@staticmethod
	def _apex_distances(L: float, S1: float, S2: float) -> Optional[tuple[float, float, float]]:
		s1 = np.sqrt(S1)
		s2 = np.sqrt(S2)
		den = s2 - s1
		if np.all(np.abs(den) <= 1e-12 * max(1.0, float(s1))):
			return None
		absden = np.abs(den)
		r1 = L * s1 / absden
		r2 = L * s2 / absden
		G = S1 / (r1 ** 2)
		return float(r1), float(r2), float(G)

	def _radiation_impedance(self, omega: np.ndarray, S_ap: float, loading: str) -> np.ndarray:
		from .acoustic import RadiationPiston  # type: ignore
		return RadiationPiston(Sd=S_ap, loading=loading).impedance(omega)

	def _load_impedance_at_mouth(self, omega: np.ndarray) -> Optional[np.ndarray]:
		Rm = float(self.R_mouth)
		if self.mouth_load == "rigid":
			ZL = np.full_like(omega, np.inf + 0j)
		elif self.mouth_load == "radiation_space":
			ZL = self._radiation_impedance(omega, self.S_mouth, self.mouth_loading)
		elif isinstance(self.mouth_load, (float, complex)):
			ZL = np.full_like(omega, complex(self.mouth_load))
		elif isinstance(self.mouth_load, str) and self.mouth_load:
			return None
		else:
			return None
		return ZL + Rm

	# --------------------------- Core cone solution ------------------------- #
	def _BA_ratio(self, k: np.ndarray, r2: float, G: float, ZL: np.ndarray) -> np.ndarray:
		Z0 = RHO0 * C0
		# Correct boundary solve at the mouth
		num = -1j * k * Z0 + G * ZL * (r2 ** 2) * (1j * k * r2 + 1.0)
		den =  1j * k * Z0 + G * ZL * (r2 ** 2) * (1j * k * r2 - 1.0)
		den = np.where(np.abs(den) < 1e-24, 1e-24 + 0j, den)
		return np.exp(-2j * k * r2) * (num / den)

	def _U(self, r: float, k: np.ndarray, G: float, BA: np.ndarray) -> np.ndarray:
		E = np.exp(-1j * k * r)
		return G * E * (-(1.0 + 1j * k * r) + BA * np.exp(2j * k * r) * (1j * k * r - 1.0))

	def _P(self, r: float, k: np.ndarray, Z0: float, BA: np.ndarray) -> np.ndarray:
		E = np.exp(-1j * k * r)
		return -1j * k * Z0 * (E + BA * np.exp(1j * k * r)) / r

	def _conical_input_impedance(self, omega: np.ndarray, ZL: np.ndarray) -> np.ndarray:
		Z0 = RHO0 * C0
		apex = self._apex_distances(self.L, self.S_throat, self.S_mouth)
		k = omega / C0
		if apex is None:
			Zc = Z0 / self.S_throat
			t = np.tan(k * self.L)
			den = Zc + 1j * ZL * t
			den = np.where(np.abs(den) < 1e-24, 1e-24 + 0j, den)
			return Zc * (ZL + 1j * Zc * t) / den
		r1, r2, G = apex
		BA = self._BA_ratio(k, r2, G, ZL)
		p1 = self._P(r1, k, Z0, BA)
		U1 = self._U(r1, k, G, BA)
		return p1 / U1

	# ------------------------------ API methods ---------------------------- #
	def impedance(self, omega: np.ndarray, ZL_external: Optional[np.ndarray] = None) -> np.ndarray:
		ZL_local = self._load_impedance_at_mouth(omega)
		ZL = ZL_external if ZL_external is not None else ZL_local
		if ZL is None:
			raise ValueError("Horn.impedance: mouth load is unspecified.")
		Zin = self._conical_input_impedance(omega, ZL)
		if self.R_throat:
			Zin = Zin + float(self.R_throat)
		return Zin

	def radiation_channels(self, omega: np.ndarray, U_in: Optional[np.ndarray] = None):
		"""
		Return radiation channels at the horn mouth.
		"""
		if not (self.mouth_label and isinstance(self.mouth_label, str)):
			return []
		if U_in is None:
			return []

		Z0 = RHO0 * C0
		k = omega / C0
		apex = self._apex_distances(self.L, self.S_throat, self.S_mouth)
		ZL = self._load_impedance_at_mouth(omega)

		# Cylindrical case
		if apex is None:
			if ZL is None:
				return [{"label": self.mouth_label, "U": U_in, "S": self.S_mouth}]
			Zc = Z0 / self.S_throat
			c = np.cos(k * self.L)
			s = np.sin(k * self.L)
			den = c + 1j * (ZL / Zc) * s
			den = np.where(np.abs(den) < 1e-24, 1e-24 + 0j, den)
			H = 1.0 / den
			Ui = H * U_in
			return [{"label": self.mouth_label, "U": Ui, "S": self.S_mouth}]

		# Conical with load
		if ZL is not None:
			r1, r2, G = apex
			BA = self._BA_ratio(k, r2, G, ZL)
			U1 = self._U(r1, k, G, BA)
			U2 = self._U(r2, k, G, BA)
			H_umouth = U2 / U1
			Ui = H_umouth * U_in
			return [{"label": self.mouth_label, "U": Ui, "S": self.S_mouth}]

		# Conical but no ZL known
		r1, r2, _ = apex
		H_geom = (r2 / r1) ** 2 * np.exp(-1j * k * (r2 - r1))
		Ui = H_geom * U_in
		return [{"label": self.mouth_label, "U": Ui, "S": self.S_mouth}]


