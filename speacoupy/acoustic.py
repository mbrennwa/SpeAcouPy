from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Optional, Any
import numpy as np
from scipy.special import j1, struve  # wideband piston needs Bessel/Struve
from .domains import Element, Domain

# Physical constants (room temp)
from .constants import RHO0, C0, P0


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


class Acoustic(Element):
	"""Base class for acoustic one-port elements."""
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
	mouth_load: Optional[Acoustic] = None
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
		return [ { "label": label, "U": U_port } ]


@dataclass
class SplitLoad(Acoustic):
	"""
	Represents one acoustic port feeding multiple downstream acoustic loads in parallel.
	Used to model T-junctions or multiple vents. All branch impedances are combined in parallel
	for the input impedance. Radiation channels are split by admittance.
	"""
	loads: list[Any]

	def __post_init__(self):
		if not self.loads or not isinstance(self.loads, list):
			raise ValueError("SplitLoad requires a non-empty list of loads")

	def impedance(self, omega: np.ndarray) -> np.ndarray:
		Y_total = np.zeros_like(omega, dtype=complex)
		for ld in self.loads:
			if hasattr(ld, "impedance"):
				Zi = ld.impedance(omega)
			elif isinstance(ld, (float, complex)):
				Zi = np.full_like(omega, complex(ld))
			else:
				raise TypeError(f"Invalid SplitLoad branch type: {type(ld)}")
			Zi = np.where(np.abs(Zi) < 1e-24, 1e-24+0j, Zi)
			Y_total = Y_total + 1.0 / Zi
		Y_total = np.where(np.abs(Y_total) < 1e-24, 1e-24+0j, Y_total)
		return 1.0 / Y_total

	def radiation_channels(self, omega: np.ndarray, U_in: Optional[np.ndarray] = None):
		if U_in is None:
			return []
		Zs = []
		Ys = []
		for ld in self.loads:
			if hasattr(ld, "impedance"):
				Zi = ld.impedance(omega)
			elif isinstance(ld, (float, complex)):
				Zi = np.full_like(omega, complex(ld))
			else:
				raise TypeError(f"Invalid SplitLoad branch type: {type(ld)}")
			Zi = np.where(np.abs(Zi) < 1e-24, 1e-24+0j, Zi)
			Zs.append(Zi)
			Ys.append(1.0 / Zi)
		Ytot = np.zeros_like(omega, dtype=complex)
		for Y in Ys:
			Ytot = Ytot + Y
		Ytot = np.where(np.abs(Ytot) < 1e-24, 1e-24+0j, Ytot)
		channels = []
		for ld, Y in zip(self.loads, Ys):
			Ui = (Y / Ytot) * U_in
			if hasattr(ld, "radiation_channels"):
				chs = ld.radiation_channels(omega, U_in=Ui) or []
				channels.extend(chs)
		return channels


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


@dataclass
class Horn(Acoustic):
	"""
	Conical horn element with exact spherical-wave interior and a
	profile-aware **self-consistent** mouth radiation model.

	Only the **conical** profile is supported in this revision.
	"""
	L: float
	S_throat: float
	S_mouth: float
	profile: str
	mouth_termination: str
	R_throat: float = 0.0
	R_mouth: float = 0.0
	throat_load: Optional[str] = None
	throat_label: Optional[str] = None
	mouth_load: Optional[str] = None
	mouth_label: Optional[str] = None
	label: str = ""

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

	def _mouth_termination(self) -> str:
		"""Map 'free'|'baffle' → loading string expected by radiation models."""
		term = (self.mouth_termination or '').strip().lower()
		if term == 'free':
			return '4pi'
		elif term == 'baffle':
			return '2pi'
		raise ValueError(f"Horn: invalid mouth_termination {term!r}, must be 'free' or 'baffle'")

	@staticmethod
	def _apex_distances(L: float, S1: float, S2: float) -> Optional[tuple[float, float, float]]:
		s1 = np.sqrt(S1)
		s2 = np.sqrt(S2)
		den = s2 - s1
		# Cylindrical limit → handled by TL formula
		if np.all(np.abs(den) <= 1e-12 * max(1.0, float(s1))):
			return None
		absden = np.abs(den)
		r1 = L * s1 / absden
		r2 = L * s2 / absden
		G = S1 / (r1 ** 2)
		return float(r1), float(r2), float(G)

	def _get_mouth_grid(self, Nr: int = 48, Nphi: int = 64):
		"""Axisymmetric aperture quadrature using **rings** and an azimuth integral."""
		a = float(np.sqrt(self.S_mouth / np.pi))
		grid = getattr(self, "_mouth_grid", None)
		if grid and grid.get("a") == a and grid.get("Nr") == Nr and grid.get("Nphi") == Nphi:
			return grid
		# Radial rings (midpoint rule)
		ri = (np.arange(Nr) + 0.5) * (a / Nr)
		dr = a / Nr
		wr = 2.0 * np.pi * ri * dr  # ring areas
		# Azimuthal quadrature
		phi = 2 * np.pi * (np.arange(Nphi) + 0.5) / Nphi
		wphi = np.full(Nphi, 2*np.pi / Nphi)
		grid = {"a": a, "Nr": Nr, "Nphi": Nphi, "ri": ri.astype(float), "dr": float(dr),
				"wr": wr.astype(float), "phi": phi.astype(float), "wphi": wphi.astype(float),
				"Area": float(self.S_mouth)}
		self._mouth_grid = grid
		return grid

	def _kernel_ring_avg(self, k: float, ri: np.ndarray, rj: np.ndarray, phi: np.ndarray, wphi: np.ndarray) -> np.ndarray:
		"""Azimuth-averaged Rayleigh kernel K_ij(k) for rings i,j."""
		Ri = ri[:, None, None]
		Rj = rj[None, :, None]
		Phi = phi[None, None, :]
		R = np.sqrt(Ri*Ri + Rj*Rj - 2.0*Ri*Rj*np.cos(Phi))
		# Avoid singularity for coincident points
		dr = self._mouth_grid["dr"] if hasattr(self, "_mouth_grid") else (ri.max()/ri.size)
		R = np.where(R < 1e-12, dr, R)
		Kphi = np.exp(-1j * k * R) / R
		Kij = (Kphi * wphi[None, None, :]).sum(axis=2) / (2*np.pi)
		return Kij

	def _pavg_from_u_ring(self, omega: float, u_ring: np.ndarray, grid: dict, loading: str) -> complex:
		"""Area-averaged aperture pressure from ring-averaged normal velocity."""
		k = omega / C0
		if k == 0:
			return 0.0
		fac = 1.0 / (2 * np.pi) if str(loading).lower().startswith("2") else 1.0 / (4 * np.pi)
		ri, wr, phi, wphi, Area = grid["ri"], grid["wr"], grid["phi"], grid["wphi"], grid["Area"]
		K = self._kernel_ring_avg(k, ri, ri, phi, wphi)
		p_ring = 1j * omega * RHO0 * fac * (K @ (u_ring * wr))
		p_avg = (wr @ p_ring) / Area
		return p_avg

	def _conical_mouth_Z_eff_iter(self, omega: np.ndarray, loading: str, n_iter: int = 2) -> np.ndarray:
		"""Self-consistent effective radiation impedance for a **conical** mouth via ring integral."""
		apex = self._apex_distances(self.L, self.S_throat, self.S_mouth)
		assert apex is not None
		r1, r2, G = apex
		k = omega / C0
		g = self._get_mouth_grid(Nr=48, Nphi=64)
		ri, wr, Area = g["ri"], g["wr"], g["Area"]
		# initialize with baffled piston for stability
		ZL = RadiationPiston(Sd=self.S_mouth, loading=loading).impedance(omega)
		for _ in range(max(1, int(n_iter))):
			BA = self._BA_ratio(k, r2, G, ZL)
			Z_eff = np.empty_like(omega, dtype=complex)
			for i, wv in enumerate(omega):
				ki = k[i]
				# Conical spherical-mode velocity on aperture plane (ring centers)
				r_loc = np.sqrt(r2 * r2 + ri * ri)
				phase_out = np.exp(-1j * ki * r_loc)
				t1 = -(1.0 + 1j * ki * r_loc)
				t2 = (1j * ki * r_loc - 1.0) * np.exp(2j * ki * r_loc)
				u_shape = phase_out * (t1 + BA[i] * t2) * (r2 / (r_loc ** 3))
				# Normalize area-average to 1 (use ring areas wr)
				u_avg = (wr @ u_shape) / Area
				if u_avg == 0:
					u_avg = 1.0 + 0j
				u_ring = u_shape / u_avg
				# Area-averaged pressure from ring kernel
				p_avg = self._pavg_from_u_ring(wv, u_ring, g, loading)
				# One-port: Z = p_avg / u_avg ; u_avg ≡ 1
				Z_eff[i] = p_avg
			ZL = Z_eff
		self._mouth_BA_cache = {"omega": omega, "BA": BA, "ZL": ZL}
		return ZL

	def _load_impedance_at_mouth(self, omega: np.ndarray) -> np.ndarray:
		Rm = float(self.R_mouth)
		if self.mouth_load == "rigid":
			ZL = np.full_like(omega, np.inf + 0j)
		elif self.mouth_load == "radiation_space":
			if abs(float(self.S_mouth) - float(self.S_throat)) <= 1e-12 * max(1.0, float(self.S_mouth)):
				# Cylindrical limit: use baffled/free piston model directly
				piston = RadiationPiston(Sd=float(self.S_mouth), loading=self._mouth_termination())
				ZL = piston.impedance(omega)
			else:
				ZL = self._conical_mouth_Z_eff_iter(omega, self._mouth_termination(), n_iter=2)
		elif isinstance(self.mouth_load, (float, complex)):
			ZL = np.full_like(omega, complex(self.mouth_load))
		elif hasattr(self.mouth_load, "impedance"):
			# Chained element (e.g., another Horn): use its input impedance as the load at our mouth
			ZL = self.mouth_load.impedance(omega)
		elif isinstance(self.mouth_load, str) and self.mouth_load:
			raise ValueError(f"Unresolved mouth_load '{self.mouth_load}'.")
		else:
			raise ValueError("Mouth load not specified for horn mouth.")
		return ZL + Rm

	# --------------------------- Core cone solution ------------------------- #
	def _BA_ratio(self, k: np.ndarray, r2: float, G: float, ZL: np.ndarray) -> np.ndarray:
		Z0 = RHO0 * C0
		num = (G * ZL * k * (r2 ** 2)) - 1j * (G * ZL * r2) - (Z0 * k)
		den = (G * ZL * k * (r2 ** 2)) + 1j * (G * ZL * r2) + (Z0 * k)
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
		if apex is None:
			# Cylindrical (transmission line) input impedance
			k0 = omega / C0
			Zc = Z0 / self.S_throat
			t = np.tan(k0 * self.L)
			den = Zc + 1j * ZL * t
			den = np.where(np.abs(den) < 1e-24, 1e-24 + 0j, den)
			return Zc * (ZL + 1j * Zc * t) / den
		r1, r2, G = apex
		k = omega / C0
		BA = self._BA_ratio(k, r2, G, ZL)
		p1 = self._P(r1, k, Z0, BA)
		U1 = self._U(r1, k, G, BA)
		return p1 / U1

	# ------------------------------ API methods ---------------------------- #
	def impedance(self, omega: np.ndarray, ZL_external: Optional[np.ndarray] = None) -> np.ndarray:
		ZL_local = self._load_impedance_at_mouth(omega)
		ZL = ZL_external if ZL_external is not None else ZL_local
		Zin = self._conical_input_impedance(omega, ZL)
		if self.R_throat:
			Zin = Zin + float(self.R_throat)
		return Zin

	def radiation_channels(self, omega: np.ndarray, U_in: Optional[np.ndarray] = None):
		# If there is no label and no downstream radiator, nothing to report
		if U_in is None:
			return []
		# Compute velocity at the mouth of THIS horn
		Z0 = RHO0 * C0
		apex = self._apex_distances(self.L, self.S_throat, self.S_mouth)
		if apex is None:
			# Uniform cylindrical tube transfer to the mouth (lossless):
			# U_mouth / U_in = 1 / (cos(kL) + j (ZL/Zc) sin(kL))
			k0 = omega / C0
			Zc = Z0 / self.S_throat
			s = np.sin(k0 * self.L)
			c = np.cos(k0 * self.L)
			ZL = self._load_impedance_at_mouth(omega)
			den = c + 1j * (ZL / Zc) * s
			eps = 1e-12 + 1e-9 * np.max(np.abs(den))
			den = np.where(np.abs(den) < eps, den + 1j * eps, den)
			H_umouth = 1.0 / den
			U_mouth = H_umouth * U_in
		else:
			r1, r2, G = apex
			k = omega / C0
			ZL = self._load_impedance_at_mouth(omega)
			BA = self._BA_ratio(k, r2, G, ZL)
			U1 = self._U(r1, k, G, BA)
			U2 = self._U(r2, k, G, BA)
			den = U1
			eps = (1e-12 + 1e-9 * np.max(np.abs(den)))
			den = np.where(np.abs(den) < eps, den + 1j * eps, den)
			H_umouth = U2 / den
			U_mouth = H_umouth * U_in

		# If the mouth loads another element (e.g., another Horn), DELEGATE radiation downstream
		if hasattr(self.mouth_load, "radiation_channels"):
			return self.mouth_load.radiation_channels(omega, U_in=U_mouth)

		# Otherwise, radiate here if we have a mouth label (i.e., this is the terminal horn)
		if not (self.mouth_label and isinstance(self.mouth_label, str)):
			return []
		return [{"label": self.mouth_label, "U": U_mouth, "S": self.S_mouth}]
