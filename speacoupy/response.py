
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .radiators import collect_radiators
from .domains import Element
from .driver import Driver
from .acoustic import RHO0, P0, piston_directivity

@dataclass
class ResponseResult:
	f: np.ndarray
	Zin: np.ndarray
	Vd: np.ndarray
	Id: np.ndarray
	v: np.ndarray
	U: np.ndarray
	p_onaxis: np.ndarray
	SPL_onaxis: np.ndarray
	angles_deg: np.ndarray | None = None
	SPL_offaxis: np.ndarray | None = None  # shape (n_angles, n_freq)

def omega_logspace(fmin=10.0, fmax=20000.0, n=1000):
	f = np.logspace(np.log10(fmin), np.log10(fmax), int(n))
	return f, 2*np.pi*f

class ResponseSolver:

	# helper: sum radiation of terminal radiators
	def _sum_radiators(self, omega, U, r, loading: str, include_labels=None):
		"""Sum terminal radiators with correct signs; optional filtering by include_labels."""
		include_set = set(include_labels) if include_labels else None

		# Front/back flows for a rigid piston
		U_front_total =  U
		U_back_total  = -U

		from .radiators import collect_radiators
		front_rads = collect_radiators(self.driver.motional.front_load)
		back_rads  = collect_radiators(self.driver.motional.back_load)

		# helper: determine weights of volume-flow splitting if radiation splits into multiple, parallel elements:
		def _weights(rad_list):
			if not rad_list:
				return []

			Ys = [] # array of admittance coefficients
			for _, el in rad_list:
				try:
					Zr = el.Z_rad(omega) if hasattr(el, "Z_rad") else el.impedance(omega)
				except Exception:
					# effectively open-circuit: near-zero admittance
					Zr = 1e30 + 0j
				Y = 1.0 / np.maximum(np.abs(Zr), 1e-30)   # admittance magnitude, shape (F,)
				Ys.append(Y)

			W = np.vstack(Ys)                            # shape (N_radiators, F)
			S = np.sum(W, axis=0)                        # sum across radiators -> shape (F,)

			# Avoid divide-by-zero: if all admittances are ~0 at some freq, split equally
			zero = S <= 1e-30
			if np.any(zero):
				W[:, zero] = 1.0
				S[zero] = W.shape[0]

			W = W / S                                    # each column now sums to 1
			return W

		# Enforce explicit labels on terminal radiators
		for lbl, _ in (front_rads + back_rads):
			if lbl is None or (isinstance(lbl, str) and lbl.strip()==""):
				raise ValueError('All terminal radiators must have explicit, non-empty labels in the config.')

		# Strict validation for include set
		if include_set is not None:
			all_labels = {lbl for (lbl, _) in (front_rads + back_rads)}
			missing = sorted(list(include_set - all_labels))
			if missing:
				raise ValueError('Requested radiator(s) not found or not terminal: ' + ', '.join(missing))

		Wf = _weights(front_rads)
		Wb = _weights(back_rads)
		
		k_map = {'4pi':1.0,'2pi':2.0,'1pi':4.0,'1/2pi':8.0,'0.5pi':8.0}
		k = k_map.get((loading or '4pi').lower(), 1.0)

		p_total = np.zeros_like(omega, dtype=complex)
		p_by_radiator = {}

		for i, (lbl, el) in enumerate(front_rads):
			if include_set is not None and lbl not in include_set:
				continue
			Ui = U_front_total * (Wf[i] if len(Wf) else 1.0)
			pi = k * (1j*omega*RHO0*Ui) / (4*np.pi*r)
			p_by_radiator[lbl] = pi
			p_total += pi

		for i, (lbl, el) in enumerate(back_rads):
			if include_set is not None and lbl not in include_set:
				continue
			Ui = U_back_total * (Wb[i] if len(Wb) else 1.0)
			pi = k * (1j*omega*RHO0*Ui) / (4*np.pi*r)
			p_by_radiator[lbl] = pi
			p_total += pi

		return p_total, p_by_radiator

	def __init__(self, series_net: Element, driver: Driver, Sd: float):
		self.series = series_net
		self.driver = driver
		
	def solve(self, omega: np.ndarray, V_source: float = 2.83, r: float = 1.0,
			loading: str = "4pi", angles_deg: np.ndarray | None = None, include_radiators=None) -> ResponseResult:
			
		Z_total = self.series.impedance(omega)			# total system el. impedance
		Z_driver = self.driver.impedance(omega)			# driver el. impedance (total impedance)
		Vd = V_source * (Z_driver / Z_total)			# voltage at driver terminals
		Id = Vd / Z_driver 					# driver motor current
		Zvc = self.driver.impedance_voicecoil(omega)		# driver voice-coil impedance (without motional part)
		Zm = (self.driver.Bl**2) / (Z_driver - Zvc)		# motional impedance in mechanical domain
		v = (self.driver.Bl * Id) / Zm				# cone velocity
		U = v * self.driver.Sd()				# volume flow at (front of) driver cone
	
		# General per-radiator summation
		p_total, p_by_radiator = self._sum_radiators(omega, U, r, loading, include_labels=include_radiators)
		
		k_map = {"4pi": 1.0, "2pi": 2.0, "1pi": 4.0, "1/2pi": 8.0, "0.5pi": 8.0}
		k = k_map.get((loading or "4pi").lower(), 1.0)
		SPL_onax = 20*np.log10(np.maximum(np.abs(p_total), 1e-16) / P0)
		f = omega / (2*np.pi)

		SPL_offax = None
		if angles_deg is not None and angles_deg.size > 0:
			th = np.deg2rad(angles_deg)
			### D = piston_directivity(self.Sd, omega, th)  # shape (n_angles, n_freq)
			D = piston_directivity(self.driver.Sd(), omega, th)  # shape (n_angles, n_freq)
			p_ang = (D * p_total.reshape((1,-1)))
			SPL_off = 20*np.log10(np.maximum(np.abs(p_ang), 1e-16) / P0)

		return ResponseResult(f=f, Zin=Z_total, Vd=Vd, Id=Id, v=v, U=U,
							p_onaxis=p_total, SPL_onaxis=SPL_onax,
							angles_deg=angles_deg, SPL_offaxis=SPL_offax)
