
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

	def _sum_radiators(self, omega, U, r, loading: str):
		"""Split diaphragm flow into front/back and sum terminal radiators with correct polarity."""
		# Constants
		#### NJET!! RHO0 = self.RHO0 if hasattr(self, "RHO0") else 1.204
		
		# Equivalent loads seen by diaphragm
		Zf = self.driver.motional.front_load.impedance(omega)
		Zb = self.driver.motional.back_load.impedance(omega)
		Zsum = Zf + Zb
		Zsum = np.where(np.abs(Zsum) == 0, 1e-30, Zsum)

		U_front_total = U * (Zb / Zsum)
		U_back_total  = U * (Zf / Zsum)

		front_rads = collect_radiators(self.driver.motional.front_load, +1)
		back_rads  = collect_radiators(self.driver.motional.back_load, -1)

		def _weights(rad_list):

			if not rad_list:

				return []

			W = []

			for _, el, _ in rad_list:

				try:

					Zr = el.Z_rad(omega) if hasattr(el, "Z_rad") else el.impedance(omega)

				except Exception:

					Zr = 1e30 + 0j

				W.append(1.0/np.maximum(np.abs(Zr), 1e-30))

			W = np.array(W, dtype=float)

			S = np.sum(W)

			if not np.isfinite(S) or S == 0:

				W = np.ones(len(rad_list), dtype=float)

				S = np.sum(W)

			return (W / S)

		Wf = _weights(front_rads)
		Wb = _weights(back_rads)

		k_map = {"4pi": 1.0, "2pi": 2.0, "1pi": 4.0, "1/2pi": 8.0, "0.5pi": 8.0}
		k = k_map.get((loading or "4pi").lower(), 1.0)

		p_total = np.zeros_like(omega, dtype=complex)
		p_by_radiator = {}

		for i, (lbl, el, pol) in enumerate(front_rads):

			Ui = U_front_total * (Wf[i] if len(Wf) else 1.0)
			pi = pol * k * (1j * omega * RHO0 * Ui) / (4*np.pi*r)
			p_by_radiator[lbl] = pi
			p_total += pi

		for i, (lbl, el, pol) in enumerate(back_rads):

			Ui = U_back_total * (Wb[i] if len(Wb) else 1.0)
			pi = pol * k * (1j * omega * RHO0 * Ui) / (4*np.pi*r)
			p_by_radiator[lbl] = pi
			p_total += pi

		return p_total, p_by_radiator
	def __init__(self, series_net: Element, driver: Driver, Sd: float):
		self.series = series_net
		self.driver = driver
		
	def solve(self, omega: np.ndarray, V_source: float = 2.83, r: float = 1.0,
			loading: str = "4pi", angles_deg: np.ndarray | None = None) -> ResponseResult:
			
			
		print('**** the solver needs careful checking!!! especially the _sum_radiators(...) thing is unclear to me!')
			
			
		Z_total = self.series.impedance(omega)			# total system el. impedance
		Z_driver = self.driver.impedance(omega)			# driver el. impedance (total impedance)
		Vd = V_source * (Z_driver / Z_total)			# voltage at driver terminals
		Id = Vd / Z_driver 					# driver motor current
		Zvc = self.driver.impedance_voicecoil(omega)		# driver voice-coil impedance (without motional part)
		Zm = (self.driver.Bl**2) / (Z_driver - Zvc)		# motional mech. impedance
		v = (self.driver.Bl * Id) / Zm				# cone velocity
		U = v * self.driver.Sd()				# volume flow at (front of) driver cone
	
		# General per-radiator summation
		p_total, p_by_radiator = self._sum_radiators(omega, U, r, loading)

		k_map = {"4pi": 1.0, "2pi": 2.0, "1pi": 4.0, "1/2pi": 8.0, "0.5pi": 8.0}
		k = k_map.get((loading or "4pi").lower(), 1.0)
		p = k * (1j * omega * RHO0 * U / (4*np.pi*r))
		SPL = 20*np.log10(np.maximum(np.abs(p), 1e-16) / P0)
		f = omega / (2*np.pi)

		SPL_off = None
		if angles_deg is not None and angles_deg.size > 0:
			th = np.deg2rad(angles_deg)
			### D = piston_directivity(self.Sd, omega, th)  # shape (n_angles, n_freq)
			D = piston_directivity(self.driver.Sd(), omega, th)  # shape (n_angles, n_freq)
			p_ang = (D * p.reshape((1,-1)))
			SPL_off = 20*np.log10(np.maximum(np.abs(p_ang), 1e-16) / P0)

		return ResponseResult(f=f, Zin=Z_total, Vd=Vd, Id=Id, v=v, U=U,
							p_onaxis=p, SPL_onaxis=SPL,
							angles_deg=angles_deg, SPL_offaxis=SPL_off)
