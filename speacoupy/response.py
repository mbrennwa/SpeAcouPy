
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
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

	def _sum_radiators(self, omega, U, r, loading: str, include_labels=None):
		"""Sum radiator channels from the driver's front and back loads, symmetrically."""
		include_set = set(include_labels) if include_labels else None
		k_map = {'4pi':1.0,'2pi':2.0,'1pi':4.0,'1/2pi':8.0,'0.5pi':8.0}
		k = k_map.get((loading or '4pi').lower(), 1.0)

		channels = []

		# FRONT load: drive with +U
		front_load = getattr(self.driver.motional, 'front_load', None)
		if front_load is not None and hasattr(front_load, 'radiation_channels'):
			chs = front_load.radiation_channels(omega, U) or []
			for ch in chs:
				lbl = ch.get('label')
				Ui = ch.get('U')
				if lbl is None or Ui is None:
					continue
				if include_set and lbl not in include_set:
					continue
				channels.append((lbl, Ui))

		# BACK load: drive with -U (opposite phase)
		back_load = getattr(self.driver.motional, 'back_load', None)
		if back_load is not None and hasattr(back_load, 'radiation_channels'):
			chs = back_load.radiation_channels(omega, -U) or []
			for ch in chs:
				lbl = ch.get('label')
				Ui = ch.get('U')
				if lbl is None or Ui is None:
					continue
				if include_set and lbl not in include_set:
					continue
				channels.append((lbl, Ui))

		p_total = np.zeros_like(omega, dtype=complex)
		p_by_radiator = {}
		for lbl, Ui in channels:
			pi = k * (1j*omega*RHO0*Ui) / (4*np.pi*r)
			p_by_radiator[lbl] = pi
			p_total += pi
		return p_total, p_by_radiator

	def __init__(self, series_net: Element, driver: Driver, **kwargs):
		self.series = series_net
		self.driver = driver

	def solve(self, omega: np.ndarray, V_source: float = 2.83, r: float = 1.0,
			loading: str = '4pi', angles_deg: np.ndarray | None = None, include_radiators=None) -> ResponseResult:
		Z_total = self.series.impedance(omega)
		Z_driver = self.driver.impedance(omega)
		Vd = V_source * (Z_driver / Z_total)
		Id = Vd / Z_driver
			# Cone velocity from force balance: v = (Bl * Id) / Z_mech
		Z_mech = self.driver.motional.impedance(omega)
		v = (self.driver.Bl * Id) / np.maximum(Z_mech, 1e-30)
		U = self.driver.motional.Sd * v
		p_total, p_by_radiator = self._sum_radiators(omega, U, r, loading, include_labels=include_radiators)
		k_map = {'4pi':1.0,'2pi':2.0,'1pi':4.0,'1/2pi':8.0,'0.5pi':8.0}
		SPL_onax = 20*np.log10(np.maximum(np.abs(p_total), 1e-16) / P0)
		f = omega / (2*np.pi)
		SPL_offax = None
		if angles_deg is not None and angles_deg.size > 0:
			th = np.deg2rad(angles_deg)
			D = piston_directivity(self.driver.motional.Sd, omega, th)
			p_ang = (D * p_total.reshape((1,-1)))
			SPL_off = 20*np.log10(np.maximum(np.abs(p_ang), 1e-16) / P0)
			SPL_offax = SPL_off
		return ResponseResult(f=f, Zin=Z_total, Vd=Vd, Id=Id, v=v, U=U,
							p_onaxis=p_total, SPL_onaxis=SPL_onax,
							angles_deg=angles_deg, SPL_offaxis=SPL_offax)
