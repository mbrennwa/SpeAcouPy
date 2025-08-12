
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
	def __init__(self, series_net: Element, driver: Driver, Sd: float):
	    self.series = series_net
	    self.driver = driver
	    self.Sd = Sd
	def solve(self, omega: np.ndarray, V_source: float = 2.83, r: float = 1.0,
	          loading: str = "4pi", angles_deg: np.ndarray | None = None) -> ResponseResult:
	    Z_total = self.series.impedance(omega)
	    Z_driver = self.driver.impedance(omega)
	    Vd = V_source * (Z_driver / Z_total)
	    Id = Vd / Z_driver
	    Zvc = self.driver.Re_val + 1j*omega*self.driver.Le_val
	    Zin_drv = Z_driver
	    Zm = (self.driver.Bl**2) / (Zin_drv - Zvc)
	    v = (self.driver.Bl * Id) / Zm
	    U = v * self.Sd

	    k_map = {"4pi": 1.0, "2pi": 2.0, "1pi": 4.0, "1/2pi": 8.0, "0.5pi": 8.0}
	    k = k_map.get((loading or "4pi").lower(), 1.0)
	    p = k * (1j * omega * RHO0 * U / (4*np.pi*r))
	    SPL = 20*np.log10(np.maximum(np.abs(p), 1e-16) / P0)
	    f = omega / (2*np.pi)

	    SPL_off = None
	    if angles_deg is not None and angles_deg.size > 0:
	        th = np.deg2rad(angles_deg)
	        D = piston_directivity(self.Sd, omega, th)  # shape (n_angles, n_freq)
	        p_ang = (D * p.reshape((1,-1)))
	        SPL_off = 20*np.log10(np.maximum(np.abs(p_ang), 1e-16) / P0)

	    return ResponseResult(f=f, Zin=Z_total, Vd=Vd, Id=Id, v=v, U=U,
	                          p_onaxis=p, SPL_onaxis=SPL,
	                          angles_deg=angles_deg, SPL_offaxis=SPL_off)



# For each driver -> run collect_radiators on front/back, sum SPL per CLI filter


def solve(self, drivers, f, selected_radiators=None):
	"""Solve system response for given drivers and frequency array."""
	omega = 2 * np.pi * f
	U_by_radiator = {}
	p_by_radiator = {}

	for drv in drivers:
	    # Front
	    if drv.front_load:
	        for lbl, rad, pol in collect_radiators(drv.front_load, +1):
	            U = rad.U(omega) * pol if hasattr(rad, 'U') else np.zeros_like(f, dtype=complex)
	            p = rad.p(omega) * pol if hasattr(rad, 'p') else np.zeros_like(f, dtype=complex)
	            U_by_radiator[lbl] = U_by_radiator.get(lbl, 0) + U
	            p_by_radiator[lbl] = p_by_radiator.get(lbl, 0) + p
	    # Back
	    if drv.back_load:
	        for lbl, rad, pol in collect_radiators(drv.back_load, -1):
	            U = rad.U(omega) * pol if hasattr(rad, 'U') else np.zeros_like(f, dtype=complex)
	            p = rad.p(omega) * pol if hasattr(rad, 'p') else np.zeros_like(f, dtype=complex)
	            U_by_radiator[lbl] = U_by_radiator.get(lbl, 0) + U
	            p_by_radiator[lbl] = p_by_radiator.get(lbl, 0) + p

	# Default: sum all radiators
	if not selected_radiators:
	    selected_radiators = list(p_by_radiator.keys())

	p_total = np.zeros_like(f, dtype=complex)
	for lbl in selected_radiators:
	    p_total += p_by_radiator[lbl]

	# Convert to SPL (assuming 1 m ref distance and 20 µPa reference)
	p_ref = 20e-6
	SPL_total = 20 * np.log10(np.abs(p_total) / p_ref + 1e-20)

	return {
	    'f': f,
	    'U_by_radiator': U_by_radiator,
	    'p_by_radiator': p_by_radiator,
	    'SPL_total': SPL_total
	}
