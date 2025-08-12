import numpy as np
from .acoustic import RHO0, P0
from .directivity import piston_directivity

class ResponseResult:
	def __init__(self, f, p_total, p_by_radiator, Zin, Vd, Id, v, U, SPL, SPL_offaxis):
		self.f = f
		self.p_total = p_total
		self.p_by_radiator = p_by_radiator
		self.Zin = Zin
		self.Vd = Vd
		self.Id = Id
		self.v = v
		self.U = U
		self.SPL = SPL
		self.SPL_offaxis = SPL_offaxis

class ResponseSolver:
	def __init__(self, series_net, driver, Sd, radiation_space, angles):
		self.series_net = series_net
		self.driver = driver
		self.Sd = Sd
		self.radiation_space = radiation_space
		self.angles = angles

	def solve(self, omega, Vsrc, r):
		Z_total = self.series_net.Z(omega)
		Id = Vsrc / Z_total
		v = {}
		U = {}

		# Calculate velocities for each element
		for el in self.series_net.elements_flat():
			if hasattr(el, "velocity"):
				v[el.label] = el.velocity(Id, omega)
			if hasattr(el, "volume_velocity"):
				U[el.label] = el.volume_velocity(Id, omega)

		# Sum radiator outputs with polarity
		p_total, p_by_radiator = self._sum_radiators(omega, U, r, self.radiation_space)

		# SPL from total summed pressure
		SPL = 20 * np.log10(np.maximum(np.abs(p_total), 1e-16) / P0)

		# Off-axis SPL
		D = piston_directivity(self.Sd, omega, self.angles)  # shape (n_angles, n_freq)
		p_ang = D * p_total.reshape((1, -1))
		SPL_offaxis = 20 * np.log10(np.maximum(np.abs(p_ang), 1e-16) / P0)

		return ResponseResult(
			f=omega / (2 * np.pi),
			p_total=p_total,
			p_by_radiator=p_by_radiator,
			Zin=Z_total,
			Vd=None,
			Id=Id,
			v=v,
			U=U,
			SPL=SPL,
			SPL_offaxis=SPL_offaxis
		)

	def _sum_radiators(self, omega, U_map, r, loading):
		# Map loading space factor
		k_map = {"4pi": 1.0, "2pi": 2.0, "1pi": 4.0, "1/2pi": 8.0, "0.5pi": 8.0}
		k = k_map.get((loading or "4pi").lower(), 1.0)

		p_total = np.zeros_like(omega, dtype=complex)
		p_by_radiator = {}

		for el in self.series_net.elements_flat():
			if hasattr(el, "is_radiator") and el.is_radiator:
				U_val = U_map.get(el.label)
				if U_val is None:
					continue
				polarity = getattr(el, "polarity", +1)
				p = polarity * k * (1j * omega * RHO0 * U_val / (4 * np.pi * r))
				p_by_radiator[el.label] = p
				p_total += p

		return p_total, p_by_radiator
