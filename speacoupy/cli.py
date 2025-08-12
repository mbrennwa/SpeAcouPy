
from __future__ import annotations
import argparse, os
from typing import Any, Dict
import numpy as np
import yaml
from .radiators import collect_radiators

from . import (
	omega_logspace, Net, Series, Parallel,
	Ce, Le, Re, Driver,
	DriverMechanicalBranch, Port, SealedBox, VentedBox, RadiationPiston,
)
from .plotting import plot_spl, plot_impedance, plot_spl_multi
from .response import ResponseSolver

def _validate_linking(registry: dict, global_space: str):
		"""Validate that all radiators are explicitly linked to 'radiation_space' or another element.
		Rules:
		- Driver.front_load/back_load must be provided and be either 'radiation_space' or an existing element label/object.
		- VentedBox must have 'port_load' defined; if string, it must be 'radiation_space' or an existing label.
		"""
		from .driver import Driver as DriverClass
		# Validate drivers
		drivers = [v for v in registry.values() if isinstance(v, DriverClass)]
		for drv in drivers:
			mot = getattr(drv, 'motional', None)
			if mot is None:
			raise ValueError(f"Driver '{getattr(drv,'label','driver')}' has no motional branch")

			for side_name in ('front_load','back_load'):
				ld = getattr(mot, side_name, None)
				if ld is None:
				raise ValueError(f"Driver '{getattr(drv,'label','driver')}' missing {side_name}: must be 'radiation_space' or another element label")

				if isinstance(ld, str):
					if ld != 'radiation_space' and ld not in registry:
					raise ValueError(f"Driver '{getattr(drv,'label','driver')}' {side_name} references unknown label '{ld}'")

		# Validate vented boxes / ports
		for lbl, obj in registry.items():
			if obj.__class__.__name__ == 'VentedBox':
				pl = getattr(obj, 'port_load', None)
				if pl is None:
				raise ValueError(f"VentedBox '{lbl}' requires 'port_load' to be set to 'radiation_space' or another element label")

				if isinstance(pl, str) and (pl != 'radiation_space' and pl not in registry):
				raise ValueError(f"VentedBox '{lbl}' port_load references unknown label '{pl}'")



# ---------------- YAML ----------------
def load_config(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return yaml.safe_load(f)

# ---------------- Element builders ----------------
def build_acoustic(spec: Dict[str, Any], Sd: float):
	t = (spec.get("type") or "").lower()
	if t in ("sealed","sealed_box"):
		Vb = float(spec["Vb"])  # m^3 (MKS)
		return SealedBox(Vb=Vb)
	if t in ("vented","vented_box","bass_reflex"):
		Vb = float(spec["Vb"])  # m^3 (MKS)
		d  = float(spec["port_d_m"])
		L  = float(spec["port_L_m"])
		vb = VentedBox(Vb=Vb, port=Port(diameter=d, length=L))
		vb.port_load = spec.get('port_load')
		return vb
	if t in ("piston","piston_wideband","radiation"):
	return RadiationPiston(Sd=Sd)
	raise ValueError(f"Unknown acoustic element type: {t}")

def build_registry(cfg: dict):
	"""
	Two-pass build:
	1) create all non-driver elements from 'elements: [{type,label,...}]'
	2) create drivers (need references to front/back loads by label)
	"""
	elems = cfg.get("elements") or []
	if not isinstance(elems, list) or not elems:
		raise ValueError("Config must define a non-empty 'elements' list.")
	labels = [e.get("label") for e in elems]
	if any(l is None for l in labels):
		raise ValueError("Each element must have a 'label'.")
	if len(set(labels)) != len(labels):
		raise ValueError("All element labels must be unique.")

	reg: Dict[str, Any] = {}
	pending_drivers = []

	# First pass: non-driver elements
	for e in elems:
		typ = (e.get("type") or "").lower()
		lab = e["label"]
		if typ in ("re","resistor"):
			reg[lab] = Re(R=float(e["Re"] if "Re" in e else e["R"]))
		elif typ in ("le","inductor"):
			reg[lab] = Le(L=float(e["Le"] if "Le" in e else e["L"]))
		elif typ in ("ce","capacitor"):
			reg[lab] = Ce(C=float(e["C"]))
		elif typ in ("sealed","vented","piston","piston_wideband","radiation","sealed_box","vented_box","bass_reflex"):
			Sd_hint = float(e.get("Sd", e.get("Sd", 0.053)))
			reg[lab] = build_acoustic(e, Sd=Sd_hint)
		elif typ == "driver":
			pending_drivers.append(e)
		else:
			raise ValueError(f"Unknown element type: {typ} (label={lab})")

	# Second pass: drivers (MKS inputs: Sd[m^2], fs[Hz], Vas[m^3], Qms, Qes)
	RHO0 = 1.204   # kg/m^3
	C_AIR = 343.0  # m/s
	TWOPI = 2.0 * np.pi

	
		# Resolve port_load for VentedBox/Port (must be 'radiation_space' or another element label)
		from .acoustic import RadiationPiston
		import math
		for lbl, obj in list(reg.items()):
			port_load = getattr(obj, 'port_load', None)
			if port_load is None:
				continue
			# Auto mouth radiator Sd from port geometry
			if obj.__class__.__name__ == 'VentedBox':
				port = getattr(obj, 'port', None)
				if port is None or not hasattr(port, 'diameter'):
			raise ValueError(f"VentedBox '{lbl}' needs port geometry to derive mouth Sd")
		Sd_port = math.pi * (port.diameter * 0.5)**2
			if port_load == 'radiation_space':
				obj.mouth_radiator = RadiationPiston(Sd=Sd_port, loading=global_space)
			else:
				# Link to another element by label
				if port_load not in reg:
					raise ValueError(f"Element '{lbl}' references unknown port_load '{port_load}'")
				obj.mouth_radiator = reg[port_load]
for e in pending_drivers:
		lab = e["label"]
		Sd = float(e["Sd"])          # m^2
		fs = float(e["fs"])          # Hz
		Vas = float(e["Vas"])        # m^3
		Qms = float(e["Qms"])
		Qes = float(e["Qes"])
		Re_val = float(e.get("Re", 6.0))
		Le_val = float(e.get("Le", 0.0))

		front_lab = e.get("front_load")
		back_lab  = e.get("back_load")
		if not front_lab or not back_lab:
			raise ValueError(f"Driver '{lab}' must specify front_load and back_load labels.")
		if front_lab not in reg or back_lab not in reg:
			raise ValueError(f"Driver '{lab}' references unknown loads: front_load={front_lab}, back_load={back_lab}")

		omega_s = TWOPI * fs
		Cms = Vas / (RHO0 * (C_AIR**2) * (Sd**2))
		Mms = 1.0 / (omega_s**2 * Cms)
		Rms = (omega_s * Mms) / Qms
		Bl  = np.sqrt(omega_s * Mms * Re_val / Qes)

		mot = DriverMechanicalBranch(
			Rms_val=Rms,
			Mms_val=Mms,
			Cms_val=Cms,
			front_load=reg[front_lab],
			back_load=reg[back_lab],
			Sd=Sd,
		)
		drv = Driver(
			Re_val=Re_val,
			Le_val=Le_val,
			Bl=Bl,
			motional=mot,
		)
		reg[lab] = drv
		# Auto-bind top-level radiation_space for driver sides
		from .acoustic import RadiationPiston
		if isinstance(mot.front_load, str) and mot.front_load == 'radiation_space':
			mot.front_load = RadiationPiston(Sd=Sd, loading=global_space)
		if isinstance(mot.back_load, str) and mot.back_load == 'radiation_space':
			mot.back_load = RadiationPiston(Sd=Sd, loading=global_space)

	# Find exactly one driver
	from .driver import Driver as DriverClass
	drivers = [k for k,v in reg.items() if isinstance(v, DriverClass)]
	if len(drivers) != 1:
		raise ValueError(f"Expected exactly one driver, found: {drivers}")
	from .acoustic import RadiationPiston
	for v in reg.values():
		if isinstance(v, RadiationPiston):
			setattr(v, 'loading', global_space)
	return reg, drivers[0]

def build_net(node, reg):
	if isinstance(node, str):
		if node not in reg:
			raise ValueError(f"Unknown element label in network: {node}")
		return reg[node]
	if isinstance(node, list):
		return Series(parts=[build_net(x, reg) for x in node])
	if not isinstance(node, dict):
		raise ValueError(f"Network node must be a label, list, or dict; got {type(node)}")
	if "series" in node:
		return Series(parts=[build_net(x, reg) for x in node["series"]])
	if "parallel" in node:
		return Parallel(parts=[build_net(x, reg) for x in node["parallel"]])
	raise ValueError(f"Network dict must have 'series' or 'parallel' key; got {node}")


def _get_global_rspace(cfg: dict) -> str:
	val = str(cfg.get("radiation_space", "")).strip().lower()
	if val == "0.5pi":
		val = "1/2pi"
	allowed = {"4pi","2pi","1pi","1/2pi"}
	if val not in allowed:
		raise ValueError("Top-level 'radiation_space' is required and must be one of: 4pi, 2pi, 1pi, 1/2pi")
	return val


def build_system(cfg: dict):
	global_space = _get_global_rspace(cfg)
	fmin = float(cfg.get("frequency", {}).get("min_hz", 10.0))
	fmax = float(cfg.get("frequency", {}).get("max_hz", 20000.0))
	npts = int(cfg.get("frequency", {}).get("points", 1200))
	f, w = omega_logspace(fmin, fmax, npts)

	reg, drv_label = build_registry(cfg)
	net_spec = cfg.get("network")
	if not net_spec:
		raise ValueError("Config must define 'network' using labels.")
	net = build_net(net_spec, reg)

	Vsrc = float(cfg.get("source", {}).get("volts_rms", 2.83))
	r = float(cfg.get("source", {}).get("distance_m", 1.0))
	angles = cfg.get("angles_deg")
	angles = np.array(angles, dtype=float) if angles else None

	loading_label = global_space
	from .driver import Driver as DriverClass
	for _, v in reg.items():
		if isinstance(v, DriverClass):
			loading_label = getattr(v.motional.front_load, "loading", "4pi")
			drv = v
			break

	return f, w, net, drv, drv.motional.Sd, Vsrc, r, loading_label, angles

def main(argv=None):
	parser = argparse.ArgumentParser(prog='speacoupy', description="SpeAcouPy: MKS driver inputs (Sd, fs, Vas, Qms, Qes)")
	parser.add_argument("config", help="YAML config file")
	parser.add_argument("--outdir", default="plots", help="Output directory for plots")
	parser.add_argument("--prefix", default="", help="Filename prefix")
	args = parser.parse_args(argv)

	cfg = load_config(args.config)
	f, w, net, drv, Sd, Vsrc, r, loading_label, angles = build_system(cfg)

	solver = ResponseSolver(series_net=net, driver=drv, Sd=Sd)
	res = solver.solve(w, V_source=Vsrc, r=r, loading=loading_label, angles_deg=angles)

	os.makedirs(args.outdir, exist_ok=True)
	tag = loading_label.replace("/", "")
	pre = (args.prefix + "_") if args.prefix else ""

	plot_spl(res.f, res.SPL_onaxis, outfile=os.path.join(args.outdir, f"{pre}spl_{tag}.png"),
			title=f"On-axis SPL (1 m) [{loading_label}]")
	plot_impedance(res.f, res.Zin, outfile=os.path.join(args.outdir, f"{pre}impedance_{tag}.png"),
				title=f"Input Impedance Magnitude [{loading_label}]")
	if res.SPL_offaxis is not None and res.angles_deg is not None:
		curves = [res.SPL_offaxis[i] for i in range(len(res.angles_deg))]
		labels = [f"{ang:.0f}°" for ang in res.angles_deg]
		from .plotting import plot_spl_multi
		plot_spl_multi(res.f, curves, labels,
					outfile=os.path.join(args.outdir, f"{pre}spl_angles_{tag}.png"),
					title=f"SPL vs angle [{loading_label}]")
	print(f"Wrote plots to {args.outdir}/ with loading tag [{loading_label}]")
	return 0

if __name__ == "__main__":
	raise SystemExit(main())

