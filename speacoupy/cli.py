from __future__ import annotations
import argparse, os
from typing import Any, Dict
import numpy as np
import yaml
from pathlib import Path

from . import (
	omega_logspace, Net, Series, Parallel,
	Ce, Le, Re, Driver,
	DriverMechanicalBranch, Port, SealedBox, VentedBox, RadiationPiston,
)
from .plotting import plot_spl, plot_impedance, plot_spl_multi
from .response import ResponseSolver
import pandas as pd
from .constants import PROGRAMNAME
import datetime, pytz, csv

def _get_version_from_pyproject() -> str:
	try:
		from importlib.resources import files
		pyproj = files(__package__).joinpath("../pyproject.toml")
		text = pyproj.read_text(encoding="utf-8")
		import re
		m = re.search(r'^version\s*=\s*"(.*?)"\s*$', text, re.M)
		return m.group(1) if m else "unknown"
	except Exception:
		return "unknown"
def write_response_csv(res, outdir: str, pre: str, loading_label: str):
	outpath = os.path.join(outdir, f"{pre}DATA.csv")
	# Build columns
	data = {
		"frequency_hz": res.f,
		"Zin_mag_ohm": np.abs(res.Zin),
		"Zin_phase_deg": ((np.rad2deg(np.unwrap(np.angle(res.Zin))) + 180.0) % 360.0) - 180.0,
		"SPL_onaxis_db": res.SPL_onaxis,
	}
	# Off-axis SPL if present
	if getattr(res, "SPL_offaxis", None) is not None and getattr(res, "angles_deg", None) is not None:
		# Ensure deterministic order by sorting by angle
		try:
			pairs = sorted(zip(res.angles_deg, res.SPL_offaxis), key=lambda p: p[0])
			for ang, arr in pairs:
				col = f"SPL_deg_{int(round(float(ang)))}_db"
				data[col] = arr
		except Exception:
			# fallback naive order
			for i, arr in enumerate(res.SPL_offaxis):
				data[f"SPL_offaxis_{i}_db"] = arr
	df = pd.DataFrame(data)
	# Rename primary columns to requested headers
	mapping = {
		"frequency_hz": "Frequency (Hz)",
		"Zin_mag_ohm": "Impedance Magnitude (Ω)",
		"Zin_phase_deg": "Impedance Phase (degrees)",
		"SPL_onaxis_db": f"SPL (dB-SPL @ 1m / {loading_label})",
	}
	df.rename(columns=mapping, inplace=True)
	# Header lines
	tz = pytz.timezone("Europe/Zurich")
	timestamp = datetime.datetime.now(tz).isoformat()
	version = _get_version_from_pyproject()
	header_lines = [
		f"# Program: {PROGRAMNAME}",
		f"# Version: {version}",
		f"# Generated: {timestamp}",
	]
	with open(outpath, "w", encoding="utf-8") as f:
		for line in header_lines:
			f.write(line + "\n")
	df.to_csv(outpath, mode="a", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")


from .constants import PROGRAMNAME, FARFIELD_DIST_M

from math import log
import numpy as np



def fit_semi_inductance(points, Re, Bl, Rms, Mms, Cms):
	"""Direct total-magnitude fit: find k>=0, 0<alpha<=1 minimizing
		sum_i (log| Re + k*(j*omega_i)^alpha + (Bl^2)/Zm(omega_i) | - log|Z_meas,i|)^2
	No phase data required.
	"""
	TWOPI = 2.0 * np.pi
	# Prepare arrays
	freq = []
	Zmeas = []
	for f, Zabs in points:
		f = float(f); Zabs = float(Zabs)
		if Zabs > 0:
			freq.append(f); Zmeas.append(Zabs)
	if len(freq) < 2:
		raise ValueError("Need at least two valid Z_hf points.")
	freq = np.array(freq, dtype=float)
	omega = TWOPI * freq
	Zmeas = np.array(Zmeas, dtype=float)
	# Motional complex term
	Zm = (Rms + 0j) + 1j*omega*Mms + 1.0/(1j*omega*Cms)
	Zmot = (Bl**2) / Zm
	# Search over alpha and k (1D inner search per alpha)
	def sse_for(k, alpha):
		Zhf = k * (1j*omega)**alpha
		Ztot = Re + Zhf + Zmot
		mag = np.abs(Ztot)
		return float(np.sum((np.log(mag) - np.log(Zmeas))**2))
	# Reasonable k bounds from data at highest frequency
	mag_resid = np.maximum(Zmeas - Re - np.abs(Zmot), 1e-9)
	wmax = float(np.max(omega))
	# k upper bound so that k*wmax^alpha is on the order of max residual
	def k_bounds(alpha):
		base = np.max(mag_resid)
		if base <= 0:
			base = np.max(Zmeas)
		kmax = base / (wmax**alpha + 1e-30)
		kmax = max(kmax, 1e-9)
		return (0.0, kmax*10.0)
	# Golden section 1D minimizer
	phi = 0.5 * (1 + 5**0.5)
	def minimize_k(alpha):
		lo, hi = k_bounds(alpha)
		# initialize interior points
		c = hi - (hi - lo)/phi
		d = lo + (hi - lo)/phi
		fc = sse_for(c, alpha)
		fd = sse_for(d, alpha)
		for _ in range(60):
			if fc > fd:
				lo = c
				c = d
				fc = fd
				d = lo + (hi - lo)/phi
				fd = sse_for(d, alpha)
			else:
				hi = d
				d = c
				fd = fc
				c = hi - (hi - lo)/phi
				fc = sse_for(c, alpha)
			if abs(hi - lo) <= 1e-12:
				break
		k_opt = (lo + hi)/2
		return k_opt, sse_for(k_opt, alpha)
	best = (None, None, float('inf'))  # (k, alpha, sse)
	# Coarse-to-fine alpha search
	for alpha in np.linspace(0.25, 1.0, 31):  # coarse
		k_opt, sse = minimize_k(alpha)
		if sse < best[2]:
			best = (k_opt, float(alpha), float(sse))
	# local refine around best alpha
	acenter = best[1]
	agrid = np.linspace(max(0.1, acenter-0.1), min(1.0, acenter+0.1), 21)
	for alpha in agrid:
		k_opt, sse = minimize_k(alpha)
		if sse < best[2]:
			best = (k_opt, float(alpha), float(sse))
	k, alpha = best[0], best[1]
	return float(k), float(alpha)


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
		# carry placeholder; resolve later
		if "port_load" in spec:
			setattr(vb, "port_load", spec.get("port_load"))
		return vb
	if t in ("piston","radiation"):
		loading = (spec.get("loading") or "4pi").lower()
		return RadiationPiston(Sd=Sd, loading=loading)
	raise ValueError(f"Unknown acoustic element type: {t}")

def build_registry(cfg: dict):
	"""
	Two-pass build:
	1) create all non-driver elements from 'elements: [{type,label,...}]'
	2) create drivers (need references to front/back loads by label or 'radiation_space' placeholder)
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
			reg[lab] = Re(R=float(e.get("Re", e.get("R"))))
		elif typ in ("le","inductor"):
			reg[lab] = Le(L=float(e.get("Le", e.get("L"))))
		elif typ in ("ce","capacitor"):
			reg[lab] = Ce(C=float(e["C"]))
		elif typ in ("sealed","vented","piston","radiation","sealed_box","vented_box","bass_reflex"):
			Sd_hint = float(e.get("Sd", 0.053))
			reg[lab] = build_acoustic(e, Sd=Sd_hint)
		elif typ == "driver":
			pending_drivers.append(e)
		else:
			raise ValueError(f"Unknown element type: {typ} (label={lab})")

	# Second pass: drivers (MKS inputs: Sd[m^2], fs[Hz], Vas[m^3], Qms, Qes)
	RHO0 = 1.204   # kg/m^3
	C_AIR = 343.0  # m/s
	TWOPI = 2.0 * np.pi

	for e in pending_drivers:
		lab = e["label"]
		Sd = float(e["Sd"])          # m^2
		fs = float(e["fs"])          # Hz
		Vas = float(e["Vas"])        # m^3
		Qms = float(e["Qms"])
		Qes = float(e["Qes"])
		Re_val = float(e.get("Re", 6.0))

		front_lab = e.get("front_load")
		back_lab  = e.get("back_load")
		if not front_lab or not back_lab:
			raise ValueError(f"Driver '{lab}' must specify front_load and back_load labels.")

		def _is_known_load(lbl: str) -> bool:
			return (isinstance(lbl, str) and lbl == 'radiation_space') or (lbl in reg)

		if not _is_known_load(front_lab) or not _is_known_load(back_lab):
			raise ValueError(f"Driver '{lab}' references unknown loads: front_load={front_lab}, back_load={back_lab}")

		# Keep placeholders; translate later in build_system
		front_load_obj = reg[front_lab] if front_lab in reg else 'radiation_space'
		back_load_obj  = reg[back_lab]  if back_lab  in reg else 'radiation_space'

		omega_s = TWOPI * fs
		Cms = Vas / (RHO0 * (C_AIR**2) * (Sd**2))
		Mms = 1.0 / (omega_s**2 * Cms)
		Rms = (omega_s * Mms) / Qms
		Bl  = np.sqrt(omega_s * Mms * Re_val / Qes)

		mot = DriverMechanicalBranch(
			Rms_val=Rms,
			Mms_val=Mms,
			Cms_val=Cms,
			front_load=front_load_obj,
			back_load=back_load_obj,
			Sd=Sd,
		)
		pts = e.get('Z_hf')
		if not pts:
			raise ValueError("Driver '%s' must provide Z_hf with at least two [f_hz, Zabs] pairs" % lab)
		# normalize points
		pts_norm = []
		for item in pts:
			if isinstance(item, (list, tuple)) and len(item)>=2:
				pts_norm.append([float(item[0]), float(item[1])])
			elif isinstance(item, dict):
				pts_norm.append([float(item.get('f_hz')), float(item.get('Z_ohm'))])
			else:
				raise ValueError("Invalid Z_hf point format: %r" % (item,))
		k_semi, alpha_semi = fit_semi_inductance(pts_norm, Re_val, Bl, Rms, Mms, Cms)
		drv = Driver(
			Re_val=Re_val,
			k_semi=k_semi,
			alpha_semi=alpha_semi,
			Bl=Bl,
			motional=mot,
		)
		# Propagate driver-embedded radiator labels from YAML onto Driver object
		if isinstance(e, dict):
			if 'front_radiator_label' in e:
				setattr(drv, 'front_radiator_label', e['front_radiator_label'])
			if 'back_radiator_label' in e:
				setattr(drv, 'back_radiator_label', e['back_radiator_label'])
		reg[lab] = drv

	# Find exactly one driver
	from .driver import Driver as DriverClass
	drivers = [k for k,v in reg.items() if isinstance(v, DriverClass)]
	if len(drivers) != 1:
		raise ValueError(f"Expected exactly one driver, found: {drivers}")
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
	val = str(cfg.get("radiation_space","")).strip().lower()
	if val == "0.5pi":
		val = "1/2pi"
	if val not in {"4pi","2pi","1pi","1/2pi"}:
		raise ValueError("Top-level 'radiation_space' must be one of: 4pi, 2pi, 1pi, 1/2pi")
	return val

def build_system(cfg: dict):
	fmin = float(cfg.get("frequency", {}).get("min_hz", 10.0))
	fmax = float(cfg.get("frequency", {}).get("max_hz", 20000.0))
	npts = int(cfg.get("frequency", {}).get("points", 1200))
	f, w = omega_logspace(fmin, fmax, npts)

	# Build registry first
	reg, drv_label = build_registry(cfg)

	# Top-level radiation space
	global_space = _get_global_rspace(cfg)

	# Resolve placeholders across registry
	from .acoustic import RadiationPiston
	from .driver import Driver as DriverClass
	import math

	# Driver front/back placeholders
	drv = None
	for v in reg.values():
		if isinstance(v, DriverClass):
			drv = v
			break
	if drv is None:
		raise ValueError("No driver found in registry.")
	Sd_drv = drv.motional.Sd
	if isinstance(drv.motional.front_load, str) and drv.motional.front_load == 'radiation_space':
		fr_lab = getattr(drv, 'front_radiator_label', None)
		if not isinstance(fr_lab, str) or not fr_lab.strip():
			raise ValueError("Driver must provide 'front_radiator_label' when front_load is 'radiation_space'.")
		p = RadiationPiston(Sd=Sd_drv, loading=global_space)
		setattr(p, 'label', fr_lab)
		drv.motional.front_load = p
	if isinstance(drv.motional.back_load, str) and drv.motional.back_load == 'radiation_space':
		br_lab = getattr(drv, 'back_radiator_label', None)
		if not isinstance(br_lab, str) or not br_lab.strip():
			raise ValueError("Driver must provide 'back_radiator_label' when back_load is 'radiation_space'.")
		p = RadiationPiston(Sd=Sd_drv, loading=global_space)
		setattr(p, 'label', br_lab)
		drv.motional.back_load = p
	if isinstance(drv.motional.front_load, str) and drv.motional.front_load == 'radiation_space':
		drv.motional.front_load = RadiationPiston(Sd=Sd_drv, loading=global_space)
	if isinstance(drv.motional.back_load, str) and drv.motional.back_load == 'radiation_space':
		drv.motional.back_load = RadiationPiston(Sd=Sd_drv, loading=global_space)

	# Vented box mouth
	for lbl, obj in list(reg.items()):
		if obj.__class__.__name__ == 'VentedBox':
			pl = getattr(obj, 'port_load', None)
			if pl is None:
				continue
			if isinstance(pl, str) and pl == 'radiation_space':
				port = getattr(obj, 'port', None)
				if port is None or not hasattr(port, 'diameter'):
					raise ValueError(f"VentedBox '{lbl}' needs port geometry to derive mouth Sd")
				Sd_port = math.pi * (0.5*port.diameter)**2
				setattr(obj, 'mouth_radiator', RadiationPiston(Sd=Sd_port, loading=global_space))
			elif isinstance(pl, str):
				if pl not in reg:
					raise ValueError(f"VentedBox '{lbl}' port_load references unknown label '{pl}'")
				setattr(obj, 'mouth_radiator', reg[pl])

	net_spec = cfg.get("network")
	if not net_spec:
		raise ValueError("Config must define 'network' using labels.")
	net = build_net(net_spec, reg)

	Vsrc = float(cfg.get("source", {}).get("volts_rms", 2.83))
	r = FARFIELD_DIST_M
	angles = cfg.get("angles_deg")
	angles = np.array(angles, dtype=float) if angles else None

	loading_label = global_space
	return f, w, net, drv, drv.motional.Sd, Vsrc, r, loading_label, angles

def main(argv=None):
	parser = argparse.ArgumentParser(prog='speacoupy', description=f"{PROGRAMNAME}: Simulation of loudspeaker systems using networks of electro-mechano-acoustical elements")
	parser.add_argument("config", help="YAML config file")
	parser.add_argument('--radiators', nargs='+', default=None,
		help='Terminal radiator labels to include in SPL (must match labeled terminal radiators in config)')
	parser.add_argument("--outdir", default=str(Path.cwd()), help="Output directory for plots")
	parser.add_argument("--prefix", default="", help="Filename prefix")
	parser.add_argument("--png", action="store_true", help="Write PNG plots")
	parser.add_argument("--pdf", action="store_true", help="Write PDF plots")
	parser.add_argument("--csv", action="store_true", help="Write combined CSV data file")
	args = parser.parse_args(argv)
	if not (args.png or args.pdf or args.csv):
		parser.error("You must specify at least one output format: --png, --pdf, --csv")

	cfg = load_config(args.config)
	f, w, net, drv, Sd, Vsrc, r, loading_label, angles = build_system(cfg)	

	solver = ResponseSolver(series_net=net, driver=drv, Sd=Sd)
	res = solver.solve(w, V_source=Vsrc, r=r, loading=loading_label, angles_deg=angles, include_radiators=args.radiators)

	os.makedirs(args.outdir, exist_ok=True)
	### tag = loading_label.replace("/", "")
	pre = (args.prefix + "_") if args.prefix else ""

	outputs = []
	for fmt, enabled in (("png", args.png), ("pdf", args.pdf)):
		if enabled:
			plot_spl(res.f, res.SPL_onaxis, outfile=os.path.join(args.outdir, f"{pre}SPL.{fmt}"),
					title=f"On-axis SPL (1 m / {loading_label})")
			plot_impedance(res.f, res.Zin, outfile=os.path.join(args.outdir, f"{pre}IMPEDANCE.{fmt}"),
						title=f"Input Impedance")
			if res.SPL_offaxis is not None and res.angles_deg is not None:
				curves = [res.SPL_offaxis[i] for i in range(len(res.angles_deg))]
				labels = [f"{ang:.0f}°" for ang in res.angles_deg]
				plot_spl_multi(res.f, curves, labels,
							outfile=os.path.join(args.outdir, f"{pre}SPL_angles.{fmt}"),
							title=f"SPL vs angle (1 m / {loading_label})")
			outputs.append(fmt.upper())
	# CSV output
	if args.csv:
		write_response_csv(res, args.outdir, pre, loading_label)
		outputs.append("CSV")
	print(f'Wrote: {", ".join(outputs)} to {args.outdir}/')
	return 0

if __name__ == "__main__":
	raise SystemExit(main())