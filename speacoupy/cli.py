
from __future__ import annotations
import argparse, sys, os
import numpy as np
import yaml

from . import (
    omega_logspace, Series, Parallel,
    Ce, Le, Re, Driver,
    DriverMechanicalBranch, Port, SealedBox, VentedBox,
)
from .plotting import plot_spl, plot_impedance
from .response import ResponseSolver

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_acoustic_load(cfg: dict):
    box = cfg.get("box", {"type": "sealed", "Vb_l": 20.0})
    box_type = box.get("type", "sealed").lower()
    if box_type == "sealed":
        Vb = float(box.get("Vb_l", 20.0)) * 1e-3
        return SealedBox(Vb=Vb)
    elif box_type == "vented":
        Vb = float(box.get("Vb_l", 20.0)) * 1e-3
        port_d = float(box.get("port_d_m", 0.07))
        port_L = float(box.get("port_L_m", 0.12))
        return VentedBox(Vb=Vb, port=Port(diameter=port_d, length=port_L))
    else:
        raise ValueError(f"Unknown box type: {box_type}")

def build_driver(cfg: dict, load):
    drv_cfg = cfg["driver"]
    Sd = float(drv_cfg.get("Sd_m2", 0.053))
    mot = DriverMechanicalBranch(
        Rms_val=float(drv_cfg.get("Rms", 1.6)),
        Mms_val=float(drv_cfg.get("Mms", 0.020)),
        Cms_val=float(drv_cfg.get("Cms", 7.0e-4)),
        acoustic_load=load,
        Sd=Sd,
    )
    drv = Driver(
        Re_val=float(drv_cfg.get("Re", 5.7)),
        Le_val=float(drv_cfg.get("Le", 0.35e-3)),
        Bl=float(drv_cfg.get("Bl", 7.4)),
        motional=mot,
    )
    return drv, Sd

def make_elem(obj, drv):
    if isinstance(obj, list):
        return Series(parts=[make_elem(x, drv) for x in obj])
    if not isinstance(obj, dict):
        raise ValueError(f"Network item must be dict or list, got {type(obj)}: {obj}")
    if 'series' in obj:
        items = obj['series'] or []
        return Series(parts=[make_elem(x, drv) for x in items])
    if 'parallel' in obj:
        items = obj['parallel'] or []
        return Parallel(parts=[make_elem(x, drv) for x in items])

    if len(obj) != 1:
        raise ValueError(f"Component spec must have a single key, got: {obj}")
    (k, v), = obj.items()
    k = k.lower()
    v = v or {}
    if k == 're':
        return Re(R=float(v['R']))
    if k == 'le':
        return Le(L=float(v['L']))
    if k == 'ce':
        return Ce(C=float(v['C']))
    if k == 'driver':
        return drv
    raise ValueError(f"Unknown component type: {k}")

def build_network(cfg: dict, drv):
    if 'series_network' in cfg:
        elems = []
        for elem in cfg.get('series_network', []):
            typ = elem['type'].lower()
            if typ == 'ce':
                elems.append(Ce(C=float(elem['C'])))
            elif typ == 'le':
                elems.append(Le(L=float(elem['L'])))
            elif typ == 're':
                elems.append(Re(R=float(elem['R'])))
            elif typ == 'driver':
                elems.append(drv)
            else:
                raise ValueError(f"Unknown series element type: {typ}")
        if not any(isinstance(x, Driver) for x in elems):
            elems.append(drv)
        return Series(parts=elems)

    net_spec = cfg.get('network')
    if net_spec is None:
        return Series(parts=[drv])
    return make_elem(net_spec, drv)

def build_system(cfg: dict):
    freq = cfg.get("frequency", {})
    fmin = float(freq.get("min_hz", 10.0))
    fmax = float(freq.get("max_hz", 20000.0))
    npts = int(freq.get("points", 1200))
    f, w = omega_logspace(fmin, fmax, npts)

    load = build_acoustic_load(cfg)
    drv, Sd = build_driver(cfg, load)
    net = build_network(cfg, drv)

    src = cfg.get("source", {})
    Vsrc = float(src.get("volts_rms", 2.83))
    r = float(src.get("distance_m", 1.0))

    return f, w, net, drv, Sd, Vsrc, r

def main(argv=None):
    parser = argparse.ArgumentParser(description="SpeAcouPy: Loudspeaker SPL/Impedance plotter from YAML config (series/parallel)")
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--outdir", default="plots", help="Output directory for plots")
    parser.add_argument("--prefix", default="", help="Filename prefix")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    f, w, net, drv, Sd, Vsrc, r = build_system(cfg)

    solver = ResponseSolver(series_net=net, driver=drv, Sd=Sd)
    res = solver.solve(w, V_source=Vsrc, r=r)

    os.makedirs(args.outdir, exist_ok=True)
    pre = (args.prefix + "_") if args.prefix else ""
    spl_path = os.path.join(args.outdir, f"{pre}spl.png")
    z_path = os.path.join(args.outdir, f"{pre}impedance.png")

    plot_spl(res.f, res.SPL_onaxis, outfile=spl_path)
    plot_impedance(res.f, res.Zin, outfile=z_path)

    print(f"Wrote: {spl_path}")
    print(f"Wrote: {z_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
