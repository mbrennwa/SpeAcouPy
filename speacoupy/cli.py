
from __future__ import annotations
import argparse, os
import numpy as np
import yaml

from . import (
    omega_logspace, Net, Series, Parallel,
    Ce, Le, Re, Driver,
    DriverMechanicalBranch, Port, SealedBox, VentedBox, RadiationPistonWB,
)
from .plotting import plot_spl, plot_impedance, plot_spl_multi
from .response import ResponseSolver

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_loading(s: str) -> str:
    s = (s or "4pi").lower()
    valid = {"4pi","2pi","1pi","1/2pi","0.5pi"}
    if s not in valid:
        raise ValueError(f"Invalid loading '{s}'. Choose one of {sorted(valid)}")
    return s

def build_acoustic_from_spec(spec: dict, Sd: float):
    typ = (spec.get("type") or "").lower()
    if typ in ("sealed","sealed_box","sealedbox"):
        Vb = float(spec["Vb_l"]) * 1e-3
        return SealedBox(Vb=Vb)
    if typ in ("vented","vented_box","ventedbox","bass_reflex"):
        Vb = float(spec["Vb_l"]) * 1e-3
        d  = float(spec["port_d_m"])
        L  = float(spec["port_L_m"])
        return VentedBox(Vb=Vb, port=Port(diameter=d, length=L))
    if typ in ("piston","piston_wideband","radiation"):
        loading = parse_loading(spec.get("loading","4pi"))
        return RadiationPistonWB(Sd=Sd, loading=loading)
    raise ValueError(f"Unknown acoustic load type: {typ}")

def build_driver(cfg: dict):
    drv_cfg = cfg["driver"]
    Sd = float(drv_cfg.get("Sd_m2", 0.053))
    if "front_load" not in drv_cfg or "back_load" not in drv_cfg:
        raise ValueError("driver.front_load and driver.back_load must be specified in the YAML.")
    front = build_acoustic_from_spec(drv_cfg["front_load"], Sd=Sd)
    back  = build_acoustic_from_spec(drv_cfg["back_load"],  Sd=Sd)

    mot = DriverMechanicalBranch(
        Rms_val=float(drv_cfg.get("Rms", 1.6)),
        Mms_val=float(drv_cfg.get("Mms", 0.020)),
        Cms_val=float(drv_cfg.get("Cms", 7.0e-4)),
        front_load=front,
        back_load=back,
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
    if "net" in obj:
        spec = obj["net"] or {}
        op = (spec.get("op") or "series").lower()
        parts = spec.get("parts") or []
        return Net(op=op, parts=[make_elem(x, drv) for x in parts])
    if "series" in obj:
        items = obj["series"] or []
        return Series(parts=[make_elem(x, drv) for x in items])
    if "parallel" in obj:
        items = obj["parallel"] or []
        return Parallel(parts=[make_elem(x, drv) for x in items])
    if len(obj) != 1:
        raise ValueError(f"Component spec must have a single key, got: {obj}")
    (k, v), = obj.items()
    k = k.lower(); v = v or {}
    if k == "re":
        return Re(R=float(v["R"]))
    if k == "le":
        return Le(L=float(v["L"]))
    if k == "ce":
        return Ce(C=float(v["C"]))
    if k == "driver":
        return drv
    raise ValueError(f"Unknown component type: {k}")

def build_network(cfg: dict, drv):
    if "series_network" in cfg:
        elems = []
        for elem in cfg.get("series_network", []):
            typ = elem["type"].lower()
            if typ == "ce":
                elems.append(Ce(C=float(elem["C"])))
            elif typ == "le":
                elems.append(Le(L=float(elem["L"])))
            elif typ == "re":
                elems.append(Re(R=float(elem["R"])))
            elif typ == "driver":
                elems.append(drv)
            else:
                raise ValueError(f"Unknown series element type: {typ}")
        if not any(getattr(x, "__class__", None).__name__ == "Driver" for x in elems):
            elems.append(drv)
        return Series(parts=elems)
    net_spec = cfg.get("network")
    if net_spec is None:
        return Series(parts=[drv])
    return make_elem(net_spec, drv)

def build_system(cfg: dict):
    freq = cfg.get("frequency", {})
    fmin = float(freq.get("min_hz", 10.0))
    fmax = float(freq.get("max_hz", 20000.0))
    npts = int(freq.get("points", 1200))
    f, w = omega_logspace(fmin, fmax, npts)

    drv, Sd = build_driver(cfg)
    net = build_network(cfg, drv)

    src = cfg.get("source", {})
    Vsrc = float(src.get("volts_rms", 2.83))
    r = float(src.get("distance_m", 1.0))

    angles = cfg.get("driver", {}).get("angles_deg", [])
    angles = np.array(angles, dtype=float) if angles else None

    # 'room.loading' for far-field scaling still used in titles/filenames
    loading = parse_loading(cfg.get("room",{}).get("loading", "4pi"))

    return f, w, net, drv, Sd, Vsrc, r, loading, angles

def main(argv=None):
    parser = argparse.ArgumentParser(description="SpeAcouPy: wideband piston + per-side loads + directivity + labeled outputs")
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--outdir", default="plots", help="Output directory for plots")
    parser.add_argument("--prefix", default="", help="Filename prefix")
    parser.add_argument("--angles", default="", help="Comma-separated list of angles in degrees (e.g., 0,15,30,45)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.angles:
        ang = [float(x) for x in args.angles.split(",") if x.strip()]
        cfg.setdefault("driver", {})["angles_deg"] = ang

    f, w, net, drv, Sd, Vsrc, r, loading, angles = build_system(cfg)

    solver = ResponseSolver(series_net=net, driver=drv, Sd=Sd)
    res = solver.solve(w, V_source=Vsrc, r=r, loading=loading, angles_deg=angles)

    os.makedirs(args.outdir, exist_ok=True)
    pre = (args.prefix + "_") if args.prefix else ""
    tag = loading.replace("/", "")  # e.g., "1/2pi" -> "12pi" for filename friendliness

    # Titles carry loading label, filenames include it too
    spl_title = f"On-axis SPL (1 m) [{loading}]"
    z_title = f"Input Impedance Magnitude [{loading}]"
    spl_path = os.path.join(args.outdir, f"{pre}spl_{tag}.png")
    z_path = os.path.join(args.outdir, f"{pre}impedance_{tag}.png")
    plot_spl(res.f, res.SPL_onaxis, outfile=spl_path, title=spl_title)
    plot_impedance(res.f, res.Zin, outfile=z_path, title=z_title)
    print(f"Wrote: {spl_path}")
    print(f"Wrote: {z_path}")

    if res.SPL_offaxis is not None and res.angles_deg is not None:
        curves = [res.SPL_offaxis[i] for i in range(len(res.angles_deg))]
        labels = [f"{ang:.0f}°" for ang in res.angles_deg]
        ang_title = f"SPL vs angle [{loading}]"
        ang_path = os.path.join(args.outdir, f"{pre}spl_angles_{tag}.png")
        plot_spl_multi(res.f, curves, labels, outfile=ang_path, title=ang_title)
        print(f"Wrote: {ang_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
