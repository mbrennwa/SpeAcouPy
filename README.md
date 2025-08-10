
# SpeAcouPy

**SpeAcouPy** – Loudspeaker Electro-Mechanical-Acoustical Modelling in Python

- Universal two-terminal network `Net(op, parts)` (Series/Parallel are thin wrappers)
- Driver + box + port models
- Boundary-aware **piston radiation** element applied as **mechanical load**
- Room loading options: **4π, 2π, 1π, 1/2π** (affect radiation impedance and SPL)
- YAML-driven CLI

## Install (dev)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e .
```

## CLI
```bash
speacoupy examples/config_boundary.yaml --outdir plots
# Override loading:
speacoupy examples/config_boundary.yaml --loading 2pi
# Disable radiation element:
speacoupy examples/config_boundary.yaml --no-radiation
```

## YAML snippets
```yaml
room:
  loading: 2pi     # 4pi | 2pi | 1pi | 1/2pi

radiation:
  enabled: true    # include front piston radiation as mechanical load
```

Unified network node example:
```yaml
network:
  net:
    op: series
    parts:
      - re: { R: 1.0 }
      - net:
          op: parallel
          parts:
            - driver: {}
            - ce: { C: 4.7e-5 }
```

### Notes
- Radiation model uses a **low-frequency approximation**; for higher ka a wideband piston model is recommended.
- Boundary loading multiplies both the radiation resistance and reactance (simple image-source LF approximation).
