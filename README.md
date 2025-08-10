
# SpeAcouPy

**SpeAcouPy** – Loudspeaker Electro-Mechanical-Acoustical Modelling in Python

- Lumped-element networks across electrical, mechanical and acoustic domains
- Series/parallel composition (two-terminal networks) with YAML-driven CLI
- Driver, sealed/vented box, and port models
- Simple plotting helpers (SPL and |Z|)

## Install
```bash
pip install -e .
```

## CLI
```bash
speacoupy examples/config_parallel.yaml --outdir plots
```

## YAML network
```yaml
network:
  series:
    - re: { R: 1.0 }
    - parallel:
        - driver: {}
        - ce: { C: 0.000047 }
```

## Notes
- CLI assumes a single driver in a two-terminal network (series/parallel). Multi-driver/multi-branch support is a planned extension.
- SPL is a monopole far-field approximation.
