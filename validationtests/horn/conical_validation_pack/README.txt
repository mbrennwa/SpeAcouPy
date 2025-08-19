Conical Horn Validation Pack
============================

This pack validates SpeAcouPy conical horn impedance against the exact spherical-wave reference
with a rigid termination. It prints a peak comparison table and shows overlay + error plots.

Usage:
1) Place this folder in your SpeAcouPy repo (so speacoupy/ is importable).
2) Run:  python3 run_conical_validation.py

Contents:
- run_conical_validation.py : main validation script
- conical_config.yaml       : example horn config
- conical_geom.csv          : geometry data
- README.txt                : this file
- LICENSE.txt               : MIT license for helper scripts
