
# SpeAcouPy (wideband + labeled outputs)

**What changed**
- All SPL/plot outputs are labeled with the **room loading** (4pi/2pi/1pi/1/2pi) in both **titles and filenames**.
- Added `__main__.py` so you can run `python -m speacoupy`, making it pipx-friendly (console script already present).

## pipx install
With a local checkout:
```bash
pipx install .        # or: pipx install git+https://github.com/mbrennwa/SpeAcouPy.git
speacoupy --help
python -m speacoupy --help
```
pipx installs the `speacoupy` console script in an isolated venv. `__main__.py` enables `python -m speacoupy` too.

## Example
```bash
speacoupy examples/config_wideband.yaml --outdir plots --angles 0,15,30,45
# Outputs:
#   plots/spl_2pi.png
#   plots/impedance_2pi.png
#   plots/spl_angles_2pi.png
```

Filenames and titles carry the `[2pi]` (or whichever you choose). If you add data exports later, follow the same pattern.
## New configuration format (simple & explicit)

1. Define *elements* up-front with a `type`, unique `label`, and parameters.
2. Build the *network* as nested `series` / `parallel`, referencing labels.

Example:
```yaml
elements:
  - type: re
    label: R_pad
    R: 1.0
  - type: piston_wideband
    label: FL
    loading: 2pi
  - type: sealed
    label: BL
    Vb_l: 18
  - type: driver
    label: D1
    Re: 5.7
    Le: 0.00035
    Bl: 7.4
    Sd_m2: 0.053
    Rms: 1.6
    Mms: 0.020
    Cms: 0.0007
    front_load: FL
    back_load: BL

network:
  series:
    - R_pad
    - D1
```
