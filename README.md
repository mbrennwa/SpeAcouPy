
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
