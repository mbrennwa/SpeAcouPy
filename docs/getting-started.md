# Getting started

## Install SpeAcouPy
- Install SpeAcouPy following the instructions in the project README (release binaries or pip).
- Verify installation:
```bash
speacoupy --version
```

## Your first run (no files created)
```bash
speacoupy --help
```
This prints the available commands and options to your **terminal**. No files are created yet.

---

## Run a simple simulation (sealed box)
1) **Download the example config**: [sealed.yaml](examples/sealed.yaml)  
2) **Run the simulation**:
```bash
speacoupy simulate examples/sealed.yaml
```
**What happens:**
- SpeAcouPy reads parameters from `examples/sealed.yaml`.
- It runs the sealed-box simulation.
- It writes results to an output folder (plots as PNG, data as CSV). The default location and filenames depend on the CLI options; see **CLI → Overview** for details.

> Tip: Copy `examples/sealed.yaml` to your working directory and tweak values to explore different box sizes and drivers.
