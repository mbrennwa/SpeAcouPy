# Tutorial: Sealed box

**Goal:** Predict on-axis SPL and impedance of a simple sealed enclosure.

## 1) Prepare the config
Download: [examples/sealed.yaml](../examples/sealed.yaml)

```yaml
--8<-- "examples/sealed.yaml"
```

## 2) Run the simulation
```bash
speacoupy simulate examples/sealed.yaml
```

## 3) Inspect results
- **SPL plot**: expected smooth roll-off with system Q near `Qtc`.
- **Impedance plot**: single broad resonance peak near box-loaded `Fc`.

If results look odd, double-check units (liters vs m³, Hz vs rad/s) and typos.
