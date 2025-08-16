# Configuration files (YAML)

Simulations are driven by **YAML** configuration files. A minimal sealed-box example:

```yaml
system:
  driver:
    fs: 50       # Hz
    qts: 0.38
    vas: 20      # liters
enclosure:
  type: sealed
  volume: 15     # liters
```

Guidelines:
- Use **SI units** unless otherwise stated.
- Keep numbers realistic; see your driver's datasheet for T/S parameters.
- Start simple; add details (losses, ducts, filters) incrementally.
