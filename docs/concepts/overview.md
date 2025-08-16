# Concepts overview

SpeAcouPy models loudspeaker systems using **lumped acoustic elements**. Typical elements include:
- **Driver**: moving-coil loudspeaker modeled by its Thiele/Small parameters.
- **Enclosure**: sealed or vented volumes represented by compliance and losses.
- **Ports / ducts**: masses with end corrections and viscous losses.
- **Crossovers / filters**: electrical networks affecting input voltage/current.
- **Radiation**: baffle/port radiation models to predict on-axis SPL.

**Assumptions (high level):**
- Small-signal, linear operation.
- Frequency-domain steady-state response.
- Quasi-1D lumped parameters (valid below modal region of enclosures/rooms).

See the tutorials for concrete examples.
