SpeAcouPy validation test: single conical horn

Consider a loudspeaker driver coupled to a horn on the front and a closed chamber on the back.
Use 2pi radiation space and 2.83 Vrms voltage source.
Assume zero acoustical/mechanical losses in the horn and the chamber.

Driver TSP:

  fs: 35.0
  Vas: 60e-3
  Qms: 3.5
  Qes: 0.35
  Sd: 200E-4
  Re: 8.0
  Z_hf: [ [1E3,8.0] , [1E4,8.0]]

  --> Cms = Vas / (ρ c² Sd²) ≈ 1.063 × 10⁻³ m/N
  --> Mms = 1/((2π fs)² Cms) ≈ 19.46E-3 kg
  --> BL = √[(2π fs Mms Re)/Qes] ≈ 9.891 T·m
  --> Rms = (2π fs Mms)/Qms ≈ 1.223 N·s/m
  --> Le = 0
  
Front horn:
  throat area: 200E-4 m²
  mouth area: 8000E-4 m²
  length: 1 m

Back chamber:
  volume = 10 L = 0.01 m³
