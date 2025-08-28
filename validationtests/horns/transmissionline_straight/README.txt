SpeAcouPy validation test: transmissionline (straight horn with no flare)

Consider a loudspeaker driver coupled to a straight transmissionline on the back and radiation space on the front.
Use 2pi radiation space and 2.83 Vrms voltage source.
Assume zero acoustical/mechanical losses in the line.
With a 2.5 m long line, the 1/4 wave resonance peaks are expected at 34 Hz, 103 Hz, 172 Hz, etc.

Comparison with AJ Horn:
SPL and impedance curves are very similar. There are two minor inconsistencies:
(i) The SpeAcouPy output shows sharper wiggles, which is due to higher resolution of the frequency values. The SpeAcouPy output can be made to look the same as AJ Horn by using lower frequency resolution.
(ii) Peaks and dips show a minor frequency offset of maybe 1-2%. The AJ Horn peak positions align perfectly with the empirical f_peak = (2k-1) * c / (4 * Leff) formula, where Leff = L + 0.6*r takes into account an empirical end-correction. The SpeAcouPy uses a physically more accurate model for the radiation load of the transmission line radiating into space.

Dimensions and parameters used for the simulation:

* Driver TSP:
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
  
* Rear line:
  throat area: 400E-4 m²
  mouth area: 400E-4 m²
  length: 2.5 m
