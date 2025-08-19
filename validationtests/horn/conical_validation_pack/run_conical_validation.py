#!/usr/bin/env python3
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# --- path setup ---
here = Path(__file__).resolve().parent
repo_root = here.parents[2]   # conical_validation_pack -> horn -> validationtests -> repo root
sys.path.insert(0, str(repo_root))




from speacoupy.acoustic import Horn
from speacoupy.constants import RHO0, C0

REF_DAMP_EPS = 1e-4

def exact_conical_Zin_rigid(omega, L, S_throat, S_mouth, eps=REF_DAMP_EPS):
	R = np.sqrt(S_mouth / S_throat)
	r1 = L / max(R - 1.0, 1e-16)
	r2 = R * r1
	k = (omega / C0) * (1.0 - 1j * eps)
	k1, k2 = k * r1, k * r2

	E1m, E1p = np.exp(-1j * k1), np.exp(+1j * k1)
	E2m = np.exp(-1j * k2)

	num_ratio = (-1.0 - 1j * k2) / (-1.0 + 1j * k2)
	B_over_A = - (E2m * E2m) * num_ratio

	p1 = (E1m + B_over_A * E1p) / r1
	U1_nofactor = (E1m * (-1.0 - 1j * k1) + B_over_A * E1p * (-1.0 + 1j * k1))
	U1 = U1_nofactor / (1j * omega * RHO0)

	Zin = p1 / np.where(np.abs(U1) < 1e-30, 1e-30 + 0j, U1)
	return Zin

def _find_peaks_simple(y, min_separation=10):
	m = np.abs(y)
	peaks, last = [], -10**9
	for i in range(1, len(m) - 1):
		if m[i] > m[i-1] and m[i] > m[i+1]:
			if (i - last) >= min_separation:
				peaks.append(i); last = i
	return np.array(peaks, dtype=int)

def _match_peaks(f, idx_a, idx_b):
	if len(idx_a) == 0 or len(idx_b) == 0: return []
	matches, used = [], set()
	for ia in idx_a:
		j = int(np.argmin(np.abs(f[ia] - f[idx_b])))
		jb = idx_b[j]
		if j not in used:
			matches.append((ia, jb)); used.add(j)
	return matches

def main():
	L, r_throat, r_mouth = 0.30, 0.02, 0.20
	S_throat, S_mouth = np.pi*r_throat**2, np.pi*r_mouth**2

	f = np.linspace(50.0, 2000.0, 500)
	omega = 2*np.pi*f

	horn = Horn(L=L, S_throat=S_throat, S_mouth=S_mouth,
		profile="con", R_throat=0.0, R_mouth=0.0, mouth_load="rigid")

	A, B, C, D = horn._abcd_chain(omega, N=256)
	ZL = horn._load_impedance_at_mouth(omega)

	Zin_spe = np.empty_like(omega, dtype=complex)
	finite = np.isfinite(ZL)
	if np.any(finite):
		den = C[finite]*ZL[finite] + D[finite]
		den = np.where(np.abs(den)<1e-18, 1e-18+0j, den)
		Zin_spe[finite] = (A[finite]*ZL[finite] + B[finite]) / den
	infmask = ~finite
	if np.any(infmask):
		Csafe = np.where(np.abs(C[infmask])<1e-18, 1e-18+0j, C[infmask])
		Zin_spe[infmask] = A[infmask]/Csafe

	Zin_ref = exact_conical_Zin_rigid(omega, L, S_throat, S_mouth)

	idx_spe, idx_ref = _find_peaks_simple(Zin_spe), _find_peaks_simple(Zin_ref)
	pairs = _match_peaks(f, idx_spe, idx_ref)

	print("\\nPeak comparison (up to first 8 matches):")
	print("  #   f_spe [Hz]   f_ref [Hz]   Δf [Hz]   Δf/f_ref [%]")
	for n, (ia, ib) in enumerate(pairs[:8], 1):
		df, rel = f[ia]-f[ib], 100*(f[ia]-f[ib])/max(f[ib],1e-12)
		print(f"{n:3d}  {f[ia]:10.2f}  {f[ib]:10.2f}  {df:8.2f}   {rel:10.3f}")

	plt.figure(); plt.semilogy(f, np.abs(Zin_spe), label="SpeAcouPy |Zin|")
	plt.semilogy(f, np.abs(Zin_ref), "--", label="Exact conical |Zin|")
	plt.xlabel("Frequency [Hz]"); plt.ylabel("|Zin| [Pa·s/m³]"); plt.grid(True,which="both",ls=":")
	plt.title("Conical horn input impedance (rigid mouth)"); plt.legend()

	plt.figure(); plt.plot(f, np.unwrap(np.angle(Zin_spe)), label="SpeAcouPy ∠Zin")
	plt.plot(f, np.unwrap(np.angle(Zin_ref)), "--", label="Exact ∠Zin")
	plt.xlabel("Frequency [Hz]"); plt.ylabel("Phase [rad]"); plt.grid(True, ls=":"); plt.legend()

	mag_err_db = 20*np.log10(np.maximum(np.abs(Zin_spe),1e-30)/np.maximum(np.abs(Zin_ref),1e-30))
	plt.figure(); plt.plot(f, mag_err_db); plt.axhline(0, ls=":", lw=1)
	plt.xlabel("Frequency [Hz]"); plt.ylabel("Mag error [dB]"); plt.grid(True, ls=":")
	plt.title("SpeAcouPy vs exact conical |Zin| error")
	plt.show()

if __name__=="__main__": main()
