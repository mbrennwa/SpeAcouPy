from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from .constants import PROGRAMNAME

def __branding(ax):
	ax.text(
		0.99, 0.01, PROGRAMNAME,
		transform=ax.transAxes,
		ha="right", va="bottom",
		color="gray",
		bbox=dict(facecolor="white", edgecolor="none", pad=2.0),
		zorder=10
	)

def plot_spl(f: np.ndarray, spl_db: np.ndarray, outfile: str | None = None, title: str = "On-axis SPL"):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.semilogx(f, spl_db, label="On-axis")
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel("SPL (dB-SPL)")
	ax.grid(True, which="both", ls=":")
	ax.set_title(title)
	__branding(ax)
	if outfile:
		fig.savefig(outfile, bbox_inches="tight", dpi=150)
	return fig

def plot_impedance(f: np.ndarray, Zin: np.ndarray, outfile: str | None = None, title: str = "Impedance"):
	"""Plot impedance magnitude with phase overlay on a twin y-axis."""
	mag = np.abs(Zin)
	# phase wrapped to [-180, 180)
	phase_unwrapped_deg = np.rad2deg(np.unwrap(np.angle(Zin)))
	phase_wrapped = ((phase_unwrapped_deg + 180.0) % 360.0) - 180.0

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	line1, = ax1.semilogx(f, mag)
	ax1.set_xlabel("Frequency (Hz)")
	ax1.set_ylabel("Magnitude (Ω)")
	ax1.grid(True, which="both", ls=":")
	ax1.set_ylim(bottom=0)
	ax1.set_title(title)

	ax2 = ax1.twinx()
	line2, = ax2.semilogx(f, phase_wrapped, linestyle="--")
	ax2.set_ylabel("Phase (degrees)")
	ax2.set_ylim(-180, 180)
	ax2.set_yticks([-180, -90, 0, 90, 180])
	ax2.set_yticks([-180, -90, 0, 90, 180])

	# Combined legend
	lines = [line1, line2]
	labels = [l.get_label() for l in lines]
	ax1.legend([line1, line2], ["Magnitude", "Phase"], loc='upper right')

	__branding(ax1)
	if outfile:
		fig.savefig(outfile, bbox_inches="tight", dpi=150)
	return fig

def plot_spl_multi(f: np.ndarray, spl_db_list: list[np.ndarray], labels: list[str], outfile: str | None = None, title: str = "SPL vs Angle"):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	for y, lab in zip(spl_db_list, labels):
		ax.semilogx(f, y, label=lab)
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel("SPL (dB-SPL)")
	ax.grid(True, which="both", ls=":")
	ax.set_title(title)
	__branding(ax)
	if outfile:
		fig.savefig(outfile, bbox_inches="tight", dpi=150)
	return fig
