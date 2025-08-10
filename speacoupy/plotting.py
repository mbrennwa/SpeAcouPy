
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_spl(f: np.ndarray, spl_db: np.ndarray, outfile: str | None = None, title: str = "On-axis SPL (1 m)"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(f, spl_db)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB re 20 µPa)")
    ax.grid(True, which="both", ls=":")
    ax.set_title(title)
    if outfile:
        fig.savefig(outfile, bbox_inches="tight", dpi=150)
    return fig

def plot_impedance(f: np.ndarray, Zin: np.ndarray, outfile: str | None = None, title: str = "Input Impedance Magnitude"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(f, np.abs(Zin))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Z| (Ohms)")
    ax.grid(True, which="both", ls=":")
    ax.set_title(title)
    if outfile:
        fig.savefig(outfile, bbox_inches="tight", dpi=150)
    return fig
