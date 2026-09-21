"""Make a cleaner Fig.~5 (Suhl-pump kinematic comparison) for main.tex.

Plots two-panel S(k, f) for SAW MEL pump (left) and matched uniform Suhl
pump (right), without panel titles and without the white dispersion lines
or yellow horizontal markers from the legacy figure.  Keeps the cyan
f = f_K dashed line and the green vertical lines at the pair-band centres
(+k_SAW/2 for SAW MEL, 0 for Suhl) so the reader can read off the offset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper", "prapplied")


def spectrum_from_xt(my_xt, cx, dt):
    """Return (k_um, f_GHz, |M|^2 normalized) on second half of run."""
    T, NX = my_xt.shape
    sub = my_xt[T // 2:, :]
    win_t = np.hanning(sub.shape[0])[:, None]
    F = np.fft.fftshift(np.fft.fft2(sub * win_t), axes=(0, 1))
    f_axis = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=dt)) / 1e9
    k_axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=cx)) / 1e6
    P = np.abs(F) ** 2
    P /= P.max()
    return k_axis, f_axis, P


def main():
    d37 = np.load(os.path.join(DATA, "sim37_suhl_control.npz"))
    cx = float(d37["CX"])
    dt = float(d37["DT_REC"])
    f_K_ghz = float(d37["f_K"]) / 1e9
    k_SAW_um = float(d37["saw_kSAW"]) / 1e6

    k_saw, f_saw, P_saw = spectrum_from_xt(d37["saw_my_xt"], cx, dt)
    k_suhl, f_suhl, P_suhl = spectrum_from_xt(d37["suhl_my_xt"], cx, dt)

    fig = plt.figure(figsize=(7.0, 3.0))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.10,
                           left=0.08, right=0.91, top=0.95, bottom=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    vmax = max(P_saw.max(), P_suhl.max()) * 0.6

    im = ax_a.pcolormesh(k_saw, f_saw, P_saw, shading="auto",
                        cmap="magma", vmin=0, vmax=vmax, rasterized=True)
    ax_b.pcolormesh(k_suhl, f_suhl, P_suhl, shading="auto",
                    cmap="magma", vmin=0, vmax=vmax, rasterized=True)

    # FFT sign convention: SAW MEL band sits at -k_SAW/2 in the plotted axis
    # so we mirror the displayed k for the SAW panel only.
    # (Equivalent to k_plot = -k_fft, audit-recommended convention.)
    ax_a.invert_xaxis()

    for ax in (ax_a, ax_b):
        ax.axhline(f_K_ghz, color="cyan", ls="--", lw=0.9, alpha=0.95)
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 10)
        ax.set_xlabel(r"$k_x$ ($\mu$m$^{-1}$)")

    # Pair-band centroid markers (only)
    ax_a.axvline(+k_SAW_um / 2, color="lime", ls="--", lw=0.9)
    ax_b.axvline(0, color="lime", ls="--", lw=0.9)

    ax_a.set_ylabel(r"$f$ (GHz)")
    ax_b.set_yticklabels([])

    # Colorbar
    cbar = fig.colorbar(im, ax=[ax_a, ax_b], shrink=0.85, aspect=22, pad=0.02)
    cbar.set_label(r"$|M_y(k,f)|^2$ (norm.)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Panel (a)/(b) tags only (no descriptive title)
    ax_a.text(0.02, 0.95, "(a)", transform=ax_a.transAxes,
              color="white", fontsize=11, fontweight="bold",
              ha="left", va="top")
    ax_b.text(0.02, 0.95, "(b)", transform=ax_b.transAxes,
              color="white", fontsize=11, fontweight="bold",
              ha="left", va="top")

    out_fig = os.path.join(FIGS, "fig_suhl_clean.pdf")
    fig.savefig(out_fig, bbox_inches="tight")
    fig.savefig(os.path.join(PAPER, "fig_suhl_clean.pdf"),
                bbox_inches="tight")
    print(f"saved {out_fig}")


if __name__ == "__main__":
    main()
