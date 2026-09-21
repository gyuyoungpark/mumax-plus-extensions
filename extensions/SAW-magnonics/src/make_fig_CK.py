"""C(K) scan: K-resolved pair-product sum for SAW MEL vs uniform Suhl.

C(K) = sum_k |M_y(k, f_K) M_y(K - k, f_K)|

For a pump with finite momentum k_pump, the pair manifold
k_1 + k_2 = k_pump produces a peak in C(K) at K = k_pump.
This identifies the pair selection rule WITHOUT imposing the
K = k_SAW symmetry by construction (as a fixed-K P(k_1) plot
inevitably does).

Operates on cached m_y(x,t) data from sim37 (SAW MEL @ 2 f_K vs
matched uniform Suhl pump at 2 omega_K with no spatial pump
phase).
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper", "prapplied")

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"
]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 7.5
mpl.rcParams["ytick.labelsize"] = 7.5
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["axes.linewidth"] = 0.6


def spectrum(my_xt, cx, dt):
    """2D FFT of m_y(x, t) on the second half of the run; return
    M_y(k, f) with k in 1/m and f in Hz."""
    T, NX = my_xt.shape
    sub = my_xt[T // 2:, :]
    win = np.hanning(sub.shape[0])[:, None]
    F = np.fft.fftshift(np.fft.fft2(sub * win), axes=(0, 1))
    f_axis = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=dt))
    k_axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=cx))
    return k_axis, f_axis, F


def CK_scan(M_at_fK, k_axis, K_values):
    """Compute C(K) = sum_k |M(k) M(K - k)| at the discrete K grid.

    Uses nearest-neighbour matching on the discrete k grid.
    """
    abs_M = np.abs(M_at_fK)
    # Precompute |M(k)| array
    out = np.zeros_like(K_values, dtype=float)
    dk = k_axis[1] - k_axis[0]
    # k -> index map
    k0 = k_axis[0]
    for j, K in enumerate(K_values):
        # k2 = K - k1; index of k2 = (K - k0)/dk - i_k1
        # but the array length is NX; we sum over all k1 indices
        # with valid k2 also inside range.
        i2_arr = np.rint((K - k_axis - k0) / dk).astype(int)
        mask = (i2_arr >= 0) & (i2_arr < len(k_axis))
        if mask.sum() == 0:
            out[j] = 0
            continue
        i1 = np.where(mask)[0]
        i2 = i2_arr[mask]
        out[j] = np.sum(abs_M[i1] * abs_M[i2])
    return out


def main():
    d37 = np.load(os.path.join(DATA, "sim37_suhl_control.npz"))
    cx = float(d37["CX"])
    dt = float(d37["DT_REC"])
    f_K = float(d37["f_K"])
    k_SAW = float(d37["saw_kSAW"])

    k_saw, f_saw, F_saw = spectrum(d37["saw_my_xt"], cx, dt)
    k_suhl, f_suhl, F_suhl = spectrum(d37["suhl_my_xt"], cx, dt)

    # Slice at +f_K
    i_fK_saw = int(np.argmin(np.abs(f_saw - f_K)))
    i_fK_suhl = int(np.argmin(np.abs(f_suhl - f_K)))
    M_saw = F_saw[i_fK_saw, :]
    M_suhl = F_suhl[i_fK_suhl, :]

    # K scan in micrometre^-1 (display in inverse micrometres).
    # Span from -2 k_SAW to +2 k_SAW so both peaks (uniform K=0 and
    # SAW K = -k_SAW under the Python FFT sign convention) are visible.
    K_grid = np.linspace(-2.4 * k_SAW, 2.4 * k_SAW, 401)

    C_saw = CK_scan(M_saw, k_saw, K_grid)
    C_suhl = CK_scan(M_suhl, k_suhl, K_grid)

    # Audit-recommended FFT-sign convention: flip the displayed K
    # so the SAW pump peak appears at +k_SAW (matching the body
    # text k_SAW/2 narrative).  Equivalent to K_display = -K_fft.
    K_display = -K_grid
    order = np.argsort(K_display)
    K_display = K_display[order]
    C_saw = C_saw[order]
    C_suhl = C_suhl[order]

    # Normalize each curve to its own maximum to compare peak structure
    C_saw_n = C_saw / C_saw.max()
    C_suhl_n = C_suhl / C_suhl.max()

    K_um = K_display / 1e6
    k_SAW_um = k_SAW / 1e6

    # save data
    np.savez(os.path.join(DATA, "fig_CK_scan.npz"),
             K_um=K_um, C_saw=C_saw, C_suhl=C_suhl,
             k_SAW_um=k_SAW_um, f_K_GHz=f_K / 1e9)

    # Plot
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(K_um, C_saw_n, color="#D55E00", lw=1.6, label=r"SAW MEL pump")
    ax.plot(K_um, C_suhl_n, color="#0072B2", lw=1.4, ls="--",
            label=r"uniform Suhl pump")
    ax.axvline(k_SAW_um, color="#D55E00", ls=":", lw=0.7, alpha=0.7)
    ax.axvline(0, color="#0072B2", ls=":", lw=0.7, alpha=0.7)
    ax.text(k_SAW_um, 1.04, r"$+k_\mathrm{SAW}$",
            color="#D55E00", ha="center", va="bottom", fontsize=7)
    ax.text(0, 1.04, r"$K=0$", color="#0072B2",
            ha="center", va="bottom", fontsize=7)
    ax.set_xlabel(r"pair total momentum $K$ ($\mu$m$^{-1}$)")
    ax.set_ylabel(r"$C(K)$ (norm.)")
    ax.set_xlim(K_um.min(), K_um.max())
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper left", frameon=False, handlelength=1.5,
              labelspacing=0.25)

    fig.tight_layout()
    out = os.path.join(FIGS, "fig_CK_scan.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.join(PAPER, "fig_CK_scan.pdf"),
                bbox_inches="tight")
    print(f"saved {out}")

    # Print peak positions for reference
    K_saw_peak = K_um[np.argmax(C_saw_n)]
    K_suhl_peak = K_um[np.argmax(C_suhl_n)]
    print(f"SAW pump  C(K) peak at K = {K_saw_peak:+.2f} /um  "
          f"(expected +k_SAW = {k_SAW_um:+.2f})")
    print(f"Suhl pump C(K) peak at K = {K_suhl_peak:+.2f} /um  "
          f"(expected 0)")


if __name__ == "__main__":
    main()
