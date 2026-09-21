"""Make sim23-style figure with vertically stacked time-domain and FFT panels.

Used as the left half of the new Fig. 4 in main.tex.  The right half (sim24
grid-convergence) is included from the existing fig_sim24_convergence.pdf.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper", "prapplied")

ORANGE = "#D55E00"
BLUE = "#0072B2"
BLACK = "#000000"
GREY = "#777777"


def main():
    d = np.load(os.path.join(DATA, "sim23_fft_proof.npz"))
    f_K = float(d["f_K"]) / 1e9  # GHz

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.0, 3.6))

    # ---------- (a) time trace ----------
    t_ns = d["mel_2fk_times"] * 1e9
    ax1.plot(t_ns, d["mel_2fk_my"], color=ORANGE, lw=0.7,
             label=r"MEL only @ $2f_K$")
    ax1.plot(t_ns, d["mr_fk_my"], color=BLUE, lw=0.7,
             label=r"MR only @ $f_K$")
    ax1.plot(t_ns, d["full_2fk_my"], color=BLACK, lw=0.5, alpha=0.7,
             label=r"Full @ $2f_K$")
    ax1.set_xlabel(r"$t$ (ns)")
    ax1.set_ylabel(r"$m_y$")
    ax1.set_xlim(0, 10)
    ax1.legend(fontsize=6, loc="upper left", framealpha=0.85,
               handlelength=1.4)
    ax1.text(0.02, 0.97, "(a)", transform=ax1.transAxes,
             ha="left", va="top", fontsize=10, fontweight="bold")

    # ---------- (b) FFT power spectrum (second half) ----------
    def fft_power(t, m):
        n = len(t)
        n2 = n // 2
        m_seg = m[n2:] - m[n2:].mean()
        dt = float(t[1] - t[0])
        F = np.fft.rfft(m_seg * np.hanning(len(m_seg)))
        freq = np.fft.rfftfreq(len(m_seg), d=dt) / 1e9  # GHz
        return freq, np.abs(F) ** 2

    f_mel, P_mel = fft_power(d["mel_2fk_times"], d["mel_2fk_my"])
    f_mr,  P_mr  = fft_power(d["mr_fk_times"],  d["mr_fk_my"])
    f_full, P_full = fft_power(d["full_2fk_times"], d["full_2fk_my"])

    norm = max(P_mel.max(), P_mr.max(), P_full.max())
    ax2.plot(f_mel, P_mel / norm, color=ORANGE, lw=1.0,
             label=r"MEL only @ $2f_K$")
    ax2.plot(f_mr, P_mr / norm, color=BLUE, lw=1.0,
             label=r"MR only @ $f_K$")
    ax2.plot(f_full, P_full / norm, color=BLACK, lw=0.8, alpha=0.7,
             label=r"Full @ $2f_K$")
    ax2.axvline(f_K, color=GREY, ls=":", lw=0.7, alpha=0.7)
    ax2.axvline(2 * f_K, color=GREY, ls=":", lw=0.7, alpha=0.7)
    ax2.text(f_K, 1.02, r"$f_K$", color=GREY, ha="center", va="bottom",
             fontsize=7)
    ax2.text(2 * f_K, 1.02, r"$2f_K$", color=GREY, ha="center", va="bottom",
             fontsize=7)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("FFT power (norm.)")
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=6, loc="upper right", framealpha=0.85,
               handlelength=1.4)
    ax2.text(0.02, 0.97, "(b)", transform=ax2.transAxes,
             ha="left", va="top", fontsize=10, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(FIGS, "fig_sim23_fft_vstack.pdf")
    fig.savefig(out, bbox_inches="tight")
    # also copy to paper directory
    fig.savefig(os.path.join(PAPER, "fig_sim23_fft_vstack.pdf"),
                bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
