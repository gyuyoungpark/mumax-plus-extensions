"""Fig.~4 of the PRApplied draft: three panels stacked vertically.

(a) Time-domain trace of m_y(t)
(b) FFT power spectrum (second half)
(c) Grid-convergence of peak |m_perp| at N_x = 128, 256, 512

Conventions:
- Single Helvetica-family sans-serif font throughout, no legend frame.
- Panels (a)/(b) share their three colors (MEL, MR, Full), because they
  show the same signals.  Panel (c) uses a separate palette so that no
  individual color is reused across (a,b) and (c).
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper", "prapplied", "figures")

# --- typography (Helvetica family) ---
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"
]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["axes.labelsize"] = 8.5
mpl.rcParams["xtick.labelsize"] = 7.5
mpl.rcParams["ytick.labelsize"] = 7.5
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["lines.linewidth"] = 0.9
mpl.rcParams["axes.linewidth"] = 0.6

# (a,b) palette: configurations (shared across the two upper panels)
COL_MEL = "#D55E00"   # vermilion orange
COL_MR = "#0072B2"    # blue
COL_FULL = "#000000"  # black

# (c) palette: grid-convergence series, chosen distinct from (a,b)
COL_FULL_FK = "#882255"   # wine
COL_FULL_2FK = "#117733"  # green
COL_MR_FK = "#88CCEE"     # cyan (light)
COL_MEL_2FK = "#AA4499"   # purple


def main():
    d23 = np.load(os.path.join(DATA, "sim23_fft_proof.npz"))
    d24 = np.load(os.path.join(DATA, "sim24_grid_convergence.npz"))
    f_K_ghz = float(d23["f_K"]) / 1e9

    fig, axes = plt.subplots(
        3, 1, figsize=(3.4, 5.6),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0], "hspace": 0.45},
    )
    ax_a, ax_b, ax_c = axes

    # ===== (a) time trace =====
    t_ns = d23["mel_2fk_times"] * 1e9
    ax_a.plot(t_ns, d23["mel_2fk_my"], color=COL_MEL, lw=0.7,
              label=r"MEL only @ $2f_K$")
    ax_a.plot(t_ns, d23["mr_fk_my"], color=COL_MR, lw=0.7,
              label=r"MR only @ $f_K$")
    ax_a.plot(t_ns, d23["full_2fk_my"], color=COL_FULL, lw=0.5, alpha=0.75,
              label=r"Full @ $2f_K$")
    ax_a.set_xlabel(r"$t$ (ns)")
    ax_a.set_ylabel(r"$m_y$")
    ax_a.set_xlim(0, 10)
    ax_a.legend(loc="upper left", frameon=False, handlelength=1.4,
                borderpad=0.1, labelspacing=0.25)
    ax_a.text(-0.18, 1.02, "(a)", transform=ax_a.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")

    # ===== (b) FFT power spectrum (second half) =====
    def fft_power(t, m):
        n2 = len(t) // 2
        m_seg = m[n2:] - m[n2:].mean()
        dt = float(t[1] - t[0])
        F = np.fft.rfft(m_seg * np.hanning(len(m_seg)))
        freq = np.fft.rfftfreq(len(m_seg), d=dt) / 1e9
        return freq, np.abs(F) ** 2

    f_mel, P_mel = fft_power(d23["mel_2fk_times"], d23["mel_2fk_my"])
    f_mr, P_mr = fft_power(d23["mr_fk_times"], d23["mr_fk_my"])
    f_full, P_full = fft_power(d23["full_2fk_times"], d23["full_2fk_my"])
    norm = max(P_mel.max(), P_mr.max(), P_full.max())

    ax_b.plot(f_mel, P_mel / norm, color=COL_MEL, lw=1.0,
              label=r"MEL only @ $2f_K$")
    ax_b.plot(f_mr, P_mr / norm, color=COL_MR, lw=1.0,
              label=r"MR only @ $f_K$")
    ax_b.plot(f_full, P_full / norm, color=COL_FULL, lw=0.8, alpha=0.75,
              label=r"Full @ $2f_K$")
    ax_b.axvline(f_K_ghz, color="#555555", ls=":", lw=0.6, alpha=0.7)
    ax_b.axvline(2 * f_K_ghz, color="#555555", ls=":", lw=0.6, alpha=0.7)
    from matplotlib.transforms import blended_transform_factory
    tr_b4 = blended_transform_factory(ax_b.transData, ax_b.transAxes)
    ax_b.text(f_K_ghz, 1.02, r"$f_K$", color="#444444",
              ha="center", va="bottom", fontsize=7,
              transform=tr_b4, clip_on=False)
    ax_b.text(2 * f_K_ghz, 1.02, r"$2f_K$", color="#444444",
              ha="center", va="bottom", fontsize=7,
              transform=tr_b4, clip_on=False)
    ax_b.set_xlabel("Frequency (GHz)")
    ax_b.set_ylabel("FFT power (norm.)")
    ax_b.set_xlim(0, 8)
    ax_b.set_ylim(0, 1.18)
    # Legend in the empty region between the two dotted guides
    # (f_K at 3 GHz, 2 f_K at 6 GHz; centred at 4.5 GHz in data coords).
    ax_b.legend(loc="center", bbox_to_anchor=(0.5625, 0.55),
                frameon=False, handlelength=1.4,
                borderpad=0.1, labelspacing=0.25)
    ax_b.text(-0.18, 1.02, "(b)", transform=ax_b.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")

    # ===== (c) grid convergence =====
    grids = np.array(d24["grid_sizes"], dtype=float)
    full_fk = np.array([float(d24[f"full_fk_nx{int(n)}"]) for n in grids])
    full_2fk = np.array([float(d24[f"full_2fk_nx{int(n)}"]) for n in grids])
    mr_fk = np.array([float(d24[f"mr_fk_nx{int(n)}"]) for n in grids])
    mel_2fk = np.array([float(d24[f"mel_2fk_nx{int(n)}"]) for n in grids])

    ax_c.plot(grids, full_fk, "o-", color=COL_FULL_FK, ms=4,
              label=r"Full @ $f_K$")
    ax_c.plot(grids, full_2fk, "s-", color=COL_FULL_2FK, ms=4,
              label=r"Full @ $2f_K$")
    ax_c.plot(grids, mr_fk, "^-", color=COL_MR_FK, ms=4,
              label=r"MR @ $f_K$")
    ax_c.plot(grids, mel_2fk, "D-", color=COL_MEL_2FK, ms=4,
              label=r"MEL @ $2f_K$")
    ax_c.set_xticks(grids)
    ax_c.set_xticklabels([f"{int(n)}" for n in grids])
    ax_c.set_xlabel(r"$N_x$")
    ax_c.set_ylabel(r"peak $|m_\perp|$")
    ax_c.set_ylim(0, max(0.6, 1.05 * max(full_2fk.max(), mel_2fk.max())))
    ax_c.legend(loc="center right", frameon=False, handlelength=1.4,
                borderpad=0.1, labelspacing=0.25)
    ax_c.text(-0.18, 1.02, "(c)", transform=ax_c.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = os.path.join(FIGS, "fig4_vstack3.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.join(PAPER, "fig4_vstack3.pdf"), bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
