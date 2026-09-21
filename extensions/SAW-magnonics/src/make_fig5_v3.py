"""Fig.~5 of the PRApplied draft.

Layout follows Fig.~3 (reversed-SAW coherence):
  Row 0: (a) SAW MEL S(k, f), (b) matched uniform Suhl S(k, f),
          side-by-side with square data axes. Colorbar sits in the
          vertical slot to the right of (b).
  Row 1: (c) Pair-total-momentum scan
          C(K) = sum_k |M_y(k, f_K) M_y(K - k, f_K)|
          spanning the full panel width.

Unified Helvetica family, no legend frame, palette distinct from
the configurations colors used in Fig. 4.
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

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"
]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["axes.labelsize"] = 8.5
mpl.rcParams["xtick.labelsize"] = 7.5
mpl.rcParams["ytick.labelsize"] = 7.5
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["axes.linewidth"] = 0.6

# (c) palette: pair-total-momentum scan
COL_SAW = "#117733"    # green (kinematic test)
COL_SUHL = "#882255"   # wine


def spectrum(my_xt, cx, dt):
    T, NX = my_xt.shape
    sub = my_xt[T // 2:, :]
    win = np.hanning(sub.shape[0])[:, None]
    F = np.fft.fftshift(np.fft.fft2(sub * win), axes=(0, 1))
    f_axis = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=dt))
    k_axis = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=cx))
    return k_axis, f_axis, F


def main():
    d37 = np.load(os.path.join(DATA, "sim37_suhl_control.npz"))
    cx = float(d37["CX"])
    dt = float(d37["DT_REC"])
    f_K = float(d37["f_K"])
    k_SAW = float(d37["saw_kSAW"])
    f_K_ghz = f_K / 1e9
    k_SAW_um = k_SAW / 1e6

    k_saw, f_saw, F_saw = spectrum(d37["saw_my_xt"], cx, dt)
    k_suhl, f_suhl, F_suhl = spectrum(d37["suhl_my_xt"], cx, dt)

    # Power spectra (audit-recommended k_plot = -k_fft to match body
    # text convention: SAW MEL pair band appears at +k_SAW/2 in the
    # figure axis).
    P_saw = np.abs(F_saw) ** 2
    P_suhl = np.abs(F_suhl) ** 2
    P_saw /= P_saw.max()
    P_suhl /= P_suhl.max()
    k_um = -k_saw / 1e6
    f_ghz = f_saw / 1e9
    order_k = np.argsort(k_um)
    k_um_s = k_um[order_k]
    P_saw_s = P_saw[:, order_k]
    P_suhl_s = P_suhl[:, order_k]

    # C(K) on M_y(k, +f_K)
    i_fK = int(np.argmin(np.abs(f_saw - f_K)))
    M_saw = F_saw[i_fK, :]
    M_suhl = F_suhl[i_fK, :]

    abs_saw = np.abs(M_saw)
    abs_suhl = np.abs(M_suhl)
    K_grid = np.linspace(-2.4 * k_SAW, 2.4 * k_SAW, 401)
    dk = k_saw[1] - k_saw[0]
    k0 = k_saw[0]

    def CK(absM):
        out = np.zeros_like(K_grid)
        for j, K in enumerate(K_grid):
            i2_arr = np.rint((K - k_saw - k0) / dk).astype(int)
            mask = (i2_arr >= 0) & (i2_arr < len(k_saw))
            i1 = np.where(mask)[0]
            i2 = i2_arr[mask]
            out[j] = np.sum(absM[i1] * absM[i2])
        return out

    C_saw = CK(abs_saw)
    C_suhl = CK(abs_suhl)
    # Display convention K_disp = -K_fft
    K_disp_um = -K_grid / 1e6
    order_K = np.argsort(K_disp_um)
    K_disp_um = K_disp_um[order_K]
    C_saw = C_saw[order_K]
    C_suhl = C_suhl[order_K]
    C_saw /= C_saw.max()
    C_suhl /= C_suhl.max()

    # ===== Build figure (Fig.~3-style layout) =====
    # 1-column width, similar aspect to Fig. 3 (9.5 cm tall).
    fig_w = 3.4
    fig_h = 5.2
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Row 0: (a) and (b) heatmaps with square data axes.
    panel_w = 0.3394
    row0_h = panel_w * (fig_w / fig_h)   # square data area
    row0_y = 0.95 - row0_h
    ax_a = fig.add_axes([0.13, row0_y, panel_w, row0_h])
    ax_b = fig.add_axes([0.5306, row0_y, panel_w, row0_h])

    # Row 1: (c) C(K) line plot — tighter gap to row 0
    row1_h = 0.30
    row1_y = row0_y - 0.13 - row1_h
    ax_c = fig.add_axes([0.16, row1_y, 0.78, row1_h])

    vmax = max(P_saw_s.max(), P_suhl_s.max()) * 0.6

    im = ax_a.pcolormesh(k_um_s, f_ghz, P_saw_s,
                         shading="auto", cmap="magma", vmin=0, vmax=vmax,
                         rasterized=True)
    ax_a.axhline(f_K_ghz, color="cyan", ls="--", lw=0.4, alpha=0.7)
    ax_a.axvline(+k_SAW_um / 2, color="lime", ls="--", lw=0.4, alpha=0.7)
    ax_a.set_xlim(-25, 25)
    ax_a.set_ylim(0, 10)
    ax_a.set_xlabel(r"$k_x$ ($\mu$m$^{-1}$)", labelpad=1)
    ax_a.set_ylabel(r"$f$ (GHz)", labelpad=1)
    ax_a.text(-0.22, 1.04, "(a)", transform=ax_a.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax_a.text(0.97, 0.93, "SAW MEL", transform=ax_a.transAxes,
              ha="right", va="top", color="white", fontsize=8)

    ax_b.pcolormesh(k_um_s, f_ghz, P_suhl_s,
                    shading="auto", cmap="magma", vmin=0, vmax=vmax,
                    rasterized=True)
    ax_b.axhline(f_K_ghz, color="cyan", ls="--", lw=0.4, alpha=0.7)
    ax_b.axvline(0, color="lime", ls="--", lw=0.4, alpha=0.7)
    ax_b.set_xlim(-25, 25)
    ax_b.set_ylim(0, 10)
    ax_b.set_xlabel(r"$k_x$ ($\mu$m$^{-1}$)", labelpad=1)
    ax_b.set_yticklabels([])   # share y with (a)
    ax_b.text(-0.10, 1.04, "(b)", transform=ax_b.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")
    ax_b.text(0.97, 0.93, "uniform Suhl", transform=ax_b.transAxes,
              ha="right", va="top", color="white", fontsize=8)

    # Colorbar in the vertical slot to the right of (b)
    pos_b = ax_b.get_position()
    cax = fig.add_axes([pos_b.x1 + 0.018, pos_b.y0, 0.018, pos_b.height])
    cbar = fig.colorbar(im, cax=cax)
    cax.set_title(r"$|M_y|^2$", fontsize=6.8, pad=2)
    cbar.ax.tick_params(labelsize=6)

    ax_c.plot(K_disp_um, C_saw, color=COL_SAW, lw=1.5,
              label="SAW MEL pump")
    ax_c.plot(K_disp_um, C_suhl, color=COL_SUHL, lw=1.3, ls="--",
              label="uniform Suhl pump")
    ax_c.axvline(+k_SAW_um, color=COL_SAW, ls=":", lw=0.7, alpha=0.7)
    ax_c.axvline(0, color=COL_SUHL, ls=":", lw=0.7, alpha=0.7)
    ax_c.text(+k_SAW_um, 1.04, r"$K\!=\!+k_\mathrm{SAW}$",
              color=COL_SAW, ha="center", va="bottom", fontsize=7)
    ax_c.text(0, 1.04, r"$K\!=\!0$", color=COL_SUHL,
              ha="center", va="bottom", fontsize=7)
    ax_c.set_xlabel(r"pair total momentum $K$ ($\mu$m$^{-1}$)")
    ax_c.set_ylabel(r"$C(K)$ (norm.)")
    ax_c.set_xlim(K_disp_um.min(), K_disp_um.max())
    ax_c.set_ylim(0, 1.18)
    ax_c.legend(loc="upper left", frameon=False, handlelength=1.6,
                labelspacing=0.25)
    ax_c.text(-0.14, 1.04, "(c)", transform=ax_c.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")

    out = os.path.join(FIGS, "fig5_kinematic.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.join(PAPER, "fig5_kinematic.pdf"),
                bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
