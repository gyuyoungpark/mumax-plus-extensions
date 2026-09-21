"""Fig.~6 of the PRApplied draft: strain and angular diagnostics (4 panels).

(a) Peak |m_perp| vs f_SAW at fixed B_0 for several strain amplitudes;
    nonlinear-regime curves visually de-emphasised.
(b) Peak |m_perp| vs eps_0 for three channels, with finite-time
    visibility thresholds and nonlinear-regime shading.
(c) Normalized parametric (2 f_K) angular response vs theta with
    Eq.(4)-based MEL coupling-matrix-element reference curve.
(d) Direct-channel near-collinear angular zoom showing the
    MEL/MR linear crossover near theta_c ~ 1 deg (sim22 data).

Unified Helvetica typography; legend frames removed; the four panels use
distinct accent colours and we avoid using "kinematic" for the angular
panels (kinematic evidence lives in Figs. 3 and 5).
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
PAPER_FIGS = os.path.join(ROOT, "paper", "prapplied", "figures")

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [
    "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"
]
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["axes.labelsize"] = 8.5
mpl.rcParams["xtick.labelsize"] = 7.5
mpl.rcParams["ytick.labelsize"] = 7.5
mpl.rcParams["legend.fontsize"] = 6.8
mpl.rcParams["axes.linewidth"] = 0.6

# Palette: distinct accent per panel
COL_LINEAR = "#0072B2"      # blue (linear-regime strain curves and guides)
COL_NONLIN = "#999999"      # grey (nonlinear-regime, de-emphasised)
COL_MR_FK = "#117733"       # green (MR @ f_K)
COL_FULL_FK = "#000000"     # black (Full @ f_K)
COL_FULL_2FK = "#D55E00"    # vermilion (Full @ 2 f_K, parametric)
COL_MEL = "#882255"         # wine (MEL only, angular)
COL_FULL_CD = "#555555"     # dark grey (Full, angular)
COL_REF = "#0072B2"         # blue (analytical reference curve)

# Linear-regime boundary in benchmark (from main text Sec. III):
# mu_0 |h_mel| = B_0 at eps_0 ~ 4e-4
EPS_LIN_MAX = 4e-4
EPS_SAT = 1e-3  # nonlinear saturation onset


def main():
    d21 = np.load(os.path.join(DATA, "sim21_eps_threshold.npz"))
    d26 = np.load(os.path.join(DATA, "sim26b_angular_fine.npz"))
    d22 = np.load(os.path.join(DATA, "sim22_fine_angle.npz"))
    # sim40: k-resolved strain sweep with dense points at 5e-5, 7e-5
    d40 = np.load(os.path.join(DATA, "sim40_observables.npz"))

    f_saw = d21["f_saw_values"] / 1e9            # GHz
    eps = d21["eps_values"]
    sf = d21["spectra_full"]                     # (6, 40)
    sm = d21["spectra_mr_only"]                  # (6, 40)
    f_K = float(d26["f_K"]) / 1e9

    eps40 = d40["eps_values"]
    labels40 = list(d40["labels"])
    Ipair = d40["I_pair"]   # (n_eps, n_conf)
    i_full2 = labels40.index("full_2fK")
    i_fullK = labels40.index("full_fK")
    i_mrK   = labels40.index("mr_fK")

    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30,
                          left=0.09, right=0.98, top=0.95, bottom=0.10)

    # ===== (a) Peak |m| vs f_SAW for various eps (simplified) =====
    from matplotlib.transforms import blended_transform_factory
    ax_a = fig.add_subplot(gs[0, 0])
    # f_K and 2 f_K guides (prominent)
    ax_a.axvline(f_K, color="#444444", ls=":", lw=0.9, alpha=0.85)
    ax_a.axvline(2 * f_K, color="#444444", ls=":", lw=0.9, alpha=0.85)
    tr_a = blended_transform_factory(ax_a.transData, ax_a.transAxes)
    ax_a.text(f_K, 1.02, r"$f_K$", color="#444444", fontsize=7,
              ha="center", va="bottom", transform=tr_a, clip_on=False)
    ax_a.text(2 * f_K, 1.02, r"$2 f_K$", color="#444444", fontsize=7,
              ha="center", va="bottom", transform=tr_a, clip_on=False)
    # Strain curves: linear-regime as colored solid, nonlinear as grey dashed
    cmap_lin = mpl.colormaps.get_cmap("viridis")
    n_lin = sum(1 for e in eps if e <= EPS_LIN_MAX)
    idx_lin = 0
    for i, e in enumerate(eps):
        exp = int(np.round(np.log10(e)))
        mant = e / 10**exp
        if abs(mant - 1.0) < 0.05:
            lab = fr"$\varepsilon_0\!=\!10^{{{exp}}}$"
        else:
            lab = fr"$\varepsilon_0\!=\!{int(mant)}\!\times\!10^{{{exp}}}$"
        if e <= EPS_LIN_MAX:
            col = cmap_lin(0.15 + 0.75 * idx_lin / max(1, n_lin - 1))
            ax_a.semilogy(f_saw, sf[i], color=col, lw=1.0, label=lab)
            idx_lin += 1
        else:
            ax_a.semilogy(f_saw, sf[i], color=COL_NONLIN, lw=0.7,
                          ls="--", alpha=0.7,
                          label=lab + " (nonlinear)")
    # Horizontal 2-row legend above the data, inside the panel, in the
    # empty headroom created by extending y_max.
    ax_a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0),
                ncol=3, frameon=False,
                handlelength=1.4, columnspacing=0.9, labelspacing=0.25,
                fontsize=6.3, handletextpad=0.4, borderpad=0.2)
    ax_a.set_xlabel(r"$f_\mathrm{SAW}$ (GHz)")
    ax_a.set_ylabel(r"peak $|m_\perp|$")
    ax_a.set_xlim(1, 8)
    # Extend y-range upward so the horizontal 2-row legend fits without
    # overlapping the data curves.
    ax_a.set_ylim(1e-5, 1e2)
    ax_a.text(-0.20, 1.03, "(a)", transform=ax_a.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")

    # ===== (b) Pair-band integrated power vs eps (finite-k observable) =====
    ax_b = fig.add_subplot(gs[0, 1])
    # Pair-band integrated power I_pair = integral_{pair band}
    # |M_y(k, f_K)|^2 dk from sim40 (1024 x 8 x 1 strip, 15 ns, k-resolved
    # storage). This is the finite-k observable that directly tests the
    # k_1 + k_2 = k_SAW pair channel, replacing the volume-averaged FFT
    # peak of sim21.
    Ip_full2 = Ipair[:, i_full2]
    Ip_fullK = Ipair[:, i_fullK]
    Ip_mrK   = Ipair[:, i_mrK]
    # Normalize to the low-strain noise floor of MR @ f_K so that the
    # y-axis carries a dimensionless contrast.
    Inorm = Ip_mrK[0]
    Ip_full2_n = Ip_full2 / Inorm
    Ip_fullK_n = Ip_fullK / Inorm
    Ip_mrK_n   = Ip_mrK   / Inorm

    # Nonlinear-regime shading
    ax_b.axvspan(EPS_LIN_MAX, 1e-2, color="#dddddd", alpha=0.45, zorder=0)
    ax_b.axvspan(EPS_SAT, 1e-2, color="#cfcfcf", alpha=0.55, zorder=0)
    # Visibility-margin thresholds (from main text Sec. III)
    eps_obs_Nvis1 = 1e-5
    eps_obs_Nvis4 = 4e-5
    ax_b.axvline(eps_obs_Nvis1, color=COL_LINEAR, ls=":", lw=0.7,
                 alpha=0.8)
    ax_b.axvline(eps_obs_Nvis4, color=COL_LINEAR, ls="--", lw=0.7,
                 alpha=0.8)
    # Threshold labels: outside the axes box, above each vertical guide,
    # centred on the line in x.
    tr_b = blended_transform_factory(ax_b.transData, ax_b.transAxes)
    ax_b.text(eps_obs_Nvis1, 1.02, r"$N_\mathrm{vis}\!=\!1$",
              color=COL_LINEAR, fontsize=6.5, ha="center", va="bottom",
              transform=tr_b, clip_on=False)
    ax_b.text(eps_obs_Nvis4, 1.02, r"$N_\mathrm{vis}\!=\!4$",
              color=COL_LINEAR, fontsize=6.5, ha="center", va="bottom",
              transform=tr_b, clip_on=False)

    ax_b.loglog(eps40, Ip_mrK_n, "s-", color=COL_MR_FK, ms=5,
                label=r"MR only @ $f_K$")
    ax_b.loglog(eps40, Ip_fullK_n, "o-", color=COL_FULL_FK, ms=5,
                label=r"Full @ $f_K$")
    ax_b.loglog(eps40, Ip_full2_n, "^-", color=COL_FULL_2FK, ms=5,
                label=r"Full @ $2f_K$ (parametric)")
    # Reference slope for the MR baseline (|M|^2 propto eps_0^2)
    eref = np.array([3e-5, 1e-3])
    Iref_mr = (Ip_mrK_n[1] / (eps40[1] ** 2)) * eref ** 2
    ax_b.loglog(eref, Iref_mr, ls=":", color=COL_MR_FK, lw=0.5, alpha=0.7)
    # Slope label at the upper-right end of the green dotted line, just
    # below the line endpoint.
    ax_b.text(eref[1] * 1.15, Iref_mr[1] * 1.05,
              r"$\propto \varepsilon_0^2$",
              color=COL_MR_FK, fontsize=7, ha="left", va="top")

    ax_b.set_xlabel(r"$\varepsilon_0$")
    ax_b.set_ylabel(r"$I_\mathrm{pair}(\varepsilon_0)\,/\,I_\mathrm{pair}^\mathrm{noise}$")
    ax_b.set_xlim(7e-6, 5e-3)
    ax_b.set_ylim(0.5, 5e3)
    ax_b.legend(loc="lower right", frameon=False, handlelength=1.4,
                labelspacing=0.25)
    ax_b.text(-0.20, 1.03, "(b)", transform=ax_b.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")

    # ===== (c) Parametric-channel angular sweep (Cartesian) =====
    ax_c = fig.add_subplot(gs[1, 0])
    theta = np.array(d26["thetas_deg"])
    pm = np.array(d26["peaks_mel"])
    pf = np.array(d26["peaks_full"])
    norm = max(pm.max(), pf.max())
    pm_n = pm / norm
    pf_n = pf / norm

    # Eq. (4) coupling-matrix-element model:
    # in-plane channel |cos 2theta|^2 and out-of-plane channel cos^4 theta,
    # combined as A*|cos2t|^2 + B*cos^4 t with A,B fitted to the MEL-only data.
    th_smooth = np.deg2rad(np.linspace(0, 90, 361))
    w_in = np.cos(2 * th_smooth) ** 2
    w_out = np.cos(th_smooth) ** 4
    # 2-parameter non-negative least squares on MEL-only data
    th_data = np.deg2rad(theta)
    W = np.vstack([np.cos(2 * th_data) ** 2, np.cos(th_data) ** 4]).T
    coeffs, *_ = np.linalg.lstsq(W, pm_n, rcond=None)
    A, B = coeffs[0], coeffs[1]
    ref_curve = A * w_in + B * w_out
    ax_c.plot(np.degrees(th_smooth), ref_curve, "-",
              color=COL_REF, lw=1.1,
              label=r"$A|\cos 2\theta|^2 + B\cos^4\theta$ ref.")
    ax_c.plot(theta, pm_n, "o", color=COL_MEL, ms=4.5,
              label=r"MEL only (sim)")
    ax_c.plot(theta, pf_n, "s", color=COL_FULL_CD, ms=4.0,
              label=r"Full (sim)")
    ax_c.axvline(45, color="#888888", ls=":", lw=0.5, alpha=0.7)
    tr_c = blended_transform_factory(ax_c.transData, ax_c.transAxes)
    ax_c.text(45, 1.02, r"$\theta=45^\circ$", color="#666666",
              fontsize=7, ha="center", va="bottom",
              transform=tr_c, clip_on=False)
    ax_c.set_xlabel(r"$\theta$ (deg)")
    ax_c.set_ylabel(r"normalized peak $|m_\perp|$ at $2f_K$")
    ax_c.set_xlim(0, 90)
    ax_c.set_ylim(0, 1.18)
    ax_c.legend(loc="upper right", frameon=False, handlelength=1.4,
                labelspacing=0.25)
    ax_c.text(-0.20, 1.03, "(c)", transform=ax_c.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")

    # ===== (d) Direct-channel near-collinear angular zoom =====
    ax_d = fig.add_subplot(gs[1, 1])
    th_d = np.array(d22["angles"])
    pk_d = np.array(d22["peaks"])
    theta_c = float(d22["theta_c"])
    pk_d_n = pk_d / pk_d.max()
    # Show 0 - 5 deg zoom
    sel = th_d <= 5.0
    ax_d.plot(th_d[sel], pk_d_n[sel], "o-",
              color=COL_FULL_FK, lw=1.1, ms=5,
              label=r"Full direct drive @ $f_K$")
    ax_d.axvline(theta_c, color=COL_FULL_2FK, ls="--", lw=0.9,
                 alpha=0.9)
    tr_d = blended_transform_factory(ax_d.transData, ax_d.transAxes)
    ax_d.text(theta_c, 1.02,
              rf"$\theta_c\!\approx\!{theta_c:.1f}^\circ$",
              color=COL_FULL_2FK, fontsize=7.5, ha="center", va="bottom",
              transform=tr_d, clip_on=False)
    ax_d.set_xlabel(r"$\theta$ (deg)")
    ax_d.set_ylabel(r"normalized peak $|m_\perp|$ at $f_K$")
    ax_d.set_xlim(0, 5)
    ax_d.set_ylim(0, 1.08)
    ax_d.legend(loc="lower right", frameon=False, handlelength=1.4,
                labelspacing=0.25)
    ax_d.text(-0.20, 1.03, "(d)", transform=ax_d.transAxes,
              ha="left", va="bottom", fontsize=11, fontweight="bold")

    out = os.path.join(FIGS, "fig6_signatures.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(os.path.join(PAPER_FIGS, "fig6_signatures.pdf"),
                bbox_inches="tight")
    print(f"saved {out}")
    print(f"  panel (c) ref. fit: A={A:.3f}, B={B:.3f}")


if __name__ == "__main__":
    main()
