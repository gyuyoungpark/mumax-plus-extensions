"""Generate PRL-ready v2 figures without rerunning simulations.

Outputs:
  paper/prl/fig_prl_schematic_v2.pdf
  paper/prl/fig_prl_parametric_v2.pdf
  paper/prl/fig_prl_pairrule_v2.pdf
  paper/prl/fig_prl_evidence_v2.pdf
  paper/prl/fig_prl_predictions_v2.pdf

The script reads existing cached data only.
"""

from __future__ import annotations

import os
import shutil
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

try:
    from plot_style import (
        apply_style,
        axis_label,
        label_panels,
        SINGLE_COL,
        DOUBLE_COL,
        CM_TO_INCH,
        SKY_BLUE,
        VERMILION,
        TEAL,
        ORANGE,
        BLUE,
        BLACK,
    )
except Exception:
    SINGLE_COL = 8.6 / 2.54
    DOUBLE_COL = 17.8 / 2.54
    CM_TO_INCH = 1 / 2.54
    SKY_BLUE = "#56B4E9"
    VERMILION = "#D55E00"
    TEAL = "#009E73"
    ORANGE = "#E69F00"
    BLUE = "#0072B2"
    BLACK = "#000000"

    def apply_style():
        plt.rcParams.update({"font.size": 8})

    def axis_label(var, unit=None):
        return var if unit is None else f"{var} ({unit})"

    def label_panels(axes, labels=None, x=-0.12, y=1.06, fontsize=10):
        flat = list(np.atleast_1d(axes).flat)
        if labels is None:
            labels = [f"({chr(ord('a') + i)})" for i in range(len(flat))]
        for ax, lbl in zip(flat, labels):
            ax.text(x, y, lbl, transform=ax.transAxes, fontsize=fontsize,
                    fontweight="bold", va="bottom", ha="left")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
FIG_DIR = os.path.join(ROOT_DIR, "figures")
PRL_DIR = os.path.join(ROOT_DIR, "paper", "prl")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PRL_DIR, exist_ok=True)

GAMMA = 1.76e11
MU0 = 4 * np.pi * 1e-7
MS = 140e3
B1 = -8.8e6
KMR = 1.0e6
XI = 0.68
EPS0 = 1e-4


def apply_prl_style():
    """Apply the shared style, then force PRL figure text to sans-serif."""
    apply_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans",
                            "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
    })


def label_panels(axes, labels=None, x=-0.12, y=1.06, fontsize=10):
    """Add sans-serif bold panel labels."""
    flat = list(np.atleast_1d(axes).flat)
    if labels is None:
        labels = [f"({chr(ord('a') + i)})" for i in range(len(flat))]
    for ax, lbl in zip(flat, labels):
        ax.text(x, y, lbl, transform=ax.transAxes, fontsize=fontsize,
                fontweight="bold", va="bottom", ha="left",
                fontfamily="sans-serif")


def kittel_freq(B0):
    return GAMMA / (2 * np.pi) * np.sqrt(np.maximum(B0 * (B0 + MU0 * MS), 0))


def save_figure(fig, name):
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.025,
                    facecolor="white", transparent=False)
        if ext == "pdf":
            shutil.copy2(out, os.path.join(PRL_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"Saved {name}.pdf/png")


def arrow(ax, xy0, xy1, color=BLACK, lw=1.2, mutation_scale=10, **kwargs):
    patch = FancyArrowPatch(
        xy0, xy1, arrowstyle="-|>", color=color, lw=lw,
        mutation_scale=mutation_scale, shrinkA=0, shrinkB=0, **kwargs
    )
    ax.add_patch(patch)
    return patch


def setup_blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_schematic_v2():
    """Two-panel mechanism schematic: (a) geometry, (b) frequency/momentum selection."""
    apply_prl_style()
    fig = plt.figure(figsize=(DOUBLE_COL, 4.6 * CM_TO_INCH))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.18,
                           left=0.04, right=0.99, top=0.93, bottom=0.05)
    ax_geom = fig.add_subplot(gs[0, 0])
    ax_freq = fig.add_subplot(gs[0, 1])

    # (a) Geometry
    setup_blank(ax_geom)
    ax_geom.set_xlim(0, 1)
    ax_geom.set_ylim(0, 1)
    film = Rectangle((0.10, 0.20), 0.80, 0.25, facecolor="#EEEEEE",
                     edgecolor=BLACK, lw=0.8)
    ax_geom.add_patch(film)
    for x in np.linspace(0.12, 0.86, 12):
        ax_geom.plot([x, x + 0.08], [0.20, 0.45], color="#D0D0D0", lw=0.6)
    xs = np.linspace(0.10, 0.90, 300)
    ax_geom.plot(xs, 0.60 + 0.035 * np.sin(2 * np.pi * 4 * (xs - 0.10)),
                 color=BLUE, lw=1.2)
    arrow(ax_geom, (0.20, 0.78), (0.82, 0.78), lw=1.1)
    ax_geom.text(0.50, 0.82, r"$\mathbf{k}_\mathrm{SAW}$", ha="center", va="bottom")
    arrow(ax_geom, (0.20, 0.33), (0.52, 0.33), color=BLACK, lw=1.4)
    arrow(ax_geom, (0.50, 0.33), (0.82, 0.33), color=BLACK, lw=1.4)
    ax_geom.text(0.50, 0.09, r"$\mathbf{m}_0 \parallel \mathbf{k}_\mathrm{SAW} \parallel \mathbf{B}_0$",
                 ha="center", va="center", fontsize=9)
    for x in np.linspace(0.22, 0.78, 5):
        ax_geom.plot([x - 0.025, x + 0.025], [0.27, 0.30],
                     color=VERMILION, lw=1.5)
    ax_geom.text(0.14, 0.52, r"Rayleigh SAW", color=BLUE, fontsize=8)
    ax_geom.text(0.70, 0.47, r"$\varepsilon_{xx}$", color=VERMILION, fontsize=8)
    ax_geom.text(0.13, 0.67, r"$\Omega_y$", color=BLUE, fontsize=8)

    # (b) Frequency and momentum selection
    setup_blank(ax_freq)
    ax_freq.set_xlim(0, 1)
    ax_freq.set_ylim(0, 1)
    ax_freq.plot([0.08, 0.92], [0.74, 0.74], color="#DDDDDD", lw=0.8)
    ax_freq.plot([0.08, 0.92], [0.34, 0.34], color="#DDDDDD", lw=0.8)
    ax_freq.text(0.10, 0.84, "direct MR", color=SKY_BLUE, fontsize=9,
                 fontweight="bold")
    ax_freq.text(0.10, 0.44, "parametric MEL", color=VERMILION, fontsize=9,
                 fontweight="bold")
    arrow(ax_freq, (0.36, 0.82), (0.36, 0.68), color=SKY_BLUE, lw=1.3)
    ax_freq.text(0.37, 0.82, r"$\omega_\mathrm{SAW}=\omega_K$", va="bottom",
                 fontsize=8, color=SKY_BLUE)
    ax_freq.text(0.52, 0.73, r"one magnon", va="center", fontsize=8)
    ax_freq.text(0.52, 0.64, r"$(\omega_K,k_\mathrm{SAW})$", va="center",
                 fontsize=8)
    arrow(ax_freq, (0.43, 0.49), (0.55, 0.35), color=VERMILION, lw=1.3)
    arrow(ax_freq, (0.43, 0.49), (0.31, 0.35), color=VERMILION, lw=1.3)
    ax_freq.text(0.45, 0.50, r"$2\omega_K,k_\mathrm{SAW}$", fontsize=8,
                 color=VERMILION, va="bottom")
    ax_freq.text(0.43, 0.19, r"$\omega_K,k_1$", fontsize=8, ha="right")
    ax_freq.text(0.49, 0.19, r"$\omega_K,k_2$", fontsize=8, ha="left")
    ax_freq.text(0.50, 0.05, r"$\mathbf{k}_1+\mathbf{k}_2=\mathbf{k}_\mathrm{SAW}$",
                 fontsize=10, ha="center")

    label_panels([ax_geom, ax_freq], x=-0.07, y=1.02)
    save_figure(fig, "fig_prl_schematic_v2")


def peak_from_spectrum(f_ghz, spectrum):
    idx = int(np.argmax(spectrum))
    return float(f_ghz[idx])


def fig_parametric_v2():
    apply_prl_style()
    d = np.load(os.path.join(DATA_DIR, "sim20_parametric_channels.npz"))
    f_ghz = d["f_saw_values"] * 1e-9
    b_vals = d["B0_values"]
    b_mT = b_vals * 1e3
    sf = d["spectra_full"]
    sm = d["spectra_mr_only"]
    se = d["spectra_mel_only"]

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 6.1 * CM_TO_INCH))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.98, bottom=0.20, top=0.92)

    ax = axes[0]
    bi = int(np.argmin(np.abs(b_mT - 50)))
    f_k = kittel_freq(b_vals[bi]) * 1e-9
    norm = max(np.max(sf[bi]), np.max(sm[bi]), np.max(se[bi]), 1e-12)
    ax.plot(f_ghz, sf[bi] / norm, color=BLACK, lw=1.2, label="MEL+MR")
    ax.plot(f_ghz, sm[bi] / norm, "--", color=SKY_BLUE, lw=1.2, label="MR only")
    ax.plot(f_ghz, se[bi] / norm, ":", color=VERMILION, lw=1.7,
            label="MEL only")
    ax.axvline(f_k, color=SKY_BLUE, ls=":", lw=0.8)
    ax.axvline(2 * f_k, color=VERMILION, ls=":", lw=0.8)
    ax.text(f_k, 0.94, r"$f_K$", color=SKY_BLUE, ha="center", va="top",
            fontsize=8)
    ax.text(2 * f_k, 0.94, r"$2f_K$", color=VERMILION, ha="center",
            va="top", fontsize=8)
    ax.set_xlabel(axis_label(r"$f_\mathrm{SAW}$", "GHz"))
    ax.set_ylabel(r"$|m_\perp|$ (norm.)")
    ax.set_xlim(0.5, 10.0)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    f_mr = np.array([peak_from_spectrum(f_ghz, row) for row in sm])
    f_full = np.array([peak_from_spectrum(f_ghz, row) for row in sf])
    f_mel = np.array([peak_from_spectrum(f_ghz, row) for row in se])
    valid = (b_mT >= 30) & (b_mT <= 80)
    edge = ~valid

    ax.plot(b_mT, f_mr, "s", color=SKY_BLUE, mec=BLACK, mew=0.35,
            label="MR peak")
    ax.plot(b_mT[valid], f_full[valid], "o", color=BLACK, mec=BLACK,
            mew=0.35, label="full peak")
    ax.plot(b_mT[valid], f_mel[valid], "^", color=VERMILION, mec=BLACK,
            mew=0.35, label="MEL peak")
    ax.plot(b_mT[edge], f_full[edge], "o", mfc="white", mec="#888888",
            mew=0.8, ms=5, label="out of window")
    ax.plot(b_mT[edge], f_mel[edge], "^", mfc="white", mec="#888888",
            mew=0.8, ms=5)

    b_fit = np.linspace(5, 125, 300) * 1e-3
    f_fit = kittel_freq(b_fit) * 1e-9
    ax.plot(b_fit * 1e3, f_fit, "--", color=SKY_BLUE, lw=0.8,
            label=r"$f_K$")
    ax.plot(b_fit * 1e3, 2 * f_fit, "--", color=VERMILION, lw=0.8,
            label=r"$2f_K$")
    ax.axvspan(30, 80, color=VERMILION, alpha=0.07, lw=0)
    ax.text(55, 1.25, "30--80 mT\nused for\n$2f_K$ test", color=VERMILION,
            ha="center", va="bottom", fontsize=7)
    ax.set_xlabel(axis_label(r"$B_0$", "mT"))
    ax.set_ylabel(axis_label(r"$f_\mathrm{peak}$", "GHz"))
    ax.set_xlim(5, 130)
    ax.set_ylim(0.5, 11.0)
    ax.legend(loc="upper left", fontsize=6.3, ncol=2, columnspacing=0.8,
              handletextpad=0.4)

    label_panels(axes, x=-0.15, y=1.03)
    save_figure(fig, "fig_prl_parametric_v2")


def omega_k_dispersion(k_um):
    k = k_um * 1e6
    gamma_hz = GAMMA / (2 * np.pi)
    b0 = 0.05
    aex = 3.65e-12
    field_ex = 2 * aex * k**2 / MS
    return gamma_hz * np.sqrt((b0 + field_ex) * (b0 + field_ex + MU0 * MS)) * 1e-9


def spectrum_from_xt(my_xt, cx, dt):
    signal = np.asarray(my_xt[my_xt.shape[0] // 2:], dtype=float)
    signal = signal - np.mean(signal)
    wt = np.hanning(signal.shape[0])[:, None]
    wx = np.hanning(signal.shape[1])[None, :]
    spec = np.fft.fftshift(np.fft.fft2(signal * wt * wx), axes=(0, 1))
    f = np.fft.fftshift(np.fft.fftfreq(signal.shape[0], d=dt)) * 1e-9
    k = np.fft.fftshift(np.fft.fftfreq(signal.shape[1], d=cx)) * 2 * np.pi * 1e-6
    power = np.abs(spec) ** 2
    power = power / max(np.max(power), 1e-30)
    pos = f >= 0
    return k, f[pos], power[pos]


def pair_product_from_xt(my_xt, cx, dt, f_target_hz, k_pump_rad_m):
    signal = np.asarray(my_xt[my_xt.shape[0] // 2:], dtype=float)
    signal = signal - np.mean(signal)
    m2d = np.fft.fft2(signal)
    freqs_t = np.fft.fftfreq(signal.shape[0], d=dt)
    freqs_k = np.fft.fftfreq(signal.shape[1], d=cx) * 2 * np.pi
    i_f = int(np.argmin(np.abs(freqs_t - f_target_hz)))
    mk = m2d[i_f]
    dk = freqs_k[1]
    n_pump = int(round(k_pump_rad_m / dk)) % signal.shape[1]
    n = np.arange(signal.shape[1])
    n2 = (n_pump - n) % signal.shape[1]
    p = np.abs(mk[n]) * np.abs(mk[n2])
    k_um = np.fft.fftshift(freqs_k) * 1e-6
    p = np.fft.fftshift(p)
    return k_um, p / max(np.max(p), 1e-30)


def fig_pairrule_v2():
    apply_prl_style()
    d25 = np.load(os.path.join(DATA_DIR, "sim25_kspectrum.npz"))
    d36 = np.load(os.path.join(DATA_DIR, "sim36_twomode_correlator.npz"))
    d37 = np.load(os.path.join(DATA_DIR, "sim37_suhl_control.npz"))

    cx = float(d25["CX"])
    dt = float(d25["DT_REC"])
    f_k = float(d25["f_K"])
    k_mr_pump_um = float(d25["mr_fk_kSAW"]) * 1e-6
    k_mel_pump_um = float(d25["mel_2fk_kSAW"]) * 1e-6

    k_mr, f_mr, pwr_mr = spectrum_from_xt(d25["mr_fk_my_xt"], cx, dt)
    k_mel, f_mel, pwr_mel = spectrum_from_xt(d25["mel_2fk_my_xt"], cx, dt)

    fig = plt.figure(figsize=(DOUBLE_COL, 6.4 * CM_TO_INCH))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.05],
                           wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    k_disp = np.linspace(-30, 30, 400)
    f_disp = omega_k_dispersion(k_disp)

    heat_specs = [
        (axes[0], k_mr, f_mr, pwr_mr, r"MR drive", k_mr_pump_um, False),
        (axes[1], k_mel, f_mel, pwr_mel, r"MEL pump", k_mel_pump_um, True),
    ]
    im = None
    for ax, k_ax, f_ax, pwr, title, k_ref, is_param in heat_specs:
        im = ax.pcolormesh(k_ax, f_ax, pwr, shading="auto", cmap="magma",
                           vmin=0, vmax=0.35, rasterized=True)
        ax.plot(k_disp, f_disp, color="white", lw=0.7, alpha=0.75)
        ax.plot(-k_disp, f_disp, color="white", lw=0.7, alpha=0.75)
        ax.axhline(f_k * 1e-9, color="cyan", ls="--", lw=0.8)
        if is_param:
            ax.axvline(k_ref / 2, color="lime", ls="--", lw=0.9)
            ax.axvline(-k_ref / 2, color="lime", ls="--", lw=0.5, alpha=0.45)
            ax.text(0.05, 0.91, r"pairs at $f_K$", transform=ax.transAxes,
                    color="white", fontsize=7, va="top")
        else:
            ax.axvline(k_ref, color="lime", ls="--", lw=0.9)
            ax.axvline(-k_ref, color="lime", ls="--", lw=0.5, alpha=0.45)
            ax.text(0.05, 0.91, r"single mode", transform=ax.transAxes,
                    color="white", fontsize=7, va="top")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(axis_label(r"$k_x$", r"$\mu$m$^{-1}$"))
        ax.set_xlim(-24, 24)
        ax.set_ylim(1.6, 4.4)
    axes[0].set_ylabel(axis_label(r"$f$", "GHz"))
    axes[1].set_yticklabels([])

    ax = axes[2]
    k_pair = d36["k_um"]
    p_mel = d36["P_mel"] / max(np.max(d36["P_mel"]), 1e-30)
    k_suhl, p_suhl = pair_product_from_xt(
        d37["suhl_my_xt"], float(d37["CX"]), float(d37["DT_REC"]),
        float(d37["f_K"]), 0.0)
    ax.plot(k_pair, p_mel, color=VERMILION, lw=1.2, label="SAW MEL")
    ax.plot(k_suhl, p_suhl, color=TEAL, lw=1.2, ls="--", label="uniform Suhl")
    ax.axvline(k_mel_pump_um / 2, color=VERMILION, ls=":", lw=0.9)
    ax.axvline(0, color=TEAL, ls=":", lw=0.9)
    ax.text(k_mel_pump_um / 2, 1.02, r"$k_\mathrm{SAW}/2$",
            color=VERMILION, ha="center", va="bottom", fontsize=7)
    ax.text(0, 0.87, r"$0$", color=TEAL, ha="center", va="bottom", fontsize=7)
    ax.set_xlabel(axis_label(r"$k_1$", r"$\mu$m$^{-1}$"))
    ax.set_ylabel(r"$P(k_1)$ (norm.)")
    ax.set_xlim(-18, 24)
    ax.set_ylim(-0.03, 1.10)
    ax.legend(loc="upper left", fontsize=6.8, handlelength=1.6)
    ax.set_title("pair product", fontsize=9)

    label_panels(axes, x=-0.18, y=1.05)
    cbar = fig.colorbar(im, ax=axes[:2], shrink=0.88, aspect=22, pad=0.018)
    cbar.set_label(r"$|M_y(k,f)|^2$ (norm.)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    save_figure(fig, "fig_prl_pairrule_v2")


def fig_evidence_v2():
    """Compact main-text evidence figure combining FMR and pair-rule tests."""
    apply_prl_style()

    d20 = np.load(os.path.join(DATA_DIR, "sim20_parametric_channels.npz"))
    f_ghz = d20["f_saw_values"] * 1e-9
    b_vals = d20["B0_values"]
    b_mT = b_vals * 1e3
    sf = d20["spectra_full"]
    sm = d20["spectra_mr_only"]
    se = d20["spectra_mel_only"]

    d25 = np.load(os.path.join(DATA_DIR, "sim25_kspectrum.npz"))
    d36 = np.load(os.path.join(DATA_DIR, "sim36_twomode_correlator.npz"))
    d37 = np.load(os.path.join(DATA_DIR, "sim37_suhl_control.npz"))

    cx = float(d25["CX"])
    dt = float(d25["DT_REC"])
    f_k = float(d25["f_K"])
    k_mr_pump_um = float(d25["mr_fk_kSAW"]) * 1e-6
    k_mel_pump_um = float(d25["mel_2fk_kSAW"]) * 1e-6
    k_mr, f_mr, pwr_mr = spectrum_from_xt(d25["mr_fk_my_xt"], cx, dt)
    k_mel, f_mel, pwr_mel = spectrum_from_xt(d25["mel_2fk_my_xt"], cx, dt)

    # 1-column figure with 3 rows:
    #   row 0: Kittel dispersion (full width)         -- panel (a)
    #   row 1: MR and MEL k-spectra (side by side)    -- panels (b), (c)
    #   row 2: Pair product P(k_1) (full width)        -- panel (d)
    fig = plt.figure(figsize=(SINGLE_COL, 13.0 * CM_TO_INCH))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.0],
        hspace=0.55, wspace=0.18, left=0.18, right=0.92,
        bottom=0.07, top=0.97,
    )
    ax_disp = fig.add_subplot(gs[0, :])
    ax_mr   = fig.add_subplot(gs[1, 0])
    ax_mel  = fig.add_subplot(gs[1, 1])
    ax_pair = fig.add_subplot(gs[2, :])

    # Nudge the (b)/(c) heatmap pair slightly leftward as a group.
    # Keep their inter-panel gap (same width, same separation).
    _shift = 0.05
    for _ax in (ax_mr, ax_mel):
        _p = _ax.get_position()
        _ax.set_position([_p.x0 - _shift, _p.y0, _p.width, _p.height])

    # FMR sweep panel moved to Supplemental Material (fig_supple_fmr_sweep.pdf).
    bi = int(np.argmin(np.abs(b_mT - 50)))
    fk_50 = kittel_freq(b_vals[bi]) * 1e-9
    f_mr_peak = np.array([peak_from_spectrum(f_ghz, row) for row in sm])
    f_full_peak = np.array([peak_from_spectrum(f_ghz, row) for row in sf])
    f_mel_peak = np.array([peak_from_spectrum(f_ghz, row) for row in se])
    valid = (b_mT >= 30) & (b_mT <= 80)
    edge = ~valid
    ax_disp.plot(b_mT, f_mr_peak, "s", color=SKY_BLUE, mec=BLACK,
                 mew=0.35, label="MR peak")
    ax_disp.plot(b_mT[valid], f_full_peak[valid], "o", color=BLACK,
                 mec=BLACK, mew=0.35, label="full peak")
    ax_disp.plot(b_mT[valid], f_mel_peak[valid], "^", color=VERMILION,
                 mec=BLACK, mew=0.35, label="MEL peak")
    ax_disp.plot(b_mT[edge], f_full_peak[edge], "o", mfc="white",
                 mec="#888888", mew=0.8, ms=4.7, label="out of window")
    ax_disp.plot(b_mT[edge], f_mel_peak[edge], "^", mfc="white",
                 mec="#888888", mew=0.8, ms=4.7)
    b_fit = np.linspace(5, 125, 300) * 1e-3
    f_fit = kittel_freq(b_fit) * 1e-9
    ax_disp.plot(b_fit * 1e3, f_fit, "--", color=SKY_BLUE, lw=0.8,
                 label=r"$f_K$")
    ax_disp.plot(b_fit * 1e3, 2 * f_fit, "--", color=VERMILION, lw=0.8,
                 label=r"$2f_K$")
    ax_disp.axvspan(30, 80, color=VERMILION, alpha=0.07, lw=0)
    ax_disp.set_xlabel(axis_label(r"$B_0$", "mT"))
    ax_disp.set_ylabel(axis_label(r"$f_\mathrm{peak}$", "GHz"))
    ax_disp.set_xlim(5, 130)
    ax_disp.set_ylim(0.5, 11.0)
    handles, labels = ax_disp.get_legend_handles_labels()
    order = ["full peak", "MR peak", "MEL peak", "out of window",
             r"$f_K$", r"$2f_K$"]
    hmap = dict(zip(labels, handles))
    handles = [hmap[l] for l in order if l in hmap]
    labels = [l for l in order if l in hmap]
    ax_disp.legend(handles, labels, loc="upper left", fontsize=5.9, ncol=2,
                   columnspacing=0.65, handletextpad=0.35)

    k_disp = np.linspace(-30, 30, 400)
    f_disp = omega_k_dispersion(k_disp)
    heat_specs = [
        (ax_mr, k_mr, f_mr, pwr_mr, k_mr_pump_um, False),
        (ax_mel, k_mel, f_mel, pwr_mel, k_mel_pump_um, True),
    ]
    im = None
    for ax, k_ax, f_ax, pwr, k_ref, is_param in heat_specs:
        im = ax.pcolormesh(k_ax, f_ax, pwr, shading="auto", cmap="magma",
                           vmin=0, vmax=0.35, rasterized=True)
        ax.axhline(f_k * 1e-9, color="cyan", ls="--", lw=0.55)
        if is_param:
            ax.axvline(k_ref / 2, color="lime", ls="--", lw=0.55)
        else:
            ax.axvline(k_ref, color="lime", ls="--", lw=0.55)
        ax.set_xlabel(axis_label(r"$k_x$", r"$\mu$m$^{-1}$"))
        ax.set_xlim(-24, 24)
        ax.set_ylim(1.5, 4.5)
    ax_mr.set_ylabel(axis_label(r"$f$", "GHz"))
    ax_mel.set_yticklabels([])

    # Pair product P(k_1) for SAW MEL: recompute from S to suppress trivial
    # contamination (DC mode at k=0 and the driven SAW mode at +/- k_SAW
    # dominate the raw P and mask the centered parametric band). Zero the
    # spectrum in a +/- 1.5 um^-1 window around these spectator peaks,
    # then form P(k_1) = sqrt(S(k_1) * S(k_pump - k_1)) and Gaussian-smooth
    # over ~1 k-bin to reveal the parametric band.
    from scipy.interpolate import interp1d
    from scipy.ndimage import gaussian_filter1d
    k_pair = d36["k_um"]
    S_mel  = np.array(d36["S_mel"], dtype=float)
    k_pump_36 = float(d36["k_pump_mel_um"])
    spectator_w = 2.5
    bad = (np.abs(k_pair) < spectator_w) | \
          (np.abs(np.abs(k_pair) - k_pump_36) < spectator_w)
    S_clean = S_mel.copy()
    S_clean[bad] = 0.0
    S_at = interp1d(k_pair, S_clean, bounds_error=False, fill_value=0.0)
    P_clean = np.sqrt(np.maximum(S_clean * S_at(k_pump_36 - k_pair), 0.0))
    P_clean = gaussian_filter1d(P_clean, sigma=3.0)
    P_clean /= max(P_clean.max(), 1e-30)

    k_suhl, p_suhl = pair_product_from_xt(
        d37["suhl_my_xt"], float(d37["CX"]), float(d37["DT_REC"]),
        float(d37["f_K"]), 0.0)
    p_suhl = gaussian_filter1d(np.asarray(p_suhl, dtype=float), sigma=1.0)
    p_suhl /= max(p_suhl.max(), 1e-30)

    SUHL_GRAY = "#666666"
    ax_pair.plot(k_pair, P_clean, color=VERMILION, lw=1.4,
                 label="SAW MEL")
    ax_pair.plot(k_suhl, p_suhl, color=SUHL_GRAY, lw=1.4, ls="-",
                 label="uniform Suhl")
    k_half = k_mel_pump_um / 2.0
    ax_pair.axvline(k_half, color=VERMILION, ls=":", lw=0.55, alpha=0.6)
    ax_pair.axvline(0, color=SUHL_GRAY, ls=":", lw=0.55, alpha=0.6)

    # Labels placed above the axes, directly over each dashed line
    # (x in data coords, y in axes-fraction coords).
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax_pair.transData, ax_pair.transAxes)
    ax_pair.text(k_half, 1.02, r"$+k_\mathrm{SAW}/2$",
                 transform=trans, color=VERMILION, fontsize=7,
                 ha="center", va="bottom")
    ax_pair.text(0, 1.02, r"$k\!=\!0$",
                 transform=trans, color=SUHL_GRAY, fontsize=7,
                 ha="center", va="bottom")

    ax_pair.set_xlabel(axis_label(r"$k_1$", r"$\mu$m$^{-1}$"))
    ax_pair.set_ylabel(r"$P(k_1)$")
    ax_pair.set_xlim(-12, 12)
    ax_pair.set_ylim(-0.03, 1.10)
    ax_pair.legend(loc="upper left", fontsize=6.5, handlelength=1.5,
                   borderpad=0.3, handletextpad=0.4)

    # Colorbar for (c)(d) heatmaps in the gap between (d) and (e)
    pos_d = ax_mel.get_position()
    pos_e = ax_pair.get_position()
    gap_left = pos_d.x1 + 0.015
    gap_width = 0.012
    cax = fig.add_axes([gap_left, pos_d.y0, gap_width, pos_d.height])
    cbar = fig.colorbar(im, cax=cax)
    # Title above the colorbar (matches Fig 3 style).
    cax.set_title(r"$|M_y|^2$", fontsize=6.5, pad=2)
    cbar.ax.tick_params(labelsize=5.8)
    cbar.set_ticks([0, 0.1, 0.2, 0.3])

    label_panels([ax_disp, ax_mr, ax_mel, ax_pair], x=-0.10, y=1.04)
    save_figure(fig, "fig_prl_evidence_v2")


def fig_predictions_v2():
    apply_prl_style()
    d21 = np.load(os.path.join(DATA_DIR, "sim21_eps_threshold.npz"))
    d22 = np.load(os.path.join(DATA_DIR, "sim22_fine_angle.npz"))

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 6.2 * CM_TO_INCH))
    fig.subplots_adjust(wspace=0.32, left=0.08, right=0.97, bottom=0.20, top=0.92)

    ax = axes[0]
    eps_vals = d21["eps_values"]
    f_saw = d21["f_saw_values"]
    sf = d21["spectra_full"]
    sm = d21["spectra_mr_only"]
    mask_fk = (f_saw > 2e9) & (f_saw < 4e9)
    mask_2fk = (f_saw > 4.5e9) & (f_saw < 7.5e9)
    peak_mr = np.array([np.max(row[mask_fk]) for row in sm])
    peak_full = np.array([np.max(row[mask_2fk]) for row in sf])
    ax.loglog(eps_vals, peak_mr, "s-", color=SKY_BLUE, mec=BLACK, mew=0.35,
              label=r"MR at $f_K$")
    ax.loglog(eps_vals, peak_full, "o-", color=VERMILION, mec=BLACK, mew=0.35,
              label=r"full at $2f_K$")
    ax.axvspan(3e-5, 1e-4, color=VERMILION, alpha=0.10, lw=0)
    ax.text(5.4e-5, 0.80 * np.max(peak_full), "onset\nwindow",
            color=VERMILION, ha="center", va="center", fontsize=7)
    sub = eps_vals >= 3e-4
    slope, intercept = np.polyfit(np.log10(eps_vals[sub]), np.log10(peak_mr[sub]), 1)
    eps_line = np.array([3e-4, 4e-3])
    ax.loglog(eps_line, 10 ** intercept * eps_line ** slope, "--",
              color=SKY_BLUE, lw=0.8, alpha=0.7,
              label=rf"$\propto\varepsilon_0^{{{slope:.2f}}}$")
    ax.set_xlabel(r"strain amplitude $\varepsilon_0$")
    ax.set_ylabel(r"peak $|m_\perp|$")
    ax.legend(loc="lower right", fontsize=7)

    ax = axes[1]
    angles = d22["angles"]
    peaks = d22["peaks"]
    theta_c = float(d22["theta_c"])
    norm = float(peaks[0])
    peaks_n = peaks / norm
    zoom = angles <= 1.25
    ax.plot(angles[zoom], peaks_n[zoom], "o-", color=SKY_BLUE, mec=BLACK,
            mew=0.35, lw=1.1, label="micromagnetic")
    ax.axvline(theta_c, color=BLACK, ls=":", lw=0.9,
               label=fr"$\theta_c={theta_c:.1f}^\circ$")
    dip_i = int(np.argmin(peaks_n[zoom]))
    dip_angle = float(angles[zoom][dip_i])
    dip_val = float(peaks_n[zoom][dip_i])
    ax.annotate("tenfold\nsuppression", xy=(dip_angle, dip_val),
                xytext=(0.10, 0.34), fontsize=7, color=VERMILION,
                arrowprops=dict(arrowstyle="->", color=VERMILION, lw=0.9))
    ax.set_xlabel(r"angle $\theta$ (deg)")
    ax.set_ylabel(r"$|m_\perp|/|m_\perp(0)|$")
    ax.set_xlim(-0.03, 1.30)
    ax.set_ylim(0, 1.30)
    ax.legend(loc="upper right", fontsize=7)

    inset = ax.inset_axes([0.53, 0.14, 0.42, 0.34])
    inset.plot(angles, peaks_n, "o-", color=SKY_BLUE, ms=2.5, lw=0.7)
    inset.axvspan(0, 3, color=VERMILION, alpha=0.10, lw=0)
    inset.set_xlim(0, 45)
    inset.set_ylim(0, 31)
    inset.set_xticks([0, 20, 40])
    inset.set_yticks([0, 15, 30])
    inset.tick_params(labelsize=6)
    inset.set_title(r"full range", fontsize=6)

    label_panels(axes, x=-0.15, y=1.03)
    save_figure(fig, "fig_prl_predictions_v2")


def fig_supple_fmr_sweep():
    """FMR absorption sweep at B0 = 50 mT (Supplemental Material)."""
    apply_prl_style()
    d = np.load(os.path.join(DATA_DIR, "sim20_parametric_channels.npz"))
    f_ghz = d["f_saw_values"] * 1e-9
    b_vals = d["B0_values"]
    sf = d["spectra_full"]
    sm = d["spectra_mr_only"]
    se = d["spectra_mel_only"]
    bi = int(np.argmin(np.abs(b_vals * 1e3 - 50)))
    f_k = kittel_freq(b_vals[bi]) * 1e-9
    norm = max(np.max(sf[bi]), np.max(sm[bi]), np.max(se[bi]), 1e-12)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 6.0 * CM_TO_INCH))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.22, top=0.93)
    ax.plot(f_ghz, sf[bi] / norm, color=BLACK, lw=1.2, label="MEL+MR")
    ax.plot(f_ghz, sm[bi] / norm, "--", color=SKY_BLUE, lw=1.2,
            label="MR only")
    ax.plot(f_ghz, se[bi] / norm, ":", color=VERMILION, lw=1.7,
            label="MEL only")
    ax.axvline(f_k, color=SKY_BLUE, ls=":", lw=0.8)
    ax.axvline(2 * f_k, color=VERMILION, ls=":", lw=0.8)
    ax.text(f_k, 0.94, r"$f_K$", color=SKY_BLUE, ha="center", va="top",
            fontsize=8)
    ax.text(2 * f_k, 0.94, r"$2f_K$", color=VERMILION, ha="center",
            va="top", fontsize=8)
    ax.set_xlabel(axis_label(r"$f_\mathrm{SAW}$", "GHz"))
    ax.set_ylabel(r"$|m_\perp|$ (norm.)")
    ax.set_xlim(0.5, 10.0)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper right", fontsize=7)
    save_figure(fig, "fig_supple_fmr_sweep")


def main():
    print("Generating PRL v2 figures from cached data")
    fig_schematic_v2()
    fig_parametric_v2()
    fig_pairrule_v2()
    fig_evidence_v2()
    fig_predictions_v2()
    fig_supple_fmr_sweep()
    print("Done")


if __name__ == "__main__":
    main()
