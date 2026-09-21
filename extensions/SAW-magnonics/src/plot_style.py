"""Shared plotting style for mumax+ extension papers.

Follows PRB formatting guidelines:
  - 8.6 cm (single column) figure width
  - Times/STIX serif font (PRB-compatible)
  - Axis labels: 8-10 pt, Tick labels: 7-9 pt
  - Panel labels (a), (b): 10-12 pt bold
  - Legend: 8-9 pt
  - Units in parentheses, variables in italics, units in roman
  - Okabe-Ito colorblind-safe palette
  - No figure titles, frame box on, ticks inward
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Okabe-Ito palette ----------
SKY_BLUE  = '#56B4E9'
VERMILION = '#D55E00'
TEAL      = '#009E73'
YELLOW    = '#F0E442'
PINK      = '#CC79A7'
ORANGE    = '#E69F00'
BLUE      = '#0072B2'
BLACK     = '#000000'

COLORS_3 = [SKY_BLUE, VERMILION, TEAL]
COLORS_4 = [SKY_BLUE, VERMILION, TEAL, YELLOW]
COLORS_6 = [SKY_BLUE, VERMILION, TEAL, YELLOW, PINK, ORANGE]
COLORS_7 = [SKY_BLUE, VERMILION, TEAL, YELLOW, PINK, ORANGE, BLUE]

# Markers for distinguishing data series
MARKERS = ['o', '^', 's', 'D', 'v', 'P', 'X']

# ---------- Figure dimensions ----------
# ---------------------------------------------------------------------------
# Z6 octupole-state colour key.
#
# The six Mn3Sn octupole domain states are related by 60 deg rotations, so a
# hue wheel at 60 deg steps is the natural encoding. Value is solved per hue
# for equal relative luminance (0.42), which a plain hue wheel does not give
# (yellow would come out far lighter than blue). twilight_shifted, used
# earlier, passes through near-white at its midpoint, so one of six evenly
# sampled states became invisible on a white background.
#
# Index = octupole angle / 60 deg.
#
# Restored to the palette of the earlier manuscript version (2026-06-12), which
# sampled twilight_shifted at k/6, so the domain maps match that version. Note
# the cost this reintroduces: the six are NOT equal-luminance -- 0 deg is nearly
# black (0.014) and 180 deg nearly white (0.713) -- so a state can read as
# background on a white page. The equal-luminance wheel that replaced it is
# kept below for reference.
Z6_COLORS = ['#301437', '#5e45a6', '#7ca2c2',
             '#e2d9e2', '#c6896c', '#8d2b50']

Z6_COLORS_EQUILUM = ['#e74a4a', '#717124', '#2b852b',
                     '#287d7d', '#5252ff', '#d143d1']


def z6_color(phi_deg):
    """Colour for an octupole state at in-plane angle phi (degrees)."""
    return Z6_COLORS[int(round((phi_deg % 360) / 60.0)) % 6]


def z6_cmap():
    """Discrete colormap over the six Z6 states, for domain maps."""
    from matplotlib.colors import ListedColormap
    return ListedColormap(Z6_COLORS, name='z6')


def z6_cmap_cyclic(n=256):
    """Continuous cyclic colormap through the same six state colours.

    For maps of a measured octupole angle, where domain interiors sit on the
    Z6 states but wall pixels vary continuously between them. Anchors are the
    six Z6_COLORS at phi = 0, 60, ... 300 deg, so a pixel at an exact state
    angle gets exactly the colour that state has in z6_cmap(), while wall
    pixels interpolate between the two neighbouring states. Use with
    vmin=0, vmax=2*pi (or 0, 360).
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        'z6_cyclic', Z6_COLORS + [Z6_COLORS[0]], N=n)


SINGLE_COL_CM = 8.6
DOUBLE_COL_CM = 17.8
CM_TO_INCH = 1 / 2.54

SINGLE_COL = SINGLE_COL_CM * CM_TO_INCH   # ~ 3.386 inch
DOUBLE_COL = DOUBLE_COL_CM * CM_TO_INCH   # ~ 7.008 inch


def apply_style():
    """Apply PRB-compatible matplotlib rcParams."""
    mpl.rcParams.update({
        # Font — Arial throughout (user convention). Helvetica is not
        # installed on this machine, so listing it first silently fell back;
        # Arial is named explicitly to keep figures reproducible.
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans',
                            'DejaVu Sans'],
        'font.size': 8,
        # Math must match the text face. 'stixsans' still pulls STIXGeneral
        # (a SERIF) for most glyphs, which left every symbol (eta, tau, Delta)
        # in serif next to Arial labels. A custom fontset keeps math in Arial.
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold',
        'mathtext.sf': 'Arial',
        # \mathcal{} must be named explicitly: mathtext.cal defaults to
        # 'cursive', which Windows resolves to Comic Sans MS and then embeds
        # in the figure PDF. STIX italic keeps a script look without it.
        'mathtext.cal': 'STIXGeneral:italic',
        'mathtext.default': 'it',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,

        # Axes
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.linewidth': 0.6,
        'axes.formatter.use_mathtext': True,

        # Ticks
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.top': True,
        'ytick.right': True,

        # Legend — PRL style: no frame box
        'legend.fontsize': 7,
        'legend.frameon': False,
        'legend.loc': 'best',

        # Lines
        'lines.linewidth': 1.2,
        'lines.markersize': 5,

        # Figure
        'figure.dpi': 300,
        'figure.facecolor': 'white',
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'savefig.facecolor': 'none',
        'savefig.transparent': True,
    })


def single_panel(height_cm=6.0):
    """Create a single-column, single-panel figure."""
    apply_style()
    fig, ax = plt.subplots(
        figsize=(SINGLE_COL, height_cm * CM_TO_INCH))
    _style_ax(ax)
    return fig, ax


def double_panel_h(height_cm=6.0, wspace=0.35):
    """Create a single-column figure with two horizontal panels."""
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, height_cm * CM_TO_INCH))
    fig.subplots_adjust(wspace=wspace)
    for ax in axes:
        _style_ax(ax)
    return fig, axes


def double_panel_v(height_cm=10.0, hspace=0.35):
    """Create a single-column figure with two vertical panels."""
    apply_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(SINGLE_COL, height_cm * CM_TO_INCH))
    fig.subplots_adjust(hspace=hspace)
    for ax in axes:
        _style_ax(ax)
    return fig, axes


def quad_panel(height_cm=12.0, wspace=0.35, hspace=0.35):
    """Create a 2x2 panel figure."""
    apply_style()
    fig, axes = plt.subplots(
        2, 2, figsize=(DOUBLE_COL, height_cm * CM_TO_INCH))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    for row in axes:
        for ax in row:
            _style_ax(ax)
    return fig, axes


def add_panel_label(ax, label, x=-0.12, y=1.06, fontsize=9):
    """Add panel label (a), (b) in uniform sans-serif style (9 pt, normal).

    Uses axes-relative coordinates so the label sits just outside the
    top-left corner of every panel regardless of figure layout.

    The *fontsize* is in absolute points, so it prints at the same
    physical size on A4 for both single-column (8.6 cm) and
    double-column (17.8 cm) figures — provided each figure is created
    at its target width via the helpers in this module.

    Style: Helvetica/Arial sans-serif (inherited from rcParams),
    weight=normal (no bold), uniform 9 pt across all panels regardless
    of figure column width.
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='normal', va='bottom', ha='left',
            fontfamily='sans-serif')


def add_panel_label_outside(fig, ax, label, dx=-0.020, dy=0.008, fontsize=10):
    """Place panel label outside top-left of an axis in figure coordinates.

    Kept for backward compatibility; prefer ``add_panel_label`` or
    ``label_panels`` for new code.
    """
    bb = ax.get_position()
    x = bb.x0 + dx
    y = bb.y1 + dy
    fig.text(x, y, label, fontsize=fontsize, fontweight='normal',
             va='bottom', ha='left', fontfamily='sans-serif')


def label_panels(axes, labels=None, x=-0.12, y=1.06, fontsize=9):
    """Auto-label all panels with (a), (b), ... ensuring grid alignment.

    Every panel receives its label at the *same* axes-relative (x, y),
    which guarantees that labels are aligned horizontally across columns
    and vertically down rows in any regular subplot grid.

    Parameters
    ----------
    axes : Axes or array-like of Axes
        Return value of ``plt.subplots`` — single Axes, 1-D, or 2-D array.
    labels : list of str, optional
        Custom labels.  Defaults to ``['(a)', '(b)', '(c)', ...]``.
    x, y : float
        Axes-relative position (identical for every panel).
    fontsize : float
        Font size in absolute points (same on 1-col and 2-col A4).
    """
    import numpy as np
    flat = list(np.atleast_1d(axes).flat)
    if labels is None:
        labels = [f'({chr(ord("a") + i)})' for i in range(len(flat))]
    for ax, lbl in zip(flat, labels):
        add_panel_label(ax, lbl, x=x, y=y, fontsize=fontsize)


def axis_label(var, unit=None):
    r"""Format an axis label following PRB conventions.

    PRB style rules applied automatically:
      - Variables in LaTeX math mode → rendered *italic*
      - Text-type subscripts via ``\mathrm{}`` → rendered upright
        (e.g. ``$M_\mathrm{s}$``, ``$H_\mathrm{ext}$``)
      - Variable subscripts stay italic (e.g. ``$M_x$``)
      - Units in parentheses, upright roman font

    Parameters
    ----------
    var : str
        Variable in LaTeX math notation, e.g. ``r'$\mu_0 H$'``.
    unit : str, optional
        Unit string, e.g. ``'T'``, ``'A/m'``, ``'ns'``.

    Returns
    -------
    str
        Formatted label, e.g. ``r'$\mu_0 H$ (T)'``.

    Examples
    --------
    >>> ax.set_xlabel(axis_label(r'$\mu_0 H_\mathrm{ext}$', 'T'))
    >>> ax.set_ylabel(axis_label(r'$m_z$'))
    >>> ax.set_xlabel(axis_label(r'$J$', r'MA/cm$^2$'))
    """
    if unit is None:
        return var
    return f'{var} ({unit})'


def ensure_output_dirs(script_dir):
    """Create and return (figure_dir, data_dir)."""
    fig_dir = os.path.join(script_dir, "figures")
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return fig_dir, data_dir


def save_figure(fig, out_base, formats=("pdf", "eps")):
    """Save a figure in vector formats for manuscript use."""
    for ext in formats:
        fig.savefig(f"{out_base}.{ext}", format=ext)


# Alias for backward compatibility with scripts using setup_style()
setup_style = apply_style


def smart_legend(ax, fontsize=7, loc='best', frameon=False, **kwargs):
    """Legend with sensible defaults for PRB-format figures."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(handles, labels, loc=loc, fontsize=fontsize,
                     frameon=frameon, **kwargs)


def _style_ax(ax):
    """Apply per-axis styling (frame, ticks)."""
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
