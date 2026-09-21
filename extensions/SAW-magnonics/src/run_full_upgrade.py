"""Master script: full data-quality upgrade for PRL submission.

Pipeline (executed sequentially with checkpoint/cache support):
  1. Backup existing caches  -> *_orig.npz
  2. Remove existing caches so each sim re-runs with new parameters
  3. Run sim20  (B0 sweep: 5 -> 13 points; covers 10-130 mT)
  4. Run sim25  (omega-k spectrum: NX 1024 -> 2048; doubles k-resolution)
  5. Run sim37  (uniform-Suhl control: NX 1024 -> 2048)
  6. Run sim38  (reversed-SAW direction control: NX 1024 -> 2048)
  7. Run sim36  (pair-correlation, derived from sim25 cache)
  8. Re-render Fig 2 (prl_figures_v2.py)
  9. Re-render Fig 3 (make_fig3_killer_evidence.py)
 10. Copy figures to paper/prl/
 11. Re-build main.pdf and supplemental.pdf

Each sim has its own checkpointing -> can be interrupted and resumed.
Estimated total GPU time: ~12-24 hours on a single high-end GPU.
"""

import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper", "prl")

CACHES = [
    "sim20_parametric_channels",
    "sim25_kspectrum",
    "sim37_suhl_control",
    "sim38_nonreciprocal",
    "sim36_twomode_correlator",
]

CHECKPOINTS = [
    "sim20_checkpoint",
    "sim37_checkpoint",
    "sim38_checkpoint",
]

SIM_SCRIPTS = [
    ("sim20_parametric_channels.py",
     "sim20: B0 sweep (5 -> 13 points)"),
    ("sim25_spatial_kspectrum.py",
     "sim25: omega-k (NX 1024 -> 2048)"),
    ("sim37_suhl_control.py",
     "sim37: uniform-Suhl control (NX 1024 -> 2048)"),
    ("sim38_nonreciprocal.py",
     "sim38: reversed-SAW (NX 1024 -> 2048)"),
    ("sim36_twomode_correlator.py",
     "sim36: pair correlation (from sim25 cache)"),
]

FIG_SCRIPTS = [
    ("prl_figures_v2.py", "Fig 1, 2 regeneration"),
    ("make_fig3_killer_evidence.py", "Fig 3 regeneration"),
]


def banner(msg):
    line = "=" * 78
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def backup_caches():
    banner("STEP 1/3: Backup existing caches")
    for name in CACHES:
        src = os.path.join(DATA, f"{name}.npz")
        dst = os.path.join(DATA, f"{name}_orig.npz")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"  backup: {name}.npz -> {name}_orig.npz")
        elif os.path.exists(dst):
            print(f"  skip:   {name}_orig.npz already exists")
        else:
            print(f"  skip:   {name}.npz not found")


def clear_caches():
    banner("STEP 2/3: Remove old caches so sims re-run with new params")
    for name in CACHES + CHECKPOINTS:
        path = os.path.join(DATA, f"{name}.npz")
        if os.path.exists(path):
            os.remove(path)
            print(f"  removed: {name}.npz")


def run_python(script, label):
    print(f"\n--- {label} ---", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, script], cwd=SRC)
    elapsed = (time.time() - t0) / 60.0
    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f} min): {script}")
        sys.exit(result.returncode)
    print(f"  done ({elapsed:.1f} min): {script}", flush=True)


def run_sims():
    banner("STEP 3/3: Run upgraded simulations + figure regeneration")
    for script, label in SIM_SCRIPTS:
        run_python(script, label)
    for script, label in FIG_SCRIPTS:
        run_python(script, label)


def copy_figs_to_paper():
    banner("Copy figures to paper/prl/")
    pairs = [
        ("fig_prl_evidence_v2.pdf", "fig_prl_evidence_v2.pdf"),
        ("fig_prl_killer_v3.pdf",   "fig_prl_reversed_coherence.pdf"),
        ("fig_prl_schematic_v2.pdf","fig_prl_schematic_v2.pdf"),
    ]
    for src_name, dst_name in pairs:
        src = os.path.join(FIGS, src_name)
        dst = os.path.join(PAPER, dst_name)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"  copied: {src_name} -> paper/prl/{dst_name}")
        else:
            print(f"  missing: {src_name}")


def rebuild_paper():
    banner("Re-build main.pdf and supplemental.pdf")
    for tex in ["main.tex", "supplemental.tex"]:
        for cmd in [
            ["pdflatex", "-interaction=nonstopmode", tex],
            ["bibtex", tex.replace(".tex", "")],
            ["pdflatex", "-interaction=nonstopmode", tex],
            ["pdflatex", "-interaction=nonstopmode", tex],
        ]:
            subprocess.run(cmd, cwd=PAPER, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        pdf = os.path.join(PAPER, tex.replace(".tex", ".pdf"))
        if os.path.exists(pdf):
            sz = os.path.getsize(pdf) / 1024.0
            print(f"  built: {tex.replace('.tex', '.pdf')} ({sz:.0f} KB)")
        else:
            print(f"  FAILED to build {tex}")


def main():
    t0 = time.time()
    backup_caches()
    clear_caches()
    run_sims()
    copy_figs_to_paper()
    rebuild_paper()
    elapsed = (time.time() - t0) / 60.0
    banner(f"Pipeline complete in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
