"""Run sim25/37/38/36 + figure regeneration ONLY.

Standalone path that excludes sim20 (the slow B0 sweep). Use this when
you want to upgrade only the omega-k spectral figures (Fig 2(c)(d)(e),
Fig 3 all panels) without waiting for sim20.

Pipeline:
  1. Backup sim25/37/38/36 caches  -> *_orig.npz
  2. Remove caches so sims re-run with NX=2048
  3. Run sim25  (omega-k spectrum: NX 1024 -> 2048)
  4. Run sim37  (uniform-Suhl control: NX 1024 -> 2048)
  5. Run sim38  (reversed-SAW direction control: NX 1024 -> 2048)
  6. Run sim36  (pair-correlation, derived from sim25 cache)
  7. Re-render Fig 2 (prl_figures_v2.py)
  8. Re-render Fig 3 (make_fig3_killer_evidence.py)
  9. Copy figures to paper/prl/
 10. Re-build main.pdf and supplemental.pdf

sim20 is NOT touched. Existing sim20_parametric_channels.npz (or
sim20_parametric_channels_orig.npz if backed up earlier) is reused.

CUDA WARNING: Do NOT run this concurrently with sim20 on the same GPU.
Two mumax+ processes on one GPU compete for memory and compute, slowing
both. Either:
  (a) run on a different GPU (set CUDA_VISIBLE_DEVICES first), or
  (b) stop the sim20 process first, then run this.

Estimated total GPU time: ~6 hours (4 long mumax+ sims + tiny analysis).
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
    "sim25_kspectrum",
    "sim37_suhl_control",
    "sim38_nonreciprocal",
    "sim36_twomode_correlator",
]

CHECKPOINTS = [
    "sim37_checkpoint",
    "sim38_checkpoint",
]

SIM_SCRIPTS = [
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


def restore_sim20_if_needed():
    src = os.path.join(DATA, "sim20_parametric_channels.npz")
    bak = os.path.join(DATA, "sim20_parametric_channels_orig.npz")
    if (not os.path.exists(src)) and os.path.exists(bak):
        shutil.copy(bak, src)
        print(f"  restored sim20: using original 5-point B0 data")


def backup_caches():
    banner("STEP 1/3: Backup existing caches (sim25/37/38/36 only)")
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
    banner("STEP 2/3: Remove sim25/37/38/36 caches (forces re-run)")
    for name in CACHES + CHECKPOINTS:
        path = os.path.join(DATA, f"{name}.npz")
        if os.path.exists(path):
            os.remove(path)
            print(f"  removed: {name}.npz")
    restore_sim20_if_needed()


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
    banner(f"k-spectra pipeline complete in {elapsed:.1f} min")


if __name__ == "__main__":
    main()
