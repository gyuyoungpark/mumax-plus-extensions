"""Shared harness for the commensurate SAW-magnonics GPU campaign.

Engine API used here is COPIED from, not invented for, the existing sims:
  World / Grid / Ferromagnet, msat aex alpha B1 Kmr anisU magnetization
  bias_magnetic_field enable_demag temperature, ChiralSurfaceAcousticWave.apply,
  world.timesolver.{timestep, adaptive_timestep, time, run},
  magnet.magnetization.eval()  ->  (3, NZ, NY, NX)
Sources: src/sim40_eps_kresolved.py, src/sim43_clean_seed.py,
         src/sim45_dispersion.py, src/sim46_resonant_confirm.py,
         src/saw_chiral.py, src/bindings/wrap_ferromagnet.cpp,
         src/bindings/wrap_timesolver.cpp.
Anything NOT verified in those files is guarded by try/except and recorded as
"NOT DETERMINABLE" rather than guessed.

Conventions for ALL analysis: import revision_check/saw_analysis.py.  Do not
re-implement FFTs, band bookkeeping or the pair correlator here.
"""

import collections.abc as _collections_abc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _conditions as _cond                                       # noqa: E402
import _gate                                                      # noqa: E402
import _provenance as _prov                                       # noqa: E402

# ---------------------------------------------------------------- paths -----
HERE      = os.path.dirname(os.path.abspath(__file__))
RC_DIR    = os.path.abspath(os.path.join(HERE, ".."))              # revision_check
EXT_DIR   = os.path.abspath(os.path.join(RC_DIR, ".."))            # SAW-magnonics
SRC_DIR   = os.path.join(EXT_DIR, "src")
OUT_DIR   = os.path.join(RC_DIR, "runs", "out")                    # NEVER data/
KERNEL_CU = os.path.abspath(os.path.join(EXT_DIR, "..", "..", "src",
                                         "physics", "chiralsawfield.cu"))
os.makedirs(OUT_DIR, exist_ok=True)
for p in (SRC_DIR, EXT_DIR, RC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ------------------------------------------------------------ constants -----
GAMMA = 1.76e11          # rad/s/T   (src/saw.py)
MU0   = 4e-7 * np.pi
KB    = 1.380649e-23
V_SAW = 3500.0           # m/s       (src/sim40_eps_kresolved.py)
XI    = 0.68             # Rayleigh ellipticity (src/sim40_eps_kresolved.py)

# Cell size: identical to every prior sim (sim40/43/45/46: 5 x 10 x 20 nm).
CX, CY, CZ = 5e-9, 10e-9, 20e-9
NY, NZ     = 8, 1
DT_STEP    = 2e-13       # fixed, non-adaptive, as in every prior sim

# Engine settings that build() applies and that every prior version of this
# harness passed as bare literals inside build().  They are named constants so
# that ONE place feeds both the engine call and the condition set: a value that
# build() uses and the condition set does not know about is exactly how XI, NY,
# NZ, CY, CZ, enable_mel and saw_direction stayed out of every hash.
PBC             = (2, 2, 0)      # world.pbc_repetitions
SAW_PHASE       = 0.0            # ChiralSurfaceAcousticWave(phase=...)
SAW_DIRECTION   = +1             # +1 = +x; -1 flips the recorded wavevector
ENABLE_MEL      = True           # magnetoelastic channel, build()'s default
ENABLE_BARNETT  = False          # Barnett channel (see audit sec.2: the engine
                                 # sign is unpatched, so no run may enable it)
SEED_KIND       = "bandlimited_yflat_randphase"   # make_seed's only shape

# Material sets.  LITERATURE values first (CLAUDE.md sec.4).
YIG_LIT = dict(MS=140e3, AEX=3.65e-12, ALPHA=5e-4, B1=-0.35e6, KMR=0.0,
               tag="YIGlit")
# sim40's parameter set.  |B1| is 25x the literature YIG value and K_mr is
# switched on; it is carried ONLY to reproduce the published benchmark in a
# commensurate box.  It is NOT a literature YIG parameter set.
SIM40   = dict(MS=140e3, AEX=3.65e-12, ALPHA=5e-4, B1=-8.8e6, KMR=1.0e6,
               tag="sim40set")

# ---------------------------------------------------------------- boxes -----
def box(n_periods, dx=CX, f_saw=6.0e9, v_saw=V_SAW):
    """Commensurate box of n SAW periods.  Raises unless NX is an integer.

    q L = 2 pi n exactly, so the pump sits on bin n and q/2 on bin n/2
    (the latter only when n is EVEN -- that is the whole point of the redesign).
    """
    lam = v_saw / f_saw
    L   = n_periods * lam
    nxf = L / dx
    nx  = int(round(nxf))
    if abs(nxf - nx) > 1e-9:
        raise ValueError(
            "n=%d at dx=%.6f nm gives NX=%.6f, not an integer. At dx=5 nm, "
            "lambda/dx = 116.666... so NX is integer only for n divisible by 3; "
            "even AND commensurate therefore means n in {6,12,18,24,...}."
            % (n_periods, dx * 1e9, nxf))
    if n_periods % 2:
        raise ValueError("n=%d is ODD: q/2 = (n/2) dk is NOT a grid point, "
                         "which is the ambiguity being removed." % n_periods)
    dk = 2 * np.pi / L
    q  = 2 * np.pi * f_saw / v_saw
    return dict(n=n_periods, NX=nx, dx=dx, L=L, dk=dk, q=q,
                bin_q=n_periods, bin_qhalf=n_periods // 2,
                f_saw=f_saw, v_saw=v_saw,
                tag="n%d_NX%d_dx%.4fnm" % (n_periods, nx, dx * 1e9))

BOX_PRIMARY   = box(12)                    # NX=1400, L=7.000 um, dx=5 nm
BOX_CROSS     = box(18)                    # NX=2100, L=10.500 um, dx=5 nm
BOX_CELLCONV  = box(12, dx=(V_SAW / 6.0e9) / 125)   # NX=1500, dx=4.666667 nm

# ----------------------------------------------------------- provenance -----
def sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1048576), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:                                   # noqa: BLE001
        return "NOT DETERMINABLE (%s)" % exc


def engine_build_info():
    """Whatever the installed mumaxplus actually exposes.  Nothing invented:
    every lookup is guarded and reports NOT DETERMINABLE on failure."""
    info = {}
    try:
        import mumaxplus
        info["mumaxplus_file"] = getattr(mumaxplus, "__file__", "NOT DETERMINABLE")
        info["mumaxplus_version_attr"] = getattr(mumaxplus, "__version__",
                                                 "NOT DETERMINABLE")
        try:
            from mumaxplus import _cpp
            info["cpp_module_file"] = getattr(_cpp, "__file__", "NOT DETERMINABLE")
            info["cpp_module_sha256"] = sha256_file(getattr(_cpp, "__file__", ""))
        except Exception as exc:                               # noqa: BLE001
            info["cpp_module_file"] = "NOT DETERMINABLE (%s)" % exc
    except Exception as exc:                                   # noqa: BLE001
        info["mumaxplus_import"] = "NOT DETERMINABLE (%s)" % exc
    try:
        import importlib.metadata as md
        info["dist_version"] = md.version("mumaxplus")
    except Exception as exc:                                   # noqa: BLE001
        info["dist_version"] = "NOT DETERMINABLE (%s)" % exc
    repo = os.path.abspath(os.path.join(EXT_DIR, "..", ".."))
    for cmd, key in ((["git", "rev-parse", "HEAD"], "git_head"),
                     (["git", "status", "--porcelain"], "git_dirty")):
        try:
            info[key] = subprocess.run(cmd, cwd=repo, capture_output=True,
                                       text=True, timeout=20).stdout.strip()[:400]
        except Exception as exc:                               # noqa: BLE001
            info[key] = "NOT DETERMINABLE (%s)" % exc
    return info


def provenance(script_path, params):
    """Everything a later reader needs to know WHICH code made this file.

    Past runs cannot be attributed from mtimes or present hashes; this block is
    the fix, and it is written into EVERY output npz from now on.

    A source hash and a binary hash recorded side by side do not link the two
    (audit 2026-09-18 sec.9).  `_provenance.link_summary()` writes the build
    manifest that does link them -- source files + CMake build configuration ->
    the build tree's own output -> the installed binary -- and CHECKS the stored
    manifest before refreshing it, so a rebuild between runs is visible here as
    `build_manifest_check = MISMATCH`.  `build_link_status` is one of four
    states and is never a boolean; see ../PROVENANCE_BUILD_MANIFEST.md.
    """
    link = _prov.link_summary()
    return dict(
        schema="saw_revision_check/provenance/2",
        **link,
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=platform.node(), platform=platform.platform(),
        python=sys.version.split()[0], numpy=np.__version__,
        argv=" ".join(sys.argv),
        script=os.path.abspath(script_path),
        sha256_script=sha256_file(script_path),
        sha256_harness=sha256_file(os.path.join(HERE, "_harness.py")),
        sha256_chiralsawfield_cu=sha256_file(KERNEL_CU),
        path_chiralsawfield_cu=KERNEL_CU,
        sha256_saw_chiral_py=sha256_file(os.path.join(SRC_DIR, "saw_chiral.py")),
        sha256_saw_py=sha256_file(os.path.join(SRC_DIR, "saw.py")),
        sha256_saw_analysis_py=sha256_file(os.path.join(RC_DIR, "saw_analysis.py")),
        engine=json.dumps(engine_build_info()),
        params=json.dumps(params, default=str, sort_keys=True),
    )

# ----------------------------------------------------------------- seed -----
def thermal_mode_amplitude(k, T, B0, Ms, Aex, V_tot):
    """a_th: amplitude of ONE spin-wave mode m = a cos(kx+phi) at temperature T.

    E = (1/4) Ms V_tot B_k a^2 with two quadratures, <E> = k_B T:
        a = sqrt(4 k_B T / (Ms V_tot B_k)),
        B_k = sqrt((B0 + 2A k^2/Ms)(B0 + mu0 Ms + 2A k^2/Ms)).
    At the n=12 box and k = q/2 this gives ~9.8e-3 at 300 K -- i.e. the
    sigma = 1e-4 per-cell seed of sim43-49 sits 9.25 e-foldings BELOW the room
    temperature bath (equivalent magnon temperature ~3e-6 K).  See
    revision_check/out/seed_calibration.txt.
    """
    ex = 2 * Aex * np.asarray(k) ** 2 / Ms
    Bk = np.sqrt((B0 + ex) * (B0 + MU0 * Ms + ex))
    return np.sqrt(4 * KB * T / (Ms * V_tot * Bk))


def make_seed(rng_seed, NX, dx, a_mode, k_cut, n_y=NY, n_z=NZ,
              components=("y", "z")):
    """Band-limited, y/z-uniform, random-phase transverse seed.

    Why this shape:
      * y/z-uniform (k_y = k_z = 0): the observable is the y-averaged m_y(x,t),
        so any k_y != 0 content is invisible to it and would only add energy
        that never enters the measurement.  sim43's per-cell Gaussian puts
        7/8 of its energy there and then averages it away.
      * band-limited to |k| <= k_cut: FLAT inside the band, so no k in the
        region of interest is preferred a priori, while the per-mode amplitude
        can be set to a physical (thermal) value without the per-cell canting
        running away.  A genuinely k-flat seed at the thermal per-mode
        amplitude would need sigma_cell ~ 1.5 -- impossible.
      * amplitude a_mode is the PER-MODE amplitude, so it is directly
        comparable with thermal_mode_amplitude().
      * rng_seed is an explicit integer.  NEVER derive it from hash(): CPython
        salts str hashing per process (PYTHONHASHSEED), so sim43's
        `1000*hash(label) % 7919 + s` produced a DIFFERENT seed on every run and
        the value was never written to the npz.

    Returns (mag, meta) with mag shape (3, NZ, NY, NX), |m| = 1 exactly.
    """
    rng = np.random.default_rng(int(rng_seed))
    dk  = 2 * np.pi / (NX * dx)
    modes = np.arange(1, int(np.floor(k_cut / dk)) + 1)
    if modes.size == 0:
        raise ValueError("k_cut smaller than one bin")
    x = (np.arange(NX) + 0.5) * dx        # cell centres, matching the kernel
    mag = np.zeros((3, n_z, n_y, NX), dtype=np.float64)
    phases = {}
    prof = {}
    for comp in components:
        ph = rng.uniform(0.0, 2 * np.pi, modes.size)
        phases[comp] = ph
        p = np.zeros(NX)
        for j, mno in enumerate(modes):
            p += a_mode * np.cos(mno * dk * x + ph[j])
        prof[comp] = p
        mag[{"x": 0, "y": 1, "z": 2}[comp]] = p[None, None, :]
    mag[0] = np.sqrt(np.clip(1.0 - mag[1] ** 2 - mag[2] ** 2, 0.0, None))
    meta = dict(rng_seed=int(rng_seed), a_mode=float(a_mode),
                k_cut=float(k_cut), mode_numbers=modes.astype(np.int32),
                n_modes=int(modes.size),
                phases_y=phases.get("y", np.zeros(0)),
                phases_z=phases.get("z", np.zeros(0)),
                sigma_cell_y=float(prof.get("y", np.zeros(1)).std()),
                max_canting=float(np.hypot(mag[1], mag[2]).max()),
                seed_kind="bandlimited_yflat_randphase")
    return mag, meta


def inject_plane_wave(mag, k1, amp, phase, dx):
    """Add ONE coherent transverse mode m_y += amp*cos(k1 x + phase), renormalise.

    k1 must be an exact grid wavevector (integer multiple of dk) or the
    injection itself leaks and the test is worthless.
    """
    NX = mag.shape[3]
    x = (np.arange(NX) + 0.5) * dx
    mag = mag.copy()
    mag[1] += amp * np.cos(k1 * x + phase)[None, None, :]
    nrm = np.sqrt(mag[0] ** 2 + mag[1] ** 2 + mag[2] ** 2)
    return mag / nrm

# ---------------------------------------------------------------- engine ----
def kittel_field(f_target, Ms):
    """B0 solving f = (gamma/2pi) sqrt(B0 (B0 + mu0 Ms))  (src/sim40)."""
    w = 2 * np.pi * f_target
    return (-MU0 * Ms + np.sqrt((MU0 * Ms) ** 2 + 4 * (w / GAMMA) ** 2)) / 2


def kittel_freq(B0, Ms):
    return GAMMA * np.sqrt(B0 * (B0 + MU0 * Ms)) / (2 * np.pi)


def build(bx, mat, B0, eps0, f_saw, mag_init, temperature=0.0,
          enable_mel=ENABLE_MEL, saw_direction=SAW_DIRECTION, conditions=None):
    """World + magnet, using ONLY calls present in the existing sims.

    `conditions` is the condition set the caller declared for this run.  When it
    is given, every engine setting this function is about to apply is compared
    against it BEFORE the world is allocated, and a contradiction refuses the
    build naming the fields.  That is what stops the other half of the audit's
    checkpoint defect: a run built with enable_mel=False or saw_direction=-1
    while its declaration says otherwise would otherwise inherit -- and be
    resumed from -- a forward/MEL-on checkpoint.
    """
    if conditions is None:
        raise _cond.PreconditionHalt("build requires a complete condition set")
    conditions.require_execution("build", nx=bx["NX"])
    fresh_build = condition_env()
    if any(conditions.value(f.path) != fresh_build.get(f.name)
           for f in _cond.FIELDS if f.group == "build"):
        raise _cond.PreconditionHalt("build: source/binary conditions changed since declaration")
    if conditions is not None:
        applied = dict(decl_grid_NX=bx["NX"], decl_grid_CX=bx["dx"])
        want = {
            "grid.NX": bx["NX"], "grid.CX": bx["dx"],
            "grid.NY": NY, "grid.NZ": NZ, "grid.CY": CY, "grid.CZ": CZ,
            "grid.PBC": list(PBC),
            "pump.f_saw": f_saw, "pump.q": 2 * np.pi * f_saw / bx["v_saw"],
            "pump.eps_0": eps0, "pump.xi": XI,
            "pump.phase": SAW_PHASE, "pump.direction": saw_direction,
            "pump.enable_mel": enable_mel,
            "pump.enable_barnett": ENABLE_BARNETT,
            "material.Msat": mat["MS"], "material.Aex": mat["AEX"],
            "material.alpha": mat["ALPHA"], "material.B1": mat["B1"],
            "material.Kmr": mat["KMR"], "material.B0": B0,
            "material.temperature": temperature,
            "numerics.dt_step": DT_STEP,
        }
        bad = []
        for path, val in sorted(want.items()):
            have = conditions.value(path)
            if _gate.normalise(have) != _gate.normalise(val):
                bad.append("%s: declared %r, build() would apply %r"
                           % (path, have, val))
        if bad:
            raise _cond.PreconditionHalt(
                "build: the engine settings contradict the declared conditions "
                "in %d field(s); refusing to allocate a world whose record "
                "would be wrong\n    %s" % (len(bad), "\n    ".join(bad)))
        del applied
    from mumaxplus import World, Grid, Ferromagnet
    from saw_chiral import ChiralSurfaceAcousticWave

    NX = bx["NX"]
    world = World((bx["dx"], CY, CZ), mastergrid=Grid((NX, NY, 0)),
                  pbc_repetitions=tuple(PBC))
    magnet = Ferromagnet(world, Grid((NX, NY, NZ)))
    magnet.msat  = mat["MS"]
    magnet.aex   = mat["AEX"]
    magnet.alpha = mat["ALPHA"]
    if enable_mel:
        magnet.B1 = mat["B1"]
    magnet.magnetization = mag_init
    magnet.bias_magnetic_field = (B0, 0, 0)
    magnet.enable_demag = True
    magnet.temperature = float(temperature)

    saw_meta = dict(applied=False)
    if eps0 and eps0 > 0:
        lam = bx["v_saw"] / f_saw
        saw = ChiralSurfaceAcousticWave(
            frequency=f_saw, wavelength=lam, amplitude=eps0,
            direction='x', phase=SAW_PHASE, ellipticity=XI,
            K_mr=mat["KMR"], enable_barnett=ENABLE_BARNETT)
        saw.apply(magnet, Msat=mat["MS"], enable_mel=enable_mel)
        if saw_direction < 0 and hasattr(magnet, 'saw_wavevector'):
            magnet.saw_wavevector = -abs(float(magnet.saw_wavevector))
        saw_meta = dict(applied=True, f_saw=f_saw, lam=lam, eps0=eps0,
                        K_mr=mat["KMR"], xi=XI,
                        gpu_kernel=bool(hasattr(magnet, 'enable_saw')))
        # Record what the ENGINE thinks q is, not what we think it is.
        for attr in ("saw_wavevector", "saw_frequency", "saw_amplitude",
                     "saw_ellipticity", "saw_phase", "saw_direction",
                     "enable_saw", "saw_enable_mel", "saw_enable_barnett"):
            try:
                saw_meta["engine_" + attr] = float(getattr(magnet, attr))
            except Exception:                                   # noqa: BLE001
                try:
                    saw_meta["engine_" + attr] = str(getattr(magnet, attr))
                except Exception:                               # noqa: BLE001
                    saw_meta["engine_" + attr] = "NOT DETERMINABLE"

    world.timesolver.timestep = DT_STEP
    world.timesolver.adaptive_timestep = False
    return world, magnet, saw_meta


def sample_my(magnet):
    """y,z-averaged m_y(x).  Same reduction as sim40/43/45/46."""
    return magnet.magnetization.eval()[1].mean(axis=(0, 1))


def analysis_input_gate(record, box, material, f_saw, dt_rec):
    """Revalidate stored input before an analysis can authorize another stage."""
    try:
        body, reason = _cond.read(record)
        if body is None:
            raise _cond.PreconditionHalt(reason)
        declaration = json.loads(str(record["run_manifest"]))
        stored = _cond.stored_set(body)
        current = conditions(dict(declaration, box=box, material=material, f_saw=f_saw),
                             nt=stored.value("numerics.nt"), dt_rec=dt_rec,
                             block_records=stored.value("numerics.block_records"))
        current.require_execution("analysis input", nx=box["NX"])
        gate = _cond.state("analysis_input", "checkpoint", record, current,
                           _cond.ALL_FIELDS, require_build_link=True)
        gate.require()
        nt = int(current.value("numerics.nt"))
        if (not bool(record["done"]) or int(record["next_index"]) != nt
                or record["my_xt"].shape != (nt, box["NX"])
                or not np.isfinite(record["my_xt"]).all()):
            raise _cond.PreconditionHalt("incomplete or malformed checkpoint array")
        for key in ("eps0", "B0", "rng_seed"):
            if key in record and key in declaration and float(record[key]) != float(declaration[key]):
                raise _cond.PreconditionHalt("raw/stamped contradiction: " + key)
        return gate
    except (KeyError, TypeError, ValueError, _gate.GateHalt) as exc:
        return _gate.Gate("analysis_input", _gate.NOT_EVALUABLE, str(exc))

# ------------------------------------------------------ checkpoint/resume ---
class BlockRun:
    """Checkpointed integration.  CLAUDE.md sec.1: save at BLOCK granularity,
    auto-resume if the checkpoint exists, fresh start if not.

    Mid-run resume restores BOTH the magnetisation AND world.timesolver.time.
    Restoring the time is not optional: chiralsawfield.cu computes the pump
    phase as sawK*x - sawOmega*t + sawPhi from the SOLVER time, so a resume
    that restarts the clock at 0 silently re-phases the pump.

    RUN IDENTITY (added 2026-09-18, audit section 8, P0-4).  A checkpoint used
    to be reused on the strength of its `done` flag alone: the auditor changed
    dt_rec from 1 to 0.123 and the stored result was still accepted.  Every
    parameter that changes the output now goes into a normalised manifest which
    is hashed and stored beside the data, and a checkpoint is reused only if the
    hashes agree.  A mismatch raises _gate.RunIdentityMismatch naming the
    differing fields; a checkpoint with no manifest at all cannot be verified
    and is refused for the same reason.  `manifest` is the caller's physical
    parameter dict -- the same dict that goes into provenance(params) -- and
    must NOT contain wall-clock, host or argv fields.

    PROVENANCE CHAIN (same entry).  A resumed run appends its own segment record
    instead of overwriting the earlier segment's attribution, so who computed
    records [0, k) stays readable after another process computes [k, nt).
    """

    def __init__(self, path, nt, nx, dt_rec, block_records=500,
                 manifest=None, verified=_cond.GROUPS):
        self.path, self.nt, self.nx = path, int(nt), int(nx)
        self.dt_rec, self.block = float(dt_rec), int(block_records)
        self.i0 = 0
        self.mag_state = None
        self.t_state = 0.0
        self.done = False
        self.prov_chain = []
        self._own_segment = None
        self._seg_start = 0
        self.initial_state_sha256 = None

        # -- the CONDITION SET of this run, before anything is loaded ---------
        # Same mechanism as every gate certificate (runs/_conditions.py): the
        # field list lives in ONE place, the harness constants are read live, and
        # the build group folds the forward provenance in, so a rebuilt kernel,
        # a changed XI and a changed dt_rec are all refused by the same
        # comparison.  There is no per-call-site list of "what matters".
        self.declaration = dict(manifest or {})
        self.conditions = conditions(
            self.declaration, source="BlockRun(%s)" % os.path.basename(path),
            nt=self.nt, dt_rec=self.dt_rec, block_records=self.block)
        self.conditions.require_execution("BlockRun", nx=self.nx)
        if not manifest:
            raise _cond.PreconditionHalt("BlockRun requires a physical declaration")
        self.my_xt = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.verified = tuple(verified)
        self.manifest = self.conditions.values          # the normalised set
        self.manifest_sha256 = self.conditions.digest
        # manifest_declared is ENFORCED, not merely recorded (audit item 6):
        # a checkpoint whose physics was never declared can never be skipped or
        # resumed, and its artifact is stamped so every downstream gate refuses
        # it.  It is read three times below.
        self.manifest_declared = bool(manifest)
        self.undeclared = self.conditions.undeclared

        if os.path.isfile(path):
            try:
                d = np.load(path, allow_pickle=True)
            except Exception as exc:
                raise _gate.RunIdentityMismatch("unreadable checkpoint: " + str(exc)) from exc
            body, why = _cond.read(d)
            if body is None:
                raise _gate.RunIdentityMismatch(
                    "%s %s; refusing to skip or resume it. Delete it to "
                    "recompute, or add the stamp by hand if you can establish "
                    "what produced it." % (os.path.basename(path), why))
            extra_stamp = body.get("extra")
            extra_stamp = extra_stamp if isinstance(extra_stamp, dict) else {}
            stored_declared = bool(extra_stamp.get("manifest_declared", False))
            if not (self.manifest_declared and stored_declared):
                raise _gate.RunIdentityMismatch(
                    "%s cannot be reused: manifest_declared is %s for this "
                    "call and %s for the stored record, and a checkpoint whose "
                    "physical conditions were never declared cannot be shown "
                    "to belong to any run. %d condition(s) are UNDECLARED here "
                    "(%s). Pass the run's physical declaration."
                    % (os.path.basename(path), self.manifest_declared,
                       stored_declared, len(self.undeclared),
                       ", ".join(self.undeclared[:6]) or "none"))
            problems = _cond.compare(body, self.conditions, _cond.ALL_FIELDS)
            try:
                stored_declaration = json.loads(str(d["run_manifest"]))
                if _gate.canonical_json(stored_declaration) != _gate.canonical_json(self.declaration):
                    problems.append("run_manifest: additional declared run settings differ")
            except (KeyError, ValueError, TypeError):
                problems.append("run_manifest: missing or unreadable full declaration")
            if problems:
                raise _gate.RunIdentityMismatch(
                    "%s was recorded for a DIFFERENT run: %d condition(s) "
                    "differ (stored -> requested)\n    %s"
                    % (os.path.basename(path), len(problems),
                       "\n    ".join(problems)))
            required_payload = ("done", "my_xt", "next_index", "mag_state", "t_state",
                                "initial_state_sha256")
            missing = [key for key in required_payload if key not in d]
            if missing:
                raise _gate.RunIdentityMismatch("incomplete checkpoint payload: " + ", ".join(missing))
            index, finished = int(d["next_index"]), bool(d["done"])
            expected_t = (self.nt - 1 if finished else index) * self.dt_rec
            state = np.asarray(d["mag_state"])
            if (index < 1 or index > self.nt or finished != (index == self.nt)
                    or d["my_xt"].shape != (index, self.nx)
                    or state.shape != (3, NZ, NY, self.nx)
                    or not np.isfinite(state).all() or not np.isfinite(d["my_xt"]).all()
                    or not np.isfinite(float(d["t_state"]))
                    or not np.isclose(float(d["t_state"]), expected_t, rtol=1e-8, atol=1e-18)):
                raise _gate.RunIdentityMismatch("checkpoint state, trace, completion, or solver time is inconsistent")
            if not finished and float(self.conditions.value("material.temperature")) > 0:
                raise _gate.RunIdentityMismatch("thermal mid-run resume requires an engine RNG-state checkpoint; not implemented")
            self.initial_state_sha256 = str(d["initial_state_sha256"])
            self.done = bool(d["done"]) if "done" in d else False
            if "my_xt" in d:
                got = d["my_xt"]
                self.my_xt[:got.shape[0]] = got
            self.i0 = int(d["next_index"]) if "next_index" in d else 0
            self.mag_state = d["mag_state"] if "mag_state" in d else None
            self.t_state = float(d["t_state"]) if "t_state" in d else 0.0
            if "provenance_chain" in d:
                self.prov_chain = json.loads(str(d["provenance_chain"]))
            elif "provenance" in d:
                self.prov_chain = [dict(
                    i_start=0, i_end=self.i0, utc="NOT DETERMINABLE",
                    note="reconstructed from a single legacy provenance record",
                    provenance=json.loads(str(d["provenance"])))]
            self._seg_start = self.i0
            print("  [resume] %s at record %d/%d (t=%.3f ns)"
                  % (os.path.basename(path), self.i0, self.nt,
                     self.t_state * 1e9), flush=True)

    # ------------------------------------------------------------------------
    def _chain_entry(self, extra):
        """This process's segment record.  Never touches earlier entries."""
        prov = extra.get("provenance")
        try:
            prov = json.loads(prov) if isinstance(prov, str) else prov
        except Exception:                                          # noqa: BLE001
            pass
        return dict(i_start=int(self._seg_start), i_end=int(self.i0),
                    utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    host=platform.node(), provenance=prov)

    def save(self, extra, done=False):
        entry = self._chain_entry(extra)
        if self._own_segment is None:
            self.prov_chain.append(entry)
            self._own_segment = len(self.prov_chain) - 1
        else:
            self.prov_chain[self._own_segment] = entry
        tmp = self.path + ".tmp.npz"
        stamp = self.conditions.stamp(
            verified=self.verified,
            extra=dict(manifest_declared=self.manifest_declared,
                       declaration_keys=sorted(self.declaration),
                       done=bool(done), next_index=int(self.i0),
                       nt_total=int(self.nt)))
        np.savez(tmp, my_xt=self.my_xt[:self.i0], next_index=self.i0,
                 mag_state=(self.mag_state if self.mag_state is not None
                            else np.zeros(0)),
                 t_state=self.t_state, done=bool(done), nt_total=self.nt,
                 initial_state_sha256=self.initial_state_sha256,
                 # the caller's raw declaration, kept because G9's control
                 # matching compares declarations; the CONDITION SET above is
                 # what the identity check uses, and it is derived from this
                 run_manifest=_gate.canonical_json(self.declaration),
                 run_manifest_sha256=self.manifest_sha256,
                 provenance_chain=json.dumps(self.prov_chain, default=str),
                 **dict(extra, **stamp))
        os.replace(tmp, self.path)

    def integrate(self, world, magnet, extra, progress_every=250):
        if self.done:
            print("  [skip] %s (done, run manifest %s)"
                  % (os.path.basename(self.path), self.manifest_sha256[:12]),
                  flush=True)
            return self.my_xt
        initial = np.ascontiguousarray(magnet.magnetization.eval(), dtype=np.float64)
        digest = hashlib.sha256(initial.tobytes()).hexdigest()
        if self.initial_state_sha256 is not None and self.initial_state_sha256 != digest:
            raise _gate.RunIdentityMismatch("initial magnetization differs from checkpoint construction")
        self.initial_state_sha256 = digest
        if self.mag_state is not None and self.mag_state.size:
            magnet.magnetization = self.mag_state
            world.timesolver.time = self.t_state
        self._seg_start = self.i0
        t0 = time.time()
        for i in range(self.i0, self.nt):
            self.my_xt[i] = sample_my(magnet)
            self.i0 = i + 1
            if i < self.nt - 1:
                world.timesolver.run(self.dt_rec)
            if (self.i0 % self.block == 0) or self.i0 == self.nt:
                self.mag_state = np.asarray(magnet.magnetization.eval(),
                                            dtype=np.float64)
                try:
                    self.t_state = float(world.timesolver.time)
                except Exception as exc:                       # noqa: BLE001
                    raise _cond.PreconditionHalt("cannot checkpoint an unknown solver time") from exc
                self.save(extra, done=(self.i0 == self.nt))
            if self.i0 % progress_every == 0:
                print("      %s  %d/%d  t=%.1f ns  |m_y|max=%.3e  (%.1f min)"
                      % (os.path.basename(self.path), self.i0, self.nt,
                         self.i0 * self.dt_rec * 1e9,
                         np.abs(self.my_xt[:self.i0]).max(),
                         (time.time() - t0) / 60), flush=True)
        self.done = True
        return self.my_xt

# ------------------------------------------------------------- planning -----
# Gamma[1/s] as a function of eps_0.  The ONLY measured anchors available:
#   literature YIG, B0 = 50 mT, f_SAW = 6.10 GHz, band-resolved Gamma from
#   data/sim48_checkpoints, reported in AUDIT_2026-09-17_referee1_verification.md:
#       eps_0 = 1e-4 / 2e-4 / 3e-4  ->  +13.6 / +45.9 / +79.3 per us
#   controls: -8.8 per us (SAW off), -14 per us (detuned).
#   A straight line through them reproduces the middle point to 0.8 %.
# For the sim40 parameter set the single anchor is Gamma = 3.02e8 1/s at
# eps_0 = 7e-5 (CONVENTIONS.md sec.7), scaled linearly in |B1| eps_0.
GAMMA_LIT_SLOPE, GAMMA_LIT_INTERCEPT = 3.2850e11, -1.9433e7     # 1/s per eps, 1/s
GAMMA_SIM40_AT_7E5 = 3.02e8                                     # 1/s


def gamma_estimate(mat, eps0):
    """INFERRED, not measured, except at the anchor points above."""
    if mat["tag"] == "sim40set":
        return GAMMA_SIM40_AT_7E5 * (eps0 / 7e-5)
    return GAMMA_LIT_SLOPE * eps0 + GAMMA_LIT_INTERCEPT


# --------------------------------------------------------- the run plan -----
# A plan's STATUS has to participate in the decision it is written for.  Round 1
# set `status = "INSUFFICIENT_LINEAR_WINDOW"`, documented it as blocking a
# growth-rate verdict, and every caller then read `plan["t_run"]` and nothing
# else -- so the block blocked nothing, and at a seed past the saturation level
# the planner handed back t_run = -2.2952e-09 s, from which the callers computed
# nt = int(t_run/dt_rec) + 1 = -113.
#
# The class fix is not three `if` statements in three scripts, because a fourth
# script would reopen it.  A plan is returned as a MAPPING THAT REFUSES: reading
# the run length of a blocked plan raises, so a caller that ignores the status
# cannot obtain a number to run with, whether or not it was written before this
# change.  `nt_records()` is the one conversion to a record count and refuses the
# same way, so a nonpositive nt cannot be computed either.
PLAN_OK = ("ok", "floor_clipped_to_linear_window")
PLAN_BLOCKING = ("INSUFFICIENT_LINEAR_WINDOW", "NO_LINEAR_WINDOW")
#: keys whose value only means anything for a plan that can actually be run
PLAN_GUARDED_KEYS = ("t_run", "a_end", "nt")


class PlanRefused(RuntimeError):
    """A run length was asked of a plan whose own status blocks the run."""


class RunPlan(_collections_abc.Mapping):
    """`plan_runtime`'s return value: a read-only mapping that refuses.

    Everything diagnostic (`status`, `note`, `t_linear`, `t_required`, `t_cap`,
    `gamma`) is always readable, so a caller can explain itself.  The keys in
    PLAN_GUARDED_KEYS are readable only while `status` is not blocking; ask for
    them on a blocked plan and you get PlanRefused naming the status and the
    reason, never a number.  `raw` gives the unguarded dict for reporting.
    """

    def __init__(self, fields):
        self._d = dict(fields)

    @property
    def raw(self):
        return dict(self._d)

    @property
    def status(self):
        return self._d.get("status")

    @property
    def blocked(self):
        return str(self._d.get("status")) in PLAN_BLOCKING

    def __getitem__(self, key):
        if key in PLAN_GUARDED_KEYS and self.blocked:
            raise PlanRefused(
                "this run plan is BLOCKED (status = %s) and has no usable %r. "
                "%s  Fix the plan -- lower the seed amplitude, shorten the "
                "frequency window or accept a held growth-rate verdict -- "
                "instead of reading a run length out of it."
                % (self._d.get("status"), key, self._d.get("note", "")))
        return self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return "RunPlan(status=%r, t_run=%r, t_linear=%r)" % (
            self._d.get("status"), self._d.get("t_run"),
            self._d.get("t_linear"))


def require_plan(plan, where=""):
    """Raise unless this plan may be run.  Returns the plan, so it can wrap."""
    st = str(plan["status"]) if "status" in plan else "NO_STATUS"
    if st in PLAN_BLOCKING or st not in PLAN_OK:
        raise PlanRefused("run plan refused at %s: status = %s. %s"
                          % (where or "an unnamed call site", st,
                             plan.get("note", "")))
    return plan


def nt_records(plan, dt_rec, where=""):
    """THE conversion from a plan to a record count.  Refuses a plan that is
    blocked, and refuses a count that is not a measurement (nt < 2), so the
    nt = -113 of round 1 cannot be produced from any plan by any caller."""
    require_plan(plan, where)
    t_run = float(plan["t_run"])
    if not np.isfinite(t_run) or t_run <= 0:
        raise PlanRefused("run plan gives t_run = %r at %s: not a run length"
                          % (t_run, where or "an unnamed call site"))
    nt = int(t_run / float(dt_rec)) + 1
    if nt < 2:
        raise PlanRefused("run plan gives nt = %d records at %s (t_run = %.4e "
                          "s, dt_rec = %.4e s): a single record measures nothing"
                          % (nt, where or "an unnamed call site", t_run, dt_rec))
    return nt


def plan_runtime(mat, eps0, a_seed, m_sat_level=0.1, t_window=100e-9,
                 t_min=40e-9, t_max=800e-9):
    """Run length set by the physics time scale, with the linear window as a
    HARD ceiling.

    Two competing requirements:
      growth fit : |Gamma| * T >~ 1.5  (a factor 4.5 in amplitude)
      frequency  : >= t_window of LINEAR signal, so df_bin <= 1/t_window
    and one ceiling: the run must stay linear, i.e. a_seed exp(Gamma T) <
    m_sat_level.

    FIXED 2026-09-18 (independent audit, sec.8, P1-2).  The previous form was

        t_run = min(t_max, max(t_min, min(need, t_lin)))

    in which the convenience floor `t_min` sat OUTSIDE the linear ceiling and
    therefore overrode it.  Measured consequence at the sim40 parameter set,
    eps_0 = 7e-5: t_linear = 20.805 ns, t_run = 40.000 ns, and the planner's own
    exponential ends at a = 32.92 -- a run planned to be linear that cannot be.
    The floor is now applied only up to the ceiling, and a floor that would
    exceed the ceiling is reported in `status` instead of being obeyed.

    For a predicted-DECAYING point (Gamma < 0, i.e. below threshold) there is no
    saturation ceiling, but the decay still has to be measurable: the
    requirement is 1.5/|Gamma|, not the old `need = t_max`, which padded every
    sub-threshold point to the 800 ns cap.

    Returns, besides `t_run`:
      t_required : what the growth fit and the frequency window need
      t_linear   : when a_seed exp(Gamma t) reaches m_sat_level (inf if Gamma<=0)
      t_cap      : min(t_max, t_linear), the hard ceiling actually applied
      a_end      : the planner's own end amplitude at t_run -- must be
                   <= m_sat_level by construction
      status     : "ok" | "floor_clipped_to_linear_window" (both runnable,
                 PLAN_OK) | "INSUFFICIENT_LINEAR_WINDOW" | "NO_LINEAR_WINDOW"
                 (both PLAN_BLOCKING).  INSUFFICIENT_LINEAR_WINDOW means the
                 analysis window the fit needs does not fit inside the linear
                 window at this seed amplitude; NO_LINEAR_WINDOW means the seed
                 is already at or past the saturation level, so there is no
                 window at all.

    The return value is a `RunPlan`, not a plain dict: on a BLOCKING status,
    reading `t_run`, `a_end` or `nt` raises `PlanRefused` instead of handing back
    a number.  That is what makes the status participate for every caller,
    including callers written before the status existed.  Use
    `nt_records(plan, dt_rec)` for the record count.
    """
    g = gamma_estimate(mat, eps0)
    if float(a_seed) >= float(m_sat_level):
        # There is no linear window at all: the seed is already at or past the
        # level the plan itself calls saturated.  Round 1 carried on and returned
        # t_linear = t_run = -2.2952e-09 s.  A plan with no window is not a
        # shorter plan, it is NOT A PLAN, and it is returned blocked.
        return RunPlan(dict(
            gamma=g,
            t_linear=(float(np.log(m_sat_level / a_seed) / g) if g != 0
                      else float("-inf")),
            t_required=float("nan"), t_cap=float("nan"),
            t_run=float("nan"), a_end=float("nan"),
            m_sat_level=float(m_sat_level), status="NO_LINEAR_WINDOW",
            note=("the seed amplitude a_seed = %.4e is already >= the "
                  "saturation level m_sat_level = %.4e, so no run length keeps "
                  "this run linear and no growth rate can be measured from it"
                  % (a_seed, m_sat_level)), saturates=True))
    if g > 0:
        t_lin = float(np.log(m_sat_level / a_seed) / g)
        t_req = float(max(1.5 / g, t_window))
    else:
        t_lin = np.inf                      # a decaying mode never saturates
        t_req = float(max(1.5 / abs(g), t_window)) if g != 0 else float(t_window)
    t_cap = float(min(t_max, t_lin))
    t_run = float(min(max(t_req, min(t_min, t_cap)), t_cap))
    notes, status = [], "ok"
    if t_min > t_cap * (1 + 1e-9):
        status = "floor_clipped_to_linear_window"
        notes.append("t_min floor %.3f ns exceeds the ceiling %.3f ns "
                     "(t_linear %.3f ns, t_max %.3f ns) and was NOT applied"
                     % (t_min * 1e9, t_cap * 1e9, t_lin * 1e9, t_max * 1e9))
    if t_req > t_cap * (1 + 1e-9):
        status = "INSUFFICIENT_LINEAR_WINDOW"
        notes.append("the fit/frequency requirement %.3f ns does not fit inside "
                     "the ceiling %.3f ns: lower the seed or accept a held "
                     "growth-rate verdict" % (t_req * 1e9, t_cap * 1e9))
    a_end = float(a_seed * np.exp(g * t_run)) if np.isfinite(g) else float("nan")
    if not np.isfinite(t_run) or t_run <= 0:
        status = "NO_LINEAR_WINDOW"
        notes.append("the planner arrived at t_run = %r, which is not a run "
                     "length" % (t_run,))
    return RunPlan(dict(gamma=g, t_linear=t_lin, t_required=t_req, t_cap=t_cap,
                        t_run=t_run, a_end=a_end,
                        m_sat_level=float(m_sat_level),
                        status=status, note="; ".join(notes),
                        saturates=bool(np.isfinite(t_lin)
                                       and t_lin < t_window)))


def seed_temperature(a_mode, k, B0, Ms, Aex, V_tot):
    """Inverse of thermal_mode_amplitude: the magnon temperature a seed of
    per-mode amplitude a_mode corresponds to.  Report it in every output so a
    seed can never again be quoted as 'small' without saying small compared
    with what."""
    a1 = thermal_mode_amplitude(k, 1.0, B0, Ms, Aex, V_tot)
    return float((a_mode / a1) ** 2)


def choose_seed_amplitude(mat, eps0, a_phys, t_window=100e-9,
                          m_sat_level=0.1):
    """Largest seed that still leaves t_window of LINEAR growth, capped at the
    physical (thermal) amplitude.

    Rationale.  Seeding too LOW is the CLAUDE.md sec.4 trap (it forces long
    runs, which then tempts coupling inflation).  Seeding too HIGH saturates
    the run before the frequency window is over.  a_phys is the ceiling because
    nothing in the physics is quieter than the bath.
    """
    g = gamma_estimate(mat, eps0)
    if g <= 0:
        return float(a_phys), "gamma<=0, no saturation constraint"
    a_lin = m_sat_level * np.exp(-g * t_window)
    if a_lin >= a_phys:
        return float(a_phys), "thermal ceiling (a_phys) binds"
    return float(a_lin), ("linear-window constraint binds: "
                          "a = %.1f%% of a_phys" % (100 * a_lin / a_phys))


def auto_window(gamma, t_cap=100e-9):
    """Frequency window matched to the line, not to wishful thinking.

    A mode growing at Gamma has a spectral half width Gamma/(2 pi); resolving
    finer than that buys nothing, and demanding it forces an absurdly small
    seed (at the sim40 parameter set, Gamma = 3e8 1/s, a 100 ns LINEAR window
    needs a ~ 1e-14 -- floating-point-noise seeding, the exact CLAUDE.md sec.4
    trap, arrived at from the other end).  So: window = min(t_cap, 2 pi/Gamma).
    """
    if gamma <= 0:
        return t_cap
    return float(min(t_cap, 2 * np.pi / gamma))


# ======================================================================
# uniform failure behaviour for every entry point
# ======================================================================
def run_entry(name, fn, *args, **kwargs):
    """Run one stage from __main__ and return the process exit code.

    Every script in runs/ goes through this, so a halt behaves the same way
    everywhere.  Round-1 state, which this replaces: `G4_commensurate_onset.py
    pump` printed CAMPAIGN BLOCKED and exited 0 because the gate was required
    only on the 'all' path, while G6 and G4b had no handler at all and surfaced
    a halt as a traceback.

        0  the stage ran and its gate (if it returns one) PASSED
        2  a gate blocked, a prerequisite was refused, a run plan was refused,
           or a checkpoint did not belong to this run -- a named halt
        3  an unexpected exception, which is a bug and not a verdict

    A returned Gate is REQUIRED here even for a single stage: the point of a
    per-stage invocation is to produce a usable certificate, and a FAIL or
    NOT_DETERMINABLE certificate is not one.
    """
    try:
        out = fn(*args, **kwargs)
    except _gate.GateHalt as halt:                  # incl. PreconditionHalt
        print("")
        print("  HALTED (%s): %s" % (name, halt), flush=True)
        return 2
    except _gate.RunIdentityMismatch as bad:
        print("")
        print("  HALTED (%s): a stored record does not belong to this run: %s"
              % (name, bad), flush=True)
        return 2
    except PlanRefused as bad:
        print("")
        print("  HALTED (%s): %s" % (name, bad), flush=True)
        return 2
    if isinstance(out, _gate.Gate) and out.blocks:
        print("")
        print("  HALTED (%s): %s %s -- %s"
              % (name, out.name, out.state, out.reason), flush=True)
        return 2
    state = None
    if isinstance(out, dict):
        state = str(out.get("verdict_state", out.get("gate_state", "")))
    if state in _gate.BLOCKING:
        print("")
        print("  HALTED (%s): %s" % (name, state), flush=True)
        return 2
    return 0


def prov_kw(script_path, params):
    """provenance() packed as ONE npz keyword, so it can never collide with a
    data key.  Read it back with json.loads(str(d['provenance']))."""
    return {"provenance": json.dumps(provenance(script_path, params),
                                     default=str)}


# ======================================================================
# THE condition set: one builder, one environment, one place
# ======================================================================
def condition_env():
    """The values the ENGINE takes from this module, read live.

    Read live, not captured at import, so that changing a harness constant --
    which changes the integrated field -- moves every condition set, digest,
    certificate and checkpoint identity built afterwards.  `boxes` and
    `materials` let a declaration name a registered box or material by tag
    instead of repeating it.
    """
    env = dict(NY=NY, NZ=NZ, CY=CY, CZ=CZ, PBC=list(PBC), XI=XI,
               SAW_PHASE=SAW_PHASE, SAW_DIRECTION=SAW_DIRECTION,
               ENABLE_MEL=ENABLE_MEL, ENABLE_BARNETT=ENABLE_BARNETT,
               DT_STEP=DT_STEP, SEED_KIND=SEED_KIND, V_SAW=V_SAW,
               boxes=(BOX_PRIMARY, BOX_CROSS, BOX_CELLCONV),
               materials=(YIG_LIT, SIM40))
    env.update(_prov.build_conditions())
    return env


def conditions(decl, source="", note="", **over):
    """The condition set of a run, from the caller's declaration + this module.

    `decl` is the physical declaration a stage already builds for provenance
    (box, material, B0, eps0, f_saw, temperature, rng_seed, a_seed, seed meta);
    `over` carries the numerics the harness owns (nt, dt_rec, block_records).
    Every field in _conditions.FIELDS is filled or recorded as UNDECLARED --
    there is no third possibility, and no field can be omitted by a call site
    because the field list is not written at the call site.
    """
    d = dict(decl or {})
    d.update({k: v for k, v in over.items() if v is not None})
    return _cond.ConditionSet.of(d, condition_env(), source=source, note=note)


def stage_artifact(path, cs, verified=(), criteria=None, **payload):
    """Write a stage artifact (certificate / summary) WITH its condition stamp.

    The only writer.  A stage cannot forget the stamp, because this is how a
    stage writes a file; and `verified` records which conditions the artifact
    actually checked, which is what a later stage is allowed to rely on.
    """
    kw = dict(payload)
    kw.update(cs.stamp(verified=verified, criteria=criteria))
    np.savez(path, **kw)
    return path
