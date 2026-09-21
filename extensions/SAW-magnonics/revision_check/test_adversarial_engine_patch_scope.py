"""NEW counterexamples against the ENGINE-PATCH SCOPING, from the other side.

The previous round called both `chiralsawfield.cu` defects blocking because a
direction reversal was planned.  This round narrows that to the execution paths
each defect actually reaches (`proposed_engine_patches/SCOPE.md`).  Narrowing a
gate is the dangerous direction, so these counterexamples attack the narrowing
rather than the defects:

  S1  OVER-BLOCKING.  The scope record must let the planned x-axis +/-k
      comparison with Barnett off through.  If it refuses that, it is too wide.
  S2  UNDER-BLOCKING, Barnett.  Turn the Barnett channel on in the harness and
      the scope claim must go RED -- otherwise "outside the scope" is a rubber
      stamp that survives the condition it was predicated on.
  S3  UNDER-BLOCKING, direction.  The y-rotation defect must be REACHABLE at
      saw_direction = 1, assigned after construction (the route that bypasses
      build()'s `direction='x'` literal).  A scope record whose boundary is not
      where the defect starts is worthless in either direction.
  S4  The two defects must be SEPARABLE: patch 01's test must be insensitive to
      the Barnett channel and patch 02's test to the propagation axis.  If one
      test fires on the other's condition, the two scopes cannot be maintained
      apart and the previous round's "both blocking" would have been right.
  S5  The proposals must exist as proposals: annotated patch, validating test and
      scope record present, and the ENGINE TREE UNCHANGED.

Nothing here modifies the engine tree; S5 asserts that by hash.  S2 mutates a
module attribute in memory only -- no file is written.

Run:  python test_adversarial_engine_patch_scope.py
"""

import hashlib
import io
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHES = os.path.join(HERE, "proposed_engine_patches")
RUNS = os.path.join(HERE, "runs")
SRC = os.path.join(os.path.dirname(HERE), "src")
KERNEL = os.path.join("D:/", "mumax-plus-dev", "src", "physics",
                      "chiralsawfield.cu")
KERNEL_SHA = "56a7d57ca4b1acf8b5dffa851921c249fec4999ff3136d3245136bdab7f35a20"

for p in (PATCHES, RUNS, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

OPEN = []
NCHECK = 0


def check(name, ok, detail=""):
    global NCHECK
    NCHECK += 1
    print(("  CLOSED  " if ok else "  OPEN    ") + name
          + (("   " + detail) if detail else ""))
    if not ok:
        OPEN.append(name + (("   " + detail) if detail else ""))


def _run(script, cwd=PATCHES):
    r = subprocess.run([sys.executable, script], capture_output=True, text=True,
                       cwd=cwd, timeout=1800)
    return r.returncode, r.stdout + r.stderr


# ===========================================================================
# S1.  The scope must not be too wide: the +/-k run must be let through.
# ===========================================================================
def test_the_pm_k_run_is_not_blocked():
    print("\n[S1] the planned x-axis +/-k comparison must NOT be gated")
    rc, out = _run("verify_scope_pm_k.py")
    check("the scope verifier passes the +/-k comparison with Barnett off",
          rc == 0 and "OUTSIDE both patch scopes" in out,
          "rc=%d" % rc)
    check("it establishes that from the ENGINE's own readback, not from prose",
          "engine reports saw_direction=0.0" in out
          and "saw_enable_barnett=0.0" in out,
          "the two readback lines are the load-bearing evidence")


# ===========================================================================
# S2.  The scope must not be a rubber stamp: turn Barnett on, it must go red.
# ===========================================================================
def test_turning_barnett_on_must_break_the_scope_claim():
    print("\n[S2] with the Barnett channel ON the scope claim must go RED")
    import _harness as H
    import verify_scope_pm_k as V
    was = H.ENABLE_BARNETT
    try:
        H.ENABLE_BARNETT = True
        res = V.scope_checks(H, verbose=False)
        bad = [n for n, ok, _d in res if not ok]
        check("a Barnett-on harness fails the scope claim",
              bool(bad), "%d of %d checks red: %s"
              % (len(bad), len(res), "; ".join(b[:52] for b in bad)))
        check("the failure names the Barnett condition specifically",
              any("Barnett" in b for b in bad),
              "; ".join(b[:60] for b in bad) or "no red checks")
    finally:
        H.ENABLE_BARNETT = was
        assert H.ENABLE_BARNETT is False, "failed to restore ENABLE_BARNETT"
    res = V.scope_checks(H, verbose=False)
    check("and the restored harness is green again",
          all(ok for _n, ok, _d in res), "%d checks" % len(res))


# ===========================================================================
# S3.  The scope boundary must be where the defect starts.
# ===========================================================================
def test_the_defect_is_reachable_at_saw_direction_1():
    print("\n[S3] the y-rotation defect must be REACHABLE at saw_direction = 1")
    import test_patch01_rotation_covariance as P1
    m0 = np.array([0.6, 0.48, 0.64])
    m0 = m0 / np.linalg.norm(m0)
    mR = P1.rot_z90(m0.reshape(3, 1))[:, 0]

    # the scope record's own residual route: assign the attribute AFTER
    # construction, bypassing build()'s `direction='x'` literal.
    _wa, a = P1.build(0, m0)
    _wb, b = P1.build(0, mR)
    b.saw_direction = 1
    HA = P1.saw_field(a)[:, 0, 0, :]
    HB = P1.saw_field(b)[:, 0, :, 0]
    scale = max(np.max(np.abs(HA)), 1e-300)
    err = np.max(np.abs(HB - P1.rot_z90(HA))) / scale
    check("assigning magnet.saw_direction = 1 after construction reaches the "
          "defect, so the scope record is right to name that route",
          err > 1e-6, "covariance error %.6e" % err)

    # and at saw_direction = 0 the same configuration is covariant, which is
    # why the campaign is outside the scope.
    _wc, c = P1.build(0, m0, kmr=0.0)
    _wd, d = P1.build(1, mR, kmr=0.0)
    mA, mB = P1.saw_field(c)[:, 0, 0, :], P1.saw_field(d)[:, 0, :, 0]
    merr = np.max(np.abs(mB - P1.rot_z90(mA))) / max(np.max(np.abs(mA)), 1e-300)
    check("with K_mr = 0 the same rotation IS covariant, so the defect is the "
          "magneto-rotation axis and not the axis rotation as such",
          merr <= 1e-6, "MEL-only covariance error %.3e" % merr)


# ===========================================================================
# S4.  The two defects must stay separable.
# ===========================================================================
def test_the_two_defects_are_separable():
    print("\n[S4] each validating test must respond to its OWN defect only")
    rc1, out1 = _run("test_patch01_rotation_covariance.py")
    rc2, out2 = _run("test_patch02_barnett_sign_k.py")
    check("patch 01's test runs with the Barnett channel off and still fires",
          rc1 != 0 and "magneto-rotation channel" in out1,
          "rc=%d" % rc1)
    check("patch 02's test runs at saw_direction = 0 and still fires",
          rc2 != 0 and "missing sign(k)" in out2, "rc=%d" % rc2)
    check("patch 01's test carries a control that passes, so its failure is a "
          "diagnosis and not a broken setup",
          "control, MEL only (K_mr = 0): max relative covariance error = 0.000e+00"
          in out1)
    check("patch 02's test carries the magneto-rotation reference that shows "
          "sign(k) IS applied there",
          "sign(k) IS carried there" in out2)


# ===========================================================================
# S5.  Proposals present; engine tree unchanged.
# ===========================================================================
def test_proposals_present_and_engine_untouched():
    print("\n[S5] the patches must exist as PROPOSALS and the engine be untouched")
    for f in ("SCOPE.md", "README.md",
              "patch_01_sawdir_rotation_axis.md",
              "patch_02_barnett_sign_k.md",
              "test_patch01_rotation_covariance.py",
              "test_patch02_barnett_sign_k.py",
              "verify_scope_pm_k.py"):
        check("proposed_engine_patches/%s exists" % f,
              os.path.isfile(os.path.join(PATCHES, f)))
    got = hashlib.sha256(open(KERNEL, "rb").read()).hexdigest()
    check("src/physics/chiralsawfield.cu is unchanged", got == KERNEL_SHA,
          got[:16] + "...")
    with io.open(os.path.join(PATCHES, "SCOPE.md"), encoding="utf-8") as fh:
        scope = fh.read()
    check("SCOPE.md names the scripts and the parameter values, not just the "
          "defects",
          all(s in scope for s in ("G4_commensurate_onset.py",
                                   "G4b_crosscheck_box.py",
                                   "G6_clean_seed.py",
                                   "G9_idler_injection.py",
                                   "ENABLE_BARNETT", "direction='x'",
                                   "K_mr")),
          "%d chars" % len(scope))
    check("SCOPE.md records the K_mr < 0 route to negative k, which a grep for "
          "`direction` alone would miss",
          "K_mr >= 0 else -k" in scope)


def main():
    for fn in (test_the_pm_k_run_is_not_blocked,
               test_turning_barnett_on_must_break_the_scope_claim,
               test_the_defect_is_reachable_at_saw_direction_1,
               test_the_two_defects_are_separable,
               test_proposals_present_and_engine_untouched):
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            check("%s raised" % fn.__name__, False,
                  "%s: %s" % (type(exc).__name__, exc))
    print("\n" + "=" * 78)
    print("%d counterexamples, %d still OPEN" % (NCHECK, len(OPEN)))
    for o in OPEN:
        print("   OPEN  " + o)
    print("=" * 78)
    return 1 if OPEN else 0


if __name__ == "__main__":
    sys.exit(main())
