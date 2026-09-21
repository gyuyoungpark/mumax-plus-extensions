"""ADVERSARIAL counterexamples against the 2026-09-18 record+provenance fixes.

Written by the adversarial verifier, not by the agent that made those fixes.
Every check below is a DIFFERENT counterexample aimed at a defect the fix claims
to have closed, and every one of them is EXPECTED TO FAIL until the
corresponding defect is closed.  A green run of this file means the open items
have been fixed.

It is deliberately separate from test_record_corrections.py so that suite's
142/0 is not disturbed: that suite's own red-then-green was independently
re-verified by reverting each fix in place, and its passing state is real.  What
this file records is what that suite does NOT catch.

Nothing here modifies the engine tree.  The checks that need a mutable source
tree copy D:/mumax-plus-dev/src/physics into a temporary directory and redirect
_provenance.PHYSICS_DIR at the copy; the real tree is only ever read.  The three
checks that mutate a document in this directory do so in BYTES, restore the
exact original bytes in a finally block, and assert the restore.

Run:  python test_adversarial_record_provenance.py
Exits non-zero while any counterexample still reproduces.
"""

import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
if RUNS not in sys.path:
    sys.path.insert(0, RUNS)

PHYSICS_REAL = os.path.join("D:/", "mumax-plus-dev", "src", "physics")

OPEN = []
NCHECK = 0

LF = b"\n"
CRLF = b"\r\n"


def check(name, ok, detail=""):
    """ok=True means the defect is CLOSED.  ok=False means it still reproduces."""
    global NCHECK
    NCHECK += 1
    print(("  CLOSED  " if ok else "  OPEN    ") + name
          + (("   " + detail) if detail else ""))
    if not ok:
        OPEN.append(name + (("   " + detail) if detail else ""))


def doc(name):
    p = os.path.join(HERE, name)
    if not os.path.isfile(p):
        return ""
    with io.open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def _write_bytes(path, blob):
    with open(path, "wb") as fh:
        fh.write(blob)


def _eol(blob):
    return CRLF if CRLF in blob else LF


def _lines(blob, *texts):
    """Join `texts` as lines using the file's own terminator."""
    eol = _eol(blob)
    return eol.join(t.encode("utf-8") for t in texts) + eol


def _suite():
    """Run test_record_corrections.py; return (nchecks, nfailed, stdout)."""
    r = subprocess.run([sys.executable, "test_record_corrections.py"],
                       capture_output=True, text=True, cwd=HERE, timeout=2400)
    m = re.search(r"^(\d+) checks, (\d+) FAILED", r.stdout, re.M)
    if not m:
        return (None, None, r.stdout + r.stderr)
    return (int(m.group(1)), int(m.group(2)), r.stdout)


def _physics_copy():
    """A writable copy of the engine physics sources."""
    td = tempfile.mkdtemp(prefix="adv_physics_")
    for pat in ("*.cu", "*.hpp", "*.cpp", "CMakeLists.txt"):
        for f in glob.glob(os.path.join(PHYSICS_REAL, pat)):
            shutil.copy2(f, td)
    return td


# ===========================================================================
# A1.  The retracted verdict can be REINSTATED with the suite fully green.
#      Claim under test: "single_wave_sufficient cannot be reinstated without
#      failing a test."  Every verdict check is a POSITIVE substring grep, so an
#      ADDED superseding verdict is invisible.  This is task item 4.
# ===========================================================================
def test_reinstating_the_retracted_verdict_must_fail_a_test():
    print("\n[A1] reinstating `single_wave_sufficient` must break the suite")
    p = os.path.join(HERE, "model_comparison.md")
    original = _read_bytes(p)
    anchor = b"## 1. Tolerances"
    i = original.find(anchor)
    assert i > 0, "anchor moved; update this test"
    inj = _lines(
        original,
        "",
        "## VERDICT (SUPERSEDING)",
        "",
        "> On review the operative verdict of this comparison is",
        "> **`single_wave_sufficient`**: one off-grid spatial structure",
        "> suffices, and the earlier `indistinguishable /",
        "> both_models_inadequate` label is withdrawn as an artefact of the",
        "> plane-wave parameterisation.  The retraction above no longer",
        "> applies.",
        "",
    )
    try:
        _write_bytes(p, original[:i] + inj + original[i:])
        n, nf, _out = _suite()
        check("a superseding `single_wave_sufficient` verdict is rejected",
              bool(nf), "suite returned %s checks, %s FAILED" % (n, nf))
    finally:
        _write_bytes(p, original)
        assert _read_bytes(p) == original, "failed to restore %s" % p

    t = doc("test_record_corrections.py")
    neg = [ln for ln in t.splitlines()
           if "single_wave" in ln
           and re.search(r"not in|is None|must not|forbid", ln)]
    check("some check forbids `single_wave_sufficient` as an operative verdict",
          bool(neg),
          "the only guard is a positive grep for the word next to RETRACTED")


# ===========================================================================
# A2.  The verdict checks are whole-file greps, so the pre-registered VERDICT
#      block itself can be deleted with the suite still 140/142 green.  The one
#      content check that fires is the single unique string "306.67".
# ===========================================================================
def test_deleting_the_verdict_block_must_be_caught():
    print("\n[A2] deleting the pre-registered VERDICT block must be caught")
    p = os.path.join(HERE, "model_comparison.md")
    original = _read_bytes(p)
    i, j = original.find(b"## VERDICT"), original.find(b"## 1. Tolerances")
    assert 0 <= i < j, "anchors moved; update this test"
    try:
        _write_bytes(p, original[:i] + original[j:])
        _n, _nf, out = _suite()
        # the suite prints each failure twice (inline, then in its summary)
        fails = sorted({ln.strip() for ln in (out or "").splitlines()
                        if ln.strip().startswith("FAIL")
                        and "file:line reference" not in ln})
        check("it fails the `both_models_inadequate` sub-label check",
              any("both_models_inadequate" in ln for ln in fails),
              "%d distinct content failure(s): %s"
              % (len(fails), "; ".join(f[:56] for f in fails)))
        check("it fails the four HELD-OUT value checks",
              any("18.081970" in ln for ln in fails),
              "the values are duplicated in the results table, so the "
              "whole-file grep still passes")
    finally:
        _write_bytes(p, original)
        assert _read_bytes(p) == original, "failed to restore %s" % p


# ===========================================================================
# A3.  build_manifest() pairs the SELECTED CMake cache with a binary found by a
#      fixed glob: _build_tree_binary() discards its cache_path argument.  There
#      are two SINGLE-precision build trees in this repo and build_test_single/
#      emitted no lib.* output at all, yet when it is the newest cache the
#      manifest still returns LINKED_VIA_BUILD_TREE with the reason "byte-
#      identical to the artefact THIS BUILD TREE emitted".
# ===========================================================================
def test_build_tree_binary_must_come_from_the_selected_tree():
    print("\n[A3] the binary must be shown to come from the SELECTED build tree")
    import _provenance as P
    real_cache = os.path.join(
        "D:/", "mumax-plus-dev", "build", "temp.win-amd64-cpython-314",
        "Release", "single", "CMakeCache.txt")
    a = P._build_tree_binary(real_cache)
    b = P._build_tree_binary("D:/THIS/TREE/DOES/NOT/EXIST/CMakeCache.txt")
    check("_build_tree_binary() uses its cache_path argument", a != b,
          "a nonexistent build tree returns the same .pyd: %s" % (a,))

    trees = P._find_build_trees()
    singles = [c for c in trees
               if P._parse_cmake_cache(c).get("FP_PRECISION", "").strip().upper()
               == "SINGLE"]
    check("only one SINGLE-precision build tree is selectable",
          len(singles) <= 1,
          "%d found: %s" % (len(singles),
                            [os.path.basename(os.path.dirname(c))
                             for c in singles]))

    other = next((c for c in singles if "build_test_single" in c), None)
    if not other:
        return
    orig = P._find_build_trees
    try:
        P._find_build_trees = lambda: [other] + [c for c in orig() if c != other]
        P._LINK_CACHE.clear()
        m = P.build_manifest()
        cache_dir = os.path.basename(os.path.dirname(
            m["build_config"]["cmake_cache_path"]))
        bin_path = str(m["binary"]["build_tree_path"])
        bin_dir = os.path.basename(os.path.dirname(bin_path))
        mixed = ("build_test_single" in m["build_config"]["cmake_cache_path"]
                 and "build/lib." in bin_path)
        check("a cache/binary pair from DIFFERENT trees is not called LINKED",
              not (mixed and m["link_status"] == P.LINK_LINKED),
              "cache=%s binary=%s -> %s, blocks_downstream=%s"
              % (cache_dir, bin_dir, m["link_status"], m["blocks_downstream"]))
    finally:
        P._find_build_trees = orig
        P._LINK_CACHE.clear()


# ===========================================================================
# A4/A5.  source_files() covers the chiral SAW drive but NOT the other kernels
#      the sim40 / G4 / G6 runs integrate.  magnetoelasticfield.cu carries the
#      B_1 = -8.8e6 drive that sim40 runs with enable_mel=True.
# ===========================================================================
UNCOVERED = ("magnetoelasticfield.cu", "exchange.cu", "anisotropy.cu",
             "zeeman.cu", "demag.cpp", "thermalnoise.cu", "minimizer.cu")


def test_source_list_must_cover_the_integrated_path():
    print("\n[A4] the source list must cover the kernels the runs integrate")
    import _provenance as P
    covered = " ".join(P.build_manifest()["sources"])
    for name in UNCOVERED:
        check("manifest covers %s" % name, name in covered,
              "in the sim40/G4/G6 integrated path, absent from source_files()")


def test_editing_an_uncovered_kernel_must_move_the_identity():
    print("\n[A5] editing an uncovered kernel must invalidate a checkpoint")
    import _provenance as P
    td = _physics_copy()
    real_phys, real_kern = P.PHYSICS_DIR, P.KERNEL_CU
    try:
        P.PHYSICS_DIR = td
        P.KERNEL_CU = os.path.join(td, "chiralsawfield.cu")
        P._LINK_CACHE.clear()
        before = P.run_manifest_fields()
        m0 = P.build_manifest()

        tgt = os.path.join(td, "magnetoelasticfield.cu")
        with io.open(tgt, "a", encoding="utf-8") as fh:
            fh.write("\n// adversarial edit: changes the MEL torque\n")
        os.utime(tgt, None)
        P._LINK_CACHE.clear()
        after = P.run_manifest_fields()
        m1 = P.build_manifest()

        check("a MEL-kernel edit moves source_tree_digest",
              m0["source_tree_digest"] != m1["source_tree_digest"],
              "digest unchanged: %s" % m1["source_tree_digest"][:16])
        check("a MEL-kernel edit is reported as a stale source",
              bool(m1["link"]["sources_changed_since_build"]),
              "sources_changed_since_build = %s"
              % m1["link"]["sources_changed_since_build"])
        check("a MEL-kernel edit stops the link being called LINKED",
              m1["link_status"] != P.LINK_LINKED,
              "%s, blocks_downstream=%s"
              % (m1["link_status"], m1["blocks_downstream"]))
        check("a MEL-kernel edit moves the run identity", before != after,
              "run_manifest_fields() byte-identical")

        import _harness as H
        wd = tempfile.mkdtemp(prefix="adv_ckpt_")
        try:
            p = os.path.join(wd, "block.npz")
            params = dict(Ms=140e3, B_1=-8.8e6, eps0=7e-5, enable_mel=True)
            P._LINK_CACHE.clear()
            br = H.BlockRun(p, nt=4, nx=8, dt_rec=1.0, manifest=params)
            br.i0, br.t_state = 4, 4.0
            br.save({}, done=True)
            with io.open(tgt, "a", encoding="utf-8") as fh:
                fh.write("\n// second adversarial edit\n")
            os.utime(tgt, None)
            P._LINK_CACHE.clear()
            refused = False
            try:
                H.BlockRun(p, nt=4, nx=8, dt_rec=1.0, manifest=params)
            except Exception:                                  # noqa: BLE001
                refused = True
            check("the resume gate refuses a checkpoint after a MEL-kernel edit",
                  refused, "stored done=True was accepted")
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    finally:
        P.PHYSICS_DIR, P.KERNEL_CU = real_phys, real_kern
        P._LINK_CACHE.clear()
        shutil.rmtree(td, ignore_errors=True)


# ===========================================================================
# A6.  With no engine binary resolvable -- the premise this bundle is supposed
#      to handle -- engine_binary_sha256 and build_link_status collapse to
#      constant strings, so the incomplete digest is the ONLY source-side guard.
# ===========================================================================
def test_identity_must_not_collapse_without_an_engine_binary():
    print("\n[A6] with no engine binary the identity must still track sources")
    import _provenance as P
    td = _physics_copy()
    real_phys, real_kern = P.PHYSICS_DIR, P.KERNEL_CU
    real_bin = P._installed_binary
    try:
        P.PHYSICS_DIR = td
        P.KERNEL_CU = os.path.join(td, "chiralsawfield.cu")
        P._installed_binary = lambda precision="single": None
        P._LINK_CACHE.clear()
        before = P.run_manifest_fields()
        for name in ("exchange.cu", "anisotropy.cu", "thermalnoise.cu",
                     "demag.cpp", "minimizer.cu"):
            t = os.path.join(td, name)
            if os.path.isfile(t):
                with io.open(t, "a", encoding="utf-8") as fh:
                    fh.write("\n// adversarial edit\n")
                os.utime(t, None)
        P._LINK_CACHE.clear()
        after = P.run_manifest_fields()
        check("editing five in-path kernels moves the identity when the engine "
              "binary is not resolvable", before != after,
              "identity byte-identical; fields = %s" % sorted(before))
    finally:
        P.PHYSICS_DIR, P.KERNEL_CU = real_phys, real_kern
        P._installed_binary = real_bin
        P._LINK_CACHE.clear()
        shutil.rmtree(td, ignore_errors=True)


# ===========================================================================
# A7.  _installed_binary() picks the newest .pyd on sys.path rather than the one
#      import resolution would load.
# ===========================================================================
def test_installed_binary_must_follow_import_resolution():
    print("\n[A7] the hashed .pyd must be the one python would import")
    import _provenance as P
    real = P._installed_binary()
    if real is None:
        check("an installed binary was found to compare against", False,
              "no .pyd on sys.path")
        return
    td = tempfile.mkdtemp(prefix="adv_pyd_")
    try:
        with open(os.path.join(td, os.path.basename(real)), "wb") as fh:
            fh.write(b"NOT A REAL BINARY")
        sys.path.append(td)        # APPENDED: import still resolves to `real`
        pick = P._installed_binary()
        check("_installed_binary() follows sys.path order, not mtime",
              os.path.normcase(os.path.abspath(pick))
              == os.path.normcase(os.path.abspath(real)),
              "picked %s instead of the resolvable %s" % (pick, real))
    finally:
        if td in sys.path:
            sys.path.remove(td)
        shutil.rmtree(td, ignore_errors=True)
        P._LINK_CACHE.clear()


# ===========================================================================
# A8.  blocks_downstream / build_link_blocks are read by nothing outside the
#      tests, so "blocks downstream" is a field value and not a gate.
# ===========================================================================
def test_blocks_downstream_must_be_consumed():
    print("\n[A8] `blocks_downstream` must actually block something")
    consumers = []
    for root, _dirs, files in os.walk(HERE):
        if "__pycache__" in root:
            continue
        for f in files:
            if not f.endswith(".py") or f.startswith("test_"):
                continue
            p = os.path.join(root, f)
            if os.path.basename(p) == "_provenance.py":
                continue
            with io.open(p, encoding="utf-8", errors="replace") as fh:
                body = fh.read()
            for ln in body.splitlines():
                code = ln.split("#")[0]
                if re.search(r"(blocks_downstream|build_link_blocks)", code) \
                        and '"' not in code and "'" not in code:
                    consumers.append("%s: %s" % (os.path.relpath(p, HERE),
                                                 ln.strip()[:60]))
    check("a non-test module branches on blocks_downstream / build_link_blocks",
          bool(consumers),
          "no consumer: the value is recorded and never read")


# ===========================================================================
# A9.  check_build_manifest() records build_type and the compilers but excludes
#      them from the comparison, and crashes on a null / list manifest instead
#      of returning the documented NOT_DETERMINABLE.
# ===========================================================================
def test_recorded_build_fields_must_be_compared():
    print("\n[A9] fields the manifest records must take part in the comparison")
    import _provenance as P
    td = tempfile.mkdtemp(prefix="adv_bm_")
    try:
        p = os.path.join(td, "bm.json")
        for f in ("build_type", "cxx_compiler", "cuda_compiler"):
            P.write_build_manifest(p)
            with open(p) as fh:
                st = json.load(fh)
            if st["build_config"].get(f) is None:
                continue
            st["build_config"][f] = "FORGED-" + f
            with open(p, "w") as fh:
                json.dump(st, fh)
            r = P.check_build_manifest(p)
            check("a forged build_config.%s is a MISMATCH" % f,
                  r["verdict"] == P.CHECK_MISMATCH,
                  "verdict=%s blocks_downstream=%s"
                  % (r["verdict"], r["blocks_downstream"]))
        for name, blob in (("JSON null", b"null"), ("JSON list", b"[]")):
            with open(p, "wb") as fh:
                fh.write(blob)
            try:
                r = P.check_build_manifest(p)
                ok = (r["verdict"] == P.CHECK_UNKNOWN
                      and bool(r["blocks_downstream"]))
                detail = "verdict=%s" % r["verdict"]
            except Exception as exc:                           # noqa: BLE001
                ok, detail = False, "raised %s" % type(exc).__name__
            check("a %s manifest returns NOT_DETERMINABLE, not a crash" % name,
                  ok, detail)
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ===========================================================================
# A10.  Checks in test_record_corrections.py that pass either way.
# ===========================================================================
def test_no_check_passes_vacuously():
    print("\n[A10] no check may pass on an empty or unrelated document")
    p = os.path.join(HERE, "CORRECTIONS_2026-09-18.md")
    original = _read_bytes(p)
    try:
        _write_bytes(p, _lines(original, "# Corrections", "", "none."))
        code = ("import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');"
                "import test_record_corrections as T;"
                "T.test_corrections_index();"
                "print('OPEN_COUNT', len(T.FAILS), T.NCHECK)" % (HERE, RUNS))
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, cwd=HERE, timeout=600)
        m = re.search(r"OPEN_COUNT (\d+) (\d+)", r.stdout)
        nf, nc = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        check("an EMPTY corrections index fails every structural check",
              nf is not None and nf == nc,
              "%s of %s checks failed; the rest pass on zero items "
              "(all()/count over an empty set)" % (nf, nc))
    finally:
        _write_bytes(p, original)
        assert _read_bytes(p) == original, "failed to restore %s" % p

    # scope the grep to the README test's OWN body: a sha256_file import
    # elsewhere in the file does not make this check compare anything.
    t = doc("test_record_corrections.py")
    body = t[t.find("def test_handover_readme_corrected"):]
    body = body[:body.find("\ndef ", 1)] if "\ndef " in body[1:] else body
    check("the README sha256 check compares the hash to the real file",
          re.search(r"sha256_file|hashlib|sha256\(", body) is not None,
          "its body only requires SOME 64-hex string to appear in the README "
          "(the quoted hash does in fact verify: "
          "dfbd15c2...540b71 == sha256(src/sim40_eps_kresolved.py))")
    check("the four HELD-OUT values are read from out/results.json at runtime",
          re.search(r"results\.json", t) is not None,
          "they are hard-coded in the test and only grepped in the doc; the "
          "values themselves do verify correctly against out/results.json")


def main():
    for fn in (test_reinstating_the_retracted_verdict_must_fail_a_test,
               test_deleting_the_verdict_block_must_be_caught,
               test_build_tree_binary_must_come_from_the_selected_tree,
               test_source_list_must_cover_the_integrated_path,
               test_editing_an_uncovered_kernel_must_move_the_identity,
               test_identity_must_not_collapse_without_an_engine_binary,
               test_installed_binary_must_follow_import_resolution,
               test_blocks_downstream_must_be_consumed,
               test_recorded_build_fields_must_be_compared,
               test_no_check_passes_vacuously):
        try:
            fn()
        except Exception as exc:                               # noqa: BLE001
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
