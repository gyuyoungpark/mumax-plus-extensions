"""NEW counterexamples against the single-source-of-truth verdict mechanism.

Written 2026-09-18, second correction round, AFTER the mechanism was built, to
attack it from angles the existing counterexamples (A1/A2/A10 in
test_adversarial_record_provenance.py) do not use.  Those two attacked the
document only, and both from the same place: injecting a superseding verdict
immediately before `## 1. Tolerances`, and deleting the block between the same
two anchors.  A mechanism that merely regenerates that one region would pass
both while still being trivially defeatable.

The seven checks here (six attacks and one no-over-blocking control):

  N1  inject the superseding verdict at the END of the document, outside every
      anchor the old counterexamples use and outside the generated region.
  N2  do not touch the document at all: edit VERDICT.json -- the new single
      source of truth -- and REGENERATE, so record and document agree perfectly.
      This is the defect the mechanism itself introduces, and the reason every
      recorded number names an evidence file and key.
  N3  the same attack aimed straight at the retraction: make
      `single_wave_sufficient` the operative verdict in the record and
      regenerate.
  N4  a plausible transcription slip in the record (18.081970 -> 1.808197),
      regenerated so nothing looks inconsistent.
  N5  remove the END marker only -- a half-finished hand edit -- which must be
      refused rather than silently skipped.
  N6  hand-edit a number INSIDE the generated section of the document without
      touching the record: the original drift defect, from the other direction.

  N7  the no-over-blocking control: with nothing mutated, every verdict check
      passes and `verdict_record.py --check` reports the record consistent.

Every attack mutates bytes, re-runs ONLY test_record_corrections.py's verdict
test in a fresh process, and restores the exact original bytes in a finally
block, asserting the restore.  Running the isolated test rather than the whole
suite is deliberate: other fixes land in this tree during the same round, and a
whole-suite "something failed" assertion would pass on their failures and prove
nothing here.  Every check below is differential against a baseline that is
asserted green first.  Nothing here touches the engine tree.

Run:  python test_adversarial_verdict_source.py
Exits non-zero while any counterexample still reproduces.
"""

import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
if RUNS not in sys.path:
    sys.path.insert(0, RUNS)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DOC = os.path.join(HERE, "model_comparison.md")
REC = os.path.join(HERE, "VERDICT.json")

OPEN = []
NCHECK = 0


def check(name, ok, detail=""):
    """ok=True means the defect is CLOSED (the attack was refused)."""
    global NCHECK
    NCHECK += 1
    print(("  CLOSED  " if ok else "  OPEN    ") + name
          + (("   " + detail) if detail else ""))
    if not ok:
        OPEN.append(name + (("   " + detail) if detail else ""))


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _write(path, blob):
    with open(path, "wb") as fh:
        fh.write(blob)


def _verdict_test():
    """Run ONLY test_record_corrections.py's verdict test, in a fresh process.

    Deliberately not the whole suite.  Other fixes land in the same tree during
    this round, so a whole-suite "nfailed > 0" assertion would pass on THEIR
    failures and prove nothing about this mechanism -- the same vacuity defect
    this round is removing elsewhere.  Isolating the verdict test makes every
    check below differential against a known-green baseline.
    """
    code = ("import sys; sys.path.insert(0, r'%s'); sys.path.insert(0, r'%s');"
            "import test_record_corrections as T;"
            "T.test_formal_verdict_is_generated_from_one_record();"
            "print('RESULT', len(T.FAILS), T.NCHECK)"
            % (HERE, os.path.join(HERE, "runs")))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=HERE, timeout=1200)
    out = r.stdout + r.stderr
    m = re.search(r"RESULT (\d+) (\d+)", out)
    if not m:
        return (None, None, out)
    return (int(m.group(2)), int(m.group(1)), out)          # (nchecks, nfailed)


def _suite():
    """Kept for the whole-suite information line in N7; never asserted on."""
    r = subprocess.run([sys.executable, "test_record_corrections.py"],
                       capture_output=True, text=True, cwd=HERE, timeout=2400)
    m = re.search(r"^(\d+) checks, (\d+) FAILED", r.stdout, re.M)
    if not m:
        return (None, None, r.stdout + r.stderr)
    return (int(m.group(1)), int(m.group(2)), r.stdout)


def _fails(out):
    return sorted({ln.strip() for ln in (out or "").splitlines()
                   if ln.strip().startswith("FAIL")})


BASELINE = None


def _baseline():
    """The verdict test's failure count on the untouched tree.  Must be 0."""
    global BASELINE
    if BASELINE is None:
        n, nf, out = _verdict_test()
        assert nf == 0, ("baseline is not green (%s of %s failed), so no "
                         "differential check below means anything:\n%s"
                         % (nf, n, out[-600:]))
        BASELINE = n
    return BASELINE


def _regenerate():
    """Regenerate the document section from the (possibly mutated) record."""
    r = subprocess.run([sys.executable, "verdict_record.py", "--write"],
                       capture_output=True, text=True, cwd=HERE, timeout=600)
    return r.returncode, (r.stdout + r.stderr)


SUPERSEDING = [
    "",
    "## VERDICT (SUPERSEDING)",
    "",
    "> On review the operative verdict of this comparison is",
    "> **`single_wave_sufficient`**: one off-grid spatial structure",
    "> suffices, and the earlier `indistinguishable /",
    "> both_models_inadequate` label is withdrawn as an artefact of the",
    "> plane-wave parameterisation.  The retraction above no longer applies.",
    "",
]


# ===========================================================================
# N1.  The superseding verdict appended at the END of the document, outside
#      every anchor the existing counterexamples use.
# ===========================================================================
def test_superseding_verdict_at_end_of_document():
    print("\n[N1] a superseding verdict APPENDED AT THE END must break the suite")
    original = _read(DOC)
    eol = b"\r\n" if b"\r\n" in original else b"\n"
    inj = eol.join(s.encode("utf-8") for s in SUPERSEDING) + eol
    try:
        _write(DOC, original + inj)
        n, nf, out = _verdict_test()
        check("a superseding verdict outside the generated region is rejected",
              bool(nf), "verdict test returned %s checks, %s FAILED (baseline %s/0)" % (n, nf, _baseline()))
        check("the refusal names the verdict section, not an unrelated check",
              any("verdict section" in f for f in _fails(out)),
              "; ".join(f[:70] for f in _fails(out)) or "no FAIL lines")
    finally:
        _write(DOC, original)
        assert _read(DOC) == original, "failed to restore %s" % DOC


# ===========================================================================
# N2.  The record itself is edited and the document regenerated, so the two
#      agree.  Only the evidence files can catch this.
# ===========================================================================
def _mutate_record(fn):
    """Apply fn to the loaded record, rewrite it, regenerate the document."""
    rec = json.loads(_read(REC).decode("utf-8"))
    fn(rec)
    _write(REC, (json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
           .encode("utf-8"))
    return _regenerate()


def test_editing_the_record_and_regenerating_must_fail_on_evidence():
    print("\n[N2] loosening tau_adeq in the RECORD and regenerating must fail")
    orig_rec, orig_doc = _read(REC), _read(DOC)
    try:
        def loosen(rec):
            rec["tolerance"]["value"] = "0.50"
        rc, _log = _mutate_record(loosen)
        n, nf, out = _verdict_test()
        check("a tolerance the pre-registration does not carry is rejected "
              "even with the document regenerated to match",
              bool(nf),
              "regenerate rc=%d, verdict test returned %s checks, %s FAILED "
              "(baseline %s/0)" % (rc, n, nf, _baseline()))
        check("the refusal names the evidence comparison",
              any("verifies against the evidence" in f for f in _fails(out)),
              "; ".join(f[:70] for f in _fails(out)) or "no FAIL lines")
    finally:
        _write(REC, orig_rec)
        _write(DOC, orig_doc)
        assert _read(REC) == orig_rec and _read(DOC) == orig_doc, "restore failed"


# ===========================================================================
# N3.  Reinstating the retracted verdict IN THE RECORD, document regenerated.
# ===========================================================================
def test_reinstating_the_retracted_verdict_in_the_record():
    print("\n[N3] reinstating `single_wave_sufficient` in the RECORD must fail")
    orig_rec, orig_doc = _read(REC), _read(DOC)
    try:
        def reinstate(rec):
            rec["verdict"]["operative"] = "single_wave_sufficient"
        rc, _log = _mutate_record(reinstate)
        n, nf, out = _verdict_test()
        fails = _fails(out)
        check("the retracted label cannot be made operative in the record",
              bool(nf),
              "regenerate rc=%d, verdict test returned %s checks, %s FAILED "
              "(baseline %s/0)" % (rc, n, nf, _baseline()))
        check("the refusal cites out/results.json, which never returned it",
              any("single_wave_sufficient" in f or "evidence" in f
                  for f in fails),
              "; ".join(f[:70] for f in fails) or "no FAIL lines")
    finally:
        _write(REC, orig_rec)
        _write(DOC, orig_doc)
        assert _read(REC) == orig_rec and _read(DOC) == orig_doc, "restore failed"


# ===========================================================================
# N4.  A plausible transcription slip in a HELD-OUT value, regenerated.
# ===========================================================================
def test_a_wrong_heldout_value_in_the_record():
    print("\n[N4] a mistyped HELD-OUT value in the RECORD must fail on evidence")
    orig_rec, orig_doc = _read(REC), _read(DOC)
    try:
        def slip(rec):
            for h in rec["held_out"]:
                if h["value"] == "18.081970":
                    h["value"] = "1.808197"
        rc, _log = _mutate_record(slip)
        n, nf, out = _verdict_test()
        check("a HELD-OUT value that out/results.json does not carry is "
              "rejected", bool(nf),
              "regenerate rc=%d, verdict test returned %s checks, %s FAILED "
              "(baseline %s/0)" % (rc, n, nf, _baseline()))
        # the mis-typed value also moves the sub-label derivation, so check the
        # evidence comparison itself fired
        check("the refusal names the evidence comparison",
              any("verifies against the evidence" in f for f in _fails(out)),
              "; ".join(f[:70] for f in _fails(out)) or "no FAIL lines")
    finally:
        _write(REC, orig_rec)
        _write(DOC, orig_doc)
        assert _read(REC) == orig_rec and _read(DOC) == orig_doc, "restore failed"


# ===========================================================================
# N5.  A half-finished hand edit: the END marker removed.
# ===========================================================================
def test_removing_the_end_marker_must_be_refused():
    print("\n[N5] removing the END marker must be refused, not skipped")
    original = _read(DOC)
    rec = json.loads(_read(REC).decode("utf-8"))
    end = rec["target"]["end_marker"].encode("utf-8")
    assert original.count(end) == 1, "END marker is not unique; update this test"
    try:
        _write(DOC, original.replace(end, b""))
        n, nf, out = _verdict_test()
        check("an unmarked verdict section is refused rather than skipped",
              bool(nf), "verdict test returned %s checks, %s FAILED (baseline %s/0)" % (n, nf, _baseline()))
        check("the refusal says the markers are wrong",
              any("marker" in f for f in _fails(out)),
              "; ".join(f[:70] for f in _fails(out)) or "no FAIL lines")
    finally:
        _write(DOC, original)
        assert _read(DOC) == original, "failed to restore %s" % DOC


# ===========================================================================
# N6.  Prose drift, the original defect, from the other direction: edit a
#      number inside the generated section and leave the record alone.
# ===========================================================================
def test_editing_the_generated_section_by_hand():
    print("\n[N6] hand-editing a number inside the generated section must fail")
    original = _read(DOC)
    assert original.count(b"306.67") >= 1, "anchor moved; update this test"
    try:
        _write(DOC, original.replace(b"**306.67 / 150.34 / 38.73**",
                                     b"**30.67 / 150.34 / 38.73**", 1))
        assert _read(DOC) != original, "the replacement did not apply"
        n, nf, out = _verdict_test()
        check("a hand-edited number in the generated section is rejected",
              bool(nf), "verdict test returned %s checks, %s FAILED (baseline %s/0)" % (n, nf, _baseline()))
        check("the refusal points at the byte-identity comparison",
              any("byte-identical" in f for f in _fails(out)),
              "; ".join(f[:70] for f in _fails(out)) or "no FAIL lines")
    finally:
        _write(DOC, original)
        assert _read(DOC) == original, "failed to restore %s" % DOC


# ===========================================================================
# N7.  The valid path must still go through: with nothing mutated, the suite
#      is green.  A gate that refuses everything is not a gate.
# ===========================================================================
def test_the_unmutated_tree_is_green():
    print("\n[N7] the unmutated tree must still pass (no over-blocking)")
    n, nf, _out = _verdict_test()
    check("the untouched record and document pass every verdict check",
          nf == 0, "%s checks, %s FAILED" % (n, nf))
    r = subprocess.run([sys.executable, "verdict_record.py"],
                       capture_output=True, text=True, cwd=HERE, timeout=600)
    check("`python verdict_record.py --check` reports the record consistent",
          r.returncode == 0 and "VERDICT RECORD CONSISTENT" in r.stdout,
          "rc=%d" % r.returncode)
    # Whole-suite state, reported and NOT asserted on: other fixes are landing
    # in this tree during the same round, and their failures are not evidence
    # about this mechanism either way.
    sn, snf, sout = _suite()
    others = [f for f in _fails(sout)
              if "verdict" not in f and "VERDICT" not in f
              and "HELD-OUT values" not in f]
    print("   (info) whole suite: %s checks, %s FAILED, %d of them outside the "
          "verdict mechanism%s"
          % (sn, snf, len(others),
             (": " + "; ".join(f[8:58] for f in others[:4])) if others else ""))


def main():
    # Prime the baseline BEFORE anything is mutated: every check below is
    # differential against it, so measuring it on a mutated tree would be
    # exactly the mistake these counterexamples exist to catch.
    try:
        print("baseline (untouched tree): %d verdict checks, 0 FAILED"
              % _baseline())
    except AssertionError as exc:
        check("the baseline is green before any attack runs", False, str(exc))
        print("%d counterexamples, %d still OPEN" % (NCHECK, len(OPEN)))
        return 1
    for fn in (test_superseding_verdict_at_end_of_document,
               test_editing_the_record_and_regenerating_must_fail_on_evidence,
               test_reinstating_the_retracted_verdict_in_the_record,
               test_a_wrong_heldout_value_in_the_record,
               test_removing_the_end_marker_must_be_refused,
               test_editing_the_generated_section_by_hand,
               test_the_unmutated_tree_is_green):
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
