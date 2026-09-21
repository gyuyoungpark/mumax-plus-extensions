"""Regression tests for the 2026-09-18 written-record and provenance corrections.

Every test here encodes a COUNTEREXAMPLE or a MISSTATEMENT that the independent
audit (Downloads/SAW_handover_verification_2026-09-18.md) reproduced.  Each one
is written to FAIL against the pre-correction tree and to PASS after the fix, so
the record itself is pinned and cannot silently regress.

Run:  python test_record_corrections.py
It prints one PASS/FAIL line per check and exits non-zero on any failure.
No test here fabricates a number: every numeric expectation is either computed
in the test from a file on disk, or is a value the audit reproduced and this
test recomputes from first principles.
"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
if RUNS not in sys.path:
    sys.path.insert(0, RUNS)

FAILS = []
NCHECK = 0


def check(name, ok, detail=""):
    global NCHECK
    NCHECK += 1
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not ok:
        FAILS.append(name + (("   " + detail) if detail else ""))


def doc(name):
    p = os.path.join(HERE, name)
    if not os.path.isfile(p):
        return ""
    with io.open(p, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def has(text, *needles):
    return all(n in text for n in needles)


def section(text, start_marker, end_marker=None):
    i = text.find(start_marker)
    if i < 0:
        return ""
    if not end_marker:
        return text[i:]
    j = text.find(end_marker, i + len(start_marker))
    return text[i:] if j < 0 else text[i:j]


def _raises(fn):
    try:
        fn()
    except Exception:                                         # noqa: BLE001
        return True
    return False


# ===========================================================================
# 1. THE FORMAL VERDICT IS GENERATED FROM ONE RECORD  (audit sec.3, task item 2)
#
# 2026-09-18, second round.  This test used to be four whole-file substring
# greps ("`indistinguishable` appears somewhere", "0.25 appears somewhere").
# Two defects of one class followed: the prose could state a number the evidence
# files do not support, and an ADDED verdict paragraph was invisible to every
# positive grep, so the retracted `single_wave_sufficient` label could be
# reinstated with the suite fully green.
#
# The greps are deleted.  VERDICT.json is now the single source of truth and the
# VERDICT section of model_comparison.md is GENERATED from it by
# verdict_record.py, so the whole section is pinned by ONE comparison instead of
# a fleet of greps, and every recorded number is re-read from the evidence file
# and key the record names.
# ===========================================================================
def test_formal_verdict_is_generated_from_one_record():
    print("\n[1] the VERDICT section must be generated from VERDICT.json")
    import verdict_record as VR

    ev = VR.verify_evidence(root=HERE)
    bad = [name for name, ok, _d in ev if not ok]
    check("every value in VERDICT.json verifies against the evidence file and "
          "key it names (out/results.json, out/adequacy.json, "
          "PREREGISTRATION.md)",
          len(ev) >= 14 and not bad,
          "%d values checked, %d bad%s"
          % (len(ev), len(bad), (": " + "; ".join(bad[:3])) if bad else ""))

    ok, detail = VR.check_section(root=HERE)
    check("the generated verdict section is byte-identical to the one "
          "model_comparison.md carries", ok, detail)

    ok, detail = VR.check_unique_heading(root=HERE)
    check("model_comparison.md declares exactly one verdict section and it is "
          "the generated one", ok, detail)

    # The two named checks below are redundant for DETECTION -- the section
    # comparison above already pins every character -- and are kept only so a
    # failure names the thing that broke.
    rec = VR.load(HERE)
    sec, why = VR.extract(rec, HERE)
    sec = sec or ""
    check("the generated section returns `indistinguishable / "
          "both_models_inadequate`",
          "`%s / %s`" % (rec["verdict"]["operative"],
                         rec["verdict"]["sub_label"]) in sec, why)
    with open(os.path.join(HERE, "out", "results.json")) as fh:
        res = json.load(fh)["fits_summary"]["7e-05"]
    vals = [(h["value"], res[h["config"]][h["symbol"].replace("E_M", "E")])
            for h in rec["held_out"]]
    check("the four HELD-OUT values 18.081970 / 0.998801 / 1.000063 / 1.051863 "
          "stand in the generated section and match out/results.json",
          len(vals) == 4
          and all(s in sec and abs(float(s) - f) <= 5e-7 for s, f in vals),
          "read from out/results.json fits_summary/7e-05")

    # The one negative assertion in this mechanism, and it is on the STRUCTURED
    # record, not on anybody's prose: the operative verdict field must not be a
    # retracted label, and the retracted label must be absent from the evidence
    # the verdict is read from.  Reinstating `single_wave_sufficient` therefore
    # means editing VERDICT.json, which fails here and again at the
    # evidence-comparison check above.
    retracted = [r["label"] for r in rec["verdict"]["retracted_labels"]]
    raw = doc(os.path.join("out", "results.json"))
    check("`single_wave_sufficient` is a retracted label, is not in the "
          "operative verdict fields, and is not in out/results.json",
          "single_wave_sufficient" in retracted
          and rec["verdict"]["operative"] not in retracted
          and rec["verdict"]["sub_label"] not in retracted
          and "single_wave_sufficient" not in raw,
          "retracted=%s operative=%s" % (retracted, rec["verdict"]["operative"]))


# ===========================================================================
# 2. THE THREE CATEGORIES ARE SEPARATED  (audit sec.3.1, task item 1)
# ===========================================================================
CAT_TRAIN = "TRAIN-FIT"
CAT_XFER = "TRANSFER-REFIT"
CAT_HELD = "HELD-OUT"


def test_three_category_legend():
    print("\n[2] a legend must define the three different things that were conflated")
    t = doc("model_comparison.md")
    check("legend defines %s (i)" % CAT_TRAIN, CAT_TRAIN in t)
    check("legend defines %s (ii)" % CAT_XFER, CAT_XFER in t)
    check("legend defines %s (iii)" % CAT_HELD, CAT_HELD in t)
    check("legend says (ii) REFITS amplitudes on the new block",
          CAT_XFER in t and re.search(r"refit", t, re.I) is not None)
    check("legend says (iii) freezes ALL fitted parameters",
          re.search(r"HELD-OUT[^\n]*froz|froz[^\n]*HELD-OUT", t) is not None)
    check("zero-prediction baseline E = 1 stated", "zero-prediction" in t)


def test_heldout_numbers_carry_the_heldout_tag():
    print("\n[2b] the four formal numbers are category (iii) and must be tagged")
    # The four "quoted at full precision" whole-file greps that used to sit here
    # are deleted: test [1] now reads each value from out/results.json and
    # compares it against the generated section.  What remains is the one thing
    # [1] does NOT cover -- that the values are tagged HELD-OUT wherever they
    # appear in the hand-written rest of the document.
    t = doc("model_comparison.md")
    lines = [ln for ln in t.splitlines() if "18.081970" in ln or "1.000063" in ln]
    check("held-out values sit on lines tagged %s" % CAT_HELD,
          bool(lines) and all(CAT_HELD in ln for ln in lines),
          "%d such lines" % len(lines))


def test_adequacy_numbers_are_labelled_train():
    print("\n[2c] the adequacy ratios are TRAIN residual / floor, not held-out")
    t = doc("model_comparison.md")
    sec5 = section(t, "## 5.", "## 6.")
    check("section 5 exists", bool(sec5))
    check("section 5 says the residual is the TRAIN residual", CAT_TRAIN in sec5)
    check("section 5 states n_complex = 14 = 2 TRAIN blocks x 7 bins",
          re.search(r"2 TRAIN blocks", sec5) is not None)


# ===========================================================================
# 3. THE PACKET COMPARISON IS EXPLORATORY, POST-PRE-REGISTRATION (task item 2)
# ===========================================================================
def test_exploratory_packet_section():
    print("\n[3] the Gaussian-packet comparison must be quarantined and qualified")
    t = doc("model_comparison.md")
    sec = section(t, "## 15.")
    check("a section 15 EXPLORATORY block exists", bool(sec) and "EXPLORATORY" in sec)
    check("it states it was added AFTER pre-registration",
          re.search(r"after\s+pre-registration", sec, re.I) is not None)
    check("it states the Monte-Carlo driver is ABSENT from the bundle",
          re.search(r"driver", sec, re.I) is not None and
          re.search(r"absent|not in the bundle|missing", sec, re.I) is not None)
    check("it names COMMANDS.txt / the external shell history as the only pointer",
          "COMMANDS.txt" in sec and re.search(r"shell history", sec, re.I) is not None)
    for v in ("94.6", "96.7", "74.1"):
        check("packet/two-wave figure %s is tagged %s" % (v, CAT_TRAIN),
              any(v in ln and CAT_TRAIN in ln for ln in sec.splitlines()))
    for v in ("0.1207", "0.1217"):
        check("transfer figure %s is tagged %s" % (v, CAT_XFER),
              any(v in ln and CAT_XFER in ln for ln in sec.splitlines()))
    check("it puts 0.1207 and 0.1217 side by side",
          re.search(r"0\.1207.{0,400}0\.1217|0\.1217.{0,400}0\.1207", sec, re.S)
          is not None)


def test_aicc_parameter_count_corrected():
    print("\n[3b] Delta AICc for the two-packet model, recounted from the definition")
    txt = doc(os.path.join("adversarial", "attackN.txt"))
    m4 = float(re.search(r"M4 one packet\s+RSS=([0-9.eE+-]+)", txt).group(1))
    m5 = float(re.search(r"M5 two packets\s+RSS=([0-9.eE+-]+)", txt).group(1))
    n = 28                                       # 2 TRAIN blocks x 7 bins x 2 real

    def aicc(rss, k):
        return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

    d12 = aicc(m5, 12) - aicc(m4, 7)
    d14 = aicc(m5, 14) - aicc(m4, 7)
    check("recorded k=12 reproduces the published -14.55", abs(d12 + 14.55) < 0.01,
          "computed %.4f" % d12)
    check("k=14 (6 shape + 8 block amplitudes) gives +0.953",
          abs(d14 - 0.953) < 0.01, "computed %+.4f" % d14)
    adv = os.path.join(HERE, "adversarial")
    src = "".join(doc(os.path.join("adversarial", f))
                  for f in os.listdir(adv) if f.endswith(".py"))
    check("no M5 / two-packet fitter exists in the bundle to justify k=12",
          re.search(r"\bM5\b|two[_ ]packet", src) is None,
          "searched every .py in adversarial/")
    sec = section(doc("model_comparison.md"), "## 15.")
    check("doc quotes the corrected +0.953", "+0.953" in sec)
    check("doc states the count as 14 = 6 shape + 8 block amplitudes",
          re.search(r"14\b[^\n]*6 shape[^\n]*8 block", sec) is not None)


# ===========================================================================
# 4. FOUR SPECIFIC MISSTATEMENTS  (task item 3)
# ===========================================================================
def test_percentile_not_probability():
    print("\n[4a] P(R <= R_obs) = 0.998 is a percentile, not a support probability")
    sec = section(doc("model_comparison.md"), "## 15.")
    check("0.998 is quoted", "0.998" in sec)
    check("it is called a percentile of the statistic R",
          re.search(r"percentile", sec, re.I) is not None)
    check("the opposite tail 0.002 is given", "0.002" in sec)
    check("the doc forbids reading it as a probability the model is true",
          re.search(r"not\s+(?:a\s+)?probability", sec, re.I) is not None)


def test_zero_of_500_upper_bound():
    print("\n[4b] 0/500 needs its finite-Monte-Carlo upper bound")
    ub = 1.0 - 0.05 ** (1.0 / 500.0)
    check("one-sided 95% upper bound computes to ~0.006", abs(ub - 0.005974) < 1e-5,
          "%.6f" % ub)
    sec = section(doc("model_comparison.md"), "## 15.")
    check("0/500 is quoted with its 0.006 upper bound",
          "0/500" in sec and "0.006" in sec)
    check("doc states it does not exclude all plane-wave / residual models",
          re.search(r"does not exclude", sec, re.I) is not None)


def test_pump_bin_convention_and_provenance():
    print("\n[4c] the pump-bin figure: convention, mechanism, and who measured it")
    V, F = 3500.0, 6.0e9
    Q = 2 * np.pi * F / V
    nx, dx = 1024, 5e-9
    x = (np.arange(nx) + 0.5) * dx
    S_r = np.abs(np.fft.rfft(np.sin(Q * x))) ** 2
    S_r /= S_r.sum()
    S_c = np.abs(np.fft.fft(np.exp(1j * Q * x))) ** 2
    S_c /= S_c.sum()
    S_2 = np.abs(np.fft.fft(np.sin(Q * x))) ** 2
    S_2 /= S_2.sum()
    check("rfft-of-sin one-sided bin 9 = 0.847703", abs(S_r[9] - 0.847702673) < 1e-9,
          "%.9f" % S_r[9])
    check("complex travelling wave bin 9 = 0.846921", abs(S_c[9] - 0.846920775) < 1e-9,
          "%.9f" % S_c[9])
    check("two-sided fft-of-sin bin 9 = 0.424043 (the '42.40 %' variant)",
          abs(S_2[9] - 0.424043142) < 1e-9, "%.9f" % S_2[9])
    k_nyq = np.pi / dx
    check("q is ~58x BELOW the spatial Nyquist, so nothing folds",
          k_nyq / Q > 50, "k_Nyq/q = %.1f" % (k_nyq / Q))

    p = os.path.join(HERE, "runs", "out", "G4", "G4_pump_bin.npz")
    d = np.load(p, allow_pickle=True)
    ana = json.loads(str(d["analytic"]))
    eng = json.loads(str(d["engine"]))
    check("artefact: analytic covers BOTH boxes",
          {"commensurate_n12", "sim40_incommensurate"} <= set(ana))
    check("artefact: engine record has NO sim40/NX=1024 entry",
          eng.get("top_bin") == 12 and "sim40_incommensurate" not in str(eng),
          "engine top_bin=%s" % eng.get("top_bin"))

    rp = doc("RUN_PLAN.md")
    check("RUN_PLAN quotes 0.847703 (rfft of sin)", "0.847703" in rp)
    check("RUN_PLAN quotes the complex-wave 0.846921 alongside it", "0.846921" in rp)
    check("RUN_PLAN calls it incommensurate periodic-extension leakage",
          re.search(r"periodic-extension leakage", rp) is not None)
    check("RUN_PLAN explicitly denies Nyquist aliasing",
          re.search(r"not\s+(?:Nyquist\s+)?aliasing", rp, re.I) is not None)
    check("RUN_PLAN gives q = 10.771175 against the Nyquist 628",
          "10.771175" in rp and "628" in rp)
    check("RUN_PLAN denies demonstrated dynamical scattering",
          re.search(r"dynamical scattering", rp) is not None)
    check("RUN_PLAN states the NX=1024 value is ANALYTIC and the engine record"
          " covers only n=12",
          re.search(r"analytic", rp, re.I) is not None and
          re.search(r"engine[^\n]*n\s*=\s*12|n\s*=\s*12[^\n]*engine", rp) is not None)
    row0c = next((ln for ln in rp.splitlines() if ln.strip().startswith("| 0c ")), "")
    check("row 0c no longer asserts that the box 'aliases' the pump",
          bool(row0c) and "aliases the pump" not in row0c.lower()
          and "periodic-extension leakage" in row0c, row0c[:90])
    row0d = next((ln for ln in rp.splitlines() if ln.strip().startswith("| 0d")), "")
    check("row 0d no longer attributes an NX=1024 number to the engine",
          bool(row0d) and "0.847465" not in row0d, row0d[:90])


def test_k0_share_reconciled():
    print("\n[4d] the k=0 double-count share: 84.3885 % vs 99.81 %")
    raw = doc(os.path.join("out", "step1_ipair_raw.txt"))
    blk = section(raw, "k = 0 CONTRIBUTION to band (a) itself")
    row = next(ln for ln in blk.splitlines() if ln.strip().startswith("1e-05"))
    vals = [float(v) for v in row.split()[1:]]        # full_2fK, full_fK, mr_fK
    check("artefact: full_2fK share at eps=1e-5 is 99.8102 %",
          abs(vals[0] - 0.9981024) < 1e-6, "%.7f" % vals[0])
    check("artefact: mr_fK share at eps=1e-5 is 84.3885 %",
          abs(vals[2] - 0.8438850) < 1e-6, "%.7f" % vals[2])
    n = doc("NUMBERS_FOR_MANUSCRIPT.md")
    check("doc quotes 84.3885 % for the Fig. 6(b) divisor", "84.3885" in n)
    check("doc names the divisor as mr_fK at eps_0 = 1e-5",
          re.search(r"84\.3885[^\n]*mr_fK|mr_fK[^\n]*84\.3885", n) is not None)
    check("doc explains 99.8102 % as the full_2fK share under the same definition",
          "99.8102" in n and
          re.search(r"99\.8102[^\n]*full_2fK|full_2fK[^\n]*99\.8102", n) is not None)
    check("doc says the two use the SAME definition, different configuration",
          re.search(r"same definition", n, re.I) is not None)


def test_suite_count_reconciled():
    print("\n[4e] the control-suite count: 33 in the docs vs 32 from the suite")
    # The mechanism of the off-by-one, reproduced live so it cannot drift: a naive
    # grep for PASS over the suite log counts the ALL CONTROLS PASSED banner as a
    # 33rd 'item'.  The suite is being extended in this same round, so the item
    # count itself is not a stable number and the doc must stop asserting one.
    r = subprocess.run([sys.executable, os.path.join(HERE, "test_saw_analysis.py")],
                       capture_output=True, text=True, cwd=HERE, timeout=900)
    lines = r.stdout.splitlines()
    n_items = sum(1 for ln in lines if ln.startswith("  PASS  "))
    n_naive = sum(1 for ln in lines if "PASS" in ln)
    check("the suite ran and produced result items", n_items > 0, "%d items" % n_items)
    check("a naive PASS grep over-counts by exactly 1 (the banner)",
          n_naive == n_items + 1, "naive=%d items=%d" % (n_naive, n_items))
    check("the banner is the extra line",
          any("ALL CONTROLS PASSED" in ln for ln in lines))
    n = doc("NUMBERS_FOR_MANUSCRIPT.md")
    check("docs no longer claim 33 controls", re.search(r"33\s+controls", n) is None)
    check("docs record 32 as the as-shipped item count",
          re.search(r"32\b[^\n]*(?:as shipped|as-shipped|bundle)", n, re.I) is not None
          or re.search(r"(?:as shipped|as-shipped|bundle)[^\n]*\b32\b", n, re.I)
          is not None)
    check("docs explain the off-by-one (the ALL CONTROLS PASSED banner)",
          re.search(r"banner", n, re.I) is not None)
    check("docs no longer hard-code a control count at all",
          re.search(r"#\s*\d+\s+(?:controls|PASS items)", n) is None)
    check("the suite states its own item count, so no doc has to",
          re.search(r"^%d result items\." % n_items, r.stdout, re.M) is not None)
    # DISCRIMINATION_TABLE.md is REGENERATED by the suite (test_saw_analysis.py
    # line ~693), so the category statement has to live in the generator or it
    # is wiped.  Check the file the run just wrote.
    dt = doc("DISCRIMINATION_TABLE.md")
    check("the regenerated DISCRIMINATION_TABLE.md carries the category sweep",
          "Fit-quality categories" in dt and CAT_HELD in dt)
    check("its one fit residual row is tagged TRAIN-FIT",
          re.search(r"single-off-grid-sinusoid fit residual[^|]*TRAIN-FIT", dt)
          is not None)
    check("it states that no number in it is a held-out prediction",
          re.search(r"No number in this file is a held-out prediction", dt)
          is not None)


# ===========================================================================
# 5. PROVENANCE: a build manifest that LINKS sources+config -> binary
# ===========================================================================
def test_build_manifest_module_exists():
    print("\n[5a] a build manifest must exist and must LINK, not list side by side")
    try:
        import _provenance as P
    except Exception as exc:                                  # noqa: BLE001
        check("runs/_provenance.py imports", False, str(exc))
        return
    check("runs/_provenance.py imports", True)
    m = P.build_manifest()
    check("manifest carries a schema tag",
          str(m.get("schema", "")).startswith("saw_revision_check/build_manifest/"))
    for key in ("sources", "source_tree_digest", "build_config", "binary",
                "link", "link_status", "blocks_downstream", "manifest_id"):
        check("manifest has `%s`" % key, key in m)
    check("link_status is one of the four declared states",
          m["link_status"] in (P.LINK_LINKED, P.LINK_UNPROVEN, P.LINK_REFUTED,
                               P.LINK_UNKNOWN), str(m["link_status"]))
    check("the link block names the build tree the binary came from",
          "installed_equals_build_tree_output" in m["link"])
    check("the link block compares source mtimes to the binary mtime",
          "binary_newer_than_all_sources" in m["link"] and
          "sources_changed_since_build" in m["link"])
    check("build_config records the precision the binary was built with",
          "fp_precision" in m["build_config"])
    check("build_config records the source tree CMake was configured against",
          "cmake_home_directory" in m["build_config"])
    check("blocks_downstream is True unless the link is established",
          m["blocks_downstream"] == (m["link_status"] != P.LINK_LINKED),
          "status=%s blocks=%s" % (m["link_status"], m["blocks_downstream"]))


def test_build_manifest_write_and_check():
    print("\n[5b] the harness must WRITE the manifest and CHECK it against reality")
    import _provenance as P
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "build_manifest.json")
        check("check() on a missing manifest is NOT_DETERMINABLE",
              P.check_build_manifest(path)["verdict"] == P.CHECK_UNKNOWN)
        check("a missing manifest blocks downstream",
              P.check_build_manifest(path)["blocks_downstream"] is True)
        P.write_build_manifest(path)
        check("write_build_manifest writes the file", os.path.isfile(path))
        r = P.check_build_manifest(path)
        check("check() against an unchanged tree returns MATCH",
              r["verdict"] == P.CHECK_MATCH, str(r["verdict"]))
        check("MATCH carries no differing fields", r["diffs"] == [], str(r["diffs"]))
        with open(path) as fh:
            stored = json.load(fh)
        victim = sorted(stored["sources"])[0]
        stored["sources"][victim]["sha256"] = "0" * 64
        with open(path, "w") as fh:
            json.dump(stored, fh)
        r2 = P.check_build_manifest(path)
        check("a changed source hash is detected as MISMATCH",
              r2["verdict"] == P.CHECK_MISMATCH, str(r2["verdict"]))
        check("the mismatch NAMES the file that changed",
              any(victim in str(d) for d in r2["diffs"]), str(r2["diffs"])[:160])
        check("MISMATCH blocks downstream", r2["blocks_downstream"] is True)


def test_manifest_honest_about_sim40():
    print("\n[5c] the manifest document must not overclaim")
    import _provenance as P
    txt = doc("PROVENANCE_BUILD_MANIFEST.md")
    check("PROVENANCE_BUILD_MANIFEST.md exists", bool(txt))
    check("it says this fixes FORWARD provenance only",
          re.search(r"forward", txt, re.I) is not None)
    check("it says the sim40 link stays unrecoverable",
          "sim40" in txt and
          re.search(r"unrecoverable|cannot be recovered|not recoverable", txt, re.I)
          is not None)
    check("it states what the link does and does NOT prove",
          re.search(r"does not prove|cannot prove", txt, re.I) is not None)
    check("it names the four link states",
          all(s in txt for s in (P.LINK_LINKED, P.LINK_UNPROVEN, P.LINK_REFUTED,
                                 P.LINK_UNKNOWN)))


# ===========================================================================
# 6. THE PROVENANCE RECORD MUST PARTICIPATE IN THE SKIP/RESUME DECISION
#
# The run-manifest gate itself (P0-4) is implemented in runs/_gate.py +
# runs/_harness.py.  What is pinned here is that the FORWARD PROVENANCE fields
# -- the kernel source hash, the engine binary hash, the source-tree digest, the
# build precision and the build-link status -- are part of the identity that
# gate compares, so a rebuild between runs cannot be skipped over.  No second
# comparison is introduced: these tests assert the fields reach the ONE
# existing comparison.
# ===========================================================================
# UPDATED 2026-09-18 (round 2), fixtures only: the identity a checkpoint is
# compared on is now the CONDITION SET of runs/_conditions.py, so these fields
# are read out of the condition stamp's `build` group instead of out of a flat
# `run_manifest` dict, and the checkpoints below declare their physics (an
# undeclared checkpoint is now refused for reuse outright, which is the point of
# open item 6).  Every ASSERTION in this section is unchanged.
PROV_FIELDS = ("engine_binary_sha256", "source_tree_digest",
               "compiled_source_digest", "fp_precision", "build_link_status")

#: a complete physical declaration, so the checkpoint is reusable at all
DECL = dict(box_tag="n12_NX1400_dx5.0000nm", material_tag="YIGlit",
            B0=0.0506, eps0=7e-5, f_saw=6.0e9, temperature=0.0,
            rng_seed=20260917, a_seed=1e-5)


def _write_checkpoint(path, nt, nx, dt_rec, manifest=None):
    """A finished checkpoint, written by the harness itself."""
    import _harness as H
    br = H.BlockRun(path, nt=nt, nx=nx, dt_rec=dt_rec,
                    manifest=dict(DECL) if manifest is None else manifest)
    br.i0 = nt
    br.t_state = nt * dt_rec
    br.save({}, done=True)
    return br


def _stamp_of(path):
    """The condition stamp of a checkpoint: the identity the gate compares."""
    import _conditions as C
    with np.load(path, allow_pickle=True) as d:
        body, why = C.read(d)
    assert body is not None, why
    return body


def _read(path, key):
    with np.load(path, allow_pickle=True) as d:
        return str(d[key]) if key in d.files else None


def test_resume_refuses_on_dt_rec_change():
    print("\n[6a] the audit's counterexample: dt_rec 1 -> 0.123 must NOT be accepted")
    import _harness as H
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "block.npz")
        _write_checkpoint(p, 4, 8, 1.0)
        check("the stored checkpoint really says done=True",
              _read(p, "done") in ("True", "true"), str(_read(p, "done")))
        err = {}

        def again():
            H.BlockRun(p, nt=4, nx=8, dt_rec=0.123, manifest=dict(DECL))

        raised = _raises_msg(again, err)
        check("a dt_rec change BLOCKS instead of accepting done=True", raised,
              err.get("msg", "no exception")[:110])
        check("the refusal names dt_rec", "dt_rec" in err.get("msg", ""),
              err.get("msg", "")[:160])


def test_resume_refuses_when_identity_absent():
    print("\n[6b] a checkpoint with no manifest is NOT verifiable, so not reusable")
    import _harness as H
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "legacy.npz")
        np.savez(p, my_xt=np.zeros((4, 8), np.float32), next_index=4,
                 mag_state=np.zeros(0), t_state=4.0, done=True, nt_total=4)
        err = {}
        raised = _raises_msg(lambda: H.BlockRun(p, nt=4, nx=8, dt_rec=1.0), err)
        check("legacy done=True is not accepted on trust", raised,
              err.get("msg", "no exception")[:110])
        check("the refusal says the file cannot be shown to belong to this run",
              re.search(r"cannot be shown|no run manifest", err.get("msg", ""),
                        re.I) is not None, err.get("msg", "")[:160])


def test_forward_provenance_is_in_the_run_manifest():
    print("\n[6c] MY fix: the forward provenance fields are part of the identity")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "block.npz")
        _write_checkpoint(p, 4, 8, 1.0)
        man = _stamp_of(p)["conditions"]["build"]
        for f in PROV_FIELDS:
            check("run manifest carries `%s`" % f, f in man,
                  "keys: %s" % sorted(man)[:9])
        P = __import__("_provenance")
        smap = _stamp_of(p).get("source_map") or {}
        check("the kernel hash in the manifest is the real one",
              smap.get("src/physics/chiralsawfield.cu") ==
              P.sha256_file(P.KERNEL_CU),
              "recorded %s" % str(smap.get("src/physics/chiralsawfield.cu"))[:12])


def test_resume_refuses_on_kernel_change():
    print("\n[6d] a rebuilt kernel refuses the stored checkpoint, same code path")
    import _gate
    import _harness as H
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "block.npz")
        _write_checkpoint(p, 4, 8, 1.0)
        # forge the STORED manifest: pretend the checkpoint was made by a
        # different build of the kernel, exactly as a rebuild would look.
        import _conditions as C                                    # noqa: PLC0415
        with np.load(p, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        body = _stamp_of(p)
        smap = body.get("source_map") or {}
        if "src/physics/chiralsawfield.cu" not in smap:
            check("stored manifest carries the kernel hash to forge", False,
                  "source_map keys: %s" % sorted(smap)[:4])
            return
        # pretend the checkpoint was made by a different build of the kernel:
        # the recorded file hash AND the digest computed from it both move,
        # exactly as a rebuild would look.
        smap["src/physics/chiralsawfield.cu"] = "f" * 64
        body["source_map"] = smap
        body["conditions"]["build"]["compiled_source_digest"] = "f" * 64
        body["digest"] = C._digest(body["conditions"])
        blob = json.dumps(body, sort_keys=True, default=str)
        data[C.KEY] = blob
        data[C.KEY_ALIAS] = blob
        np.savez(p, **data)
        err = {}
        raised = _raises_msg(
            lambda: H.BlockRun(p, nt=4, nx=8, dt_rec=1.0,
                               manifest=dict(DECL)), err)
        check("a changed kernel hash refuses the stored done=True", raised,
              err.get("msg", "no exception")[:110])
        check("the refusal names the kernel source",
              "chiralsawfield" in err.get("msg", ""), err.get("msg", "")[:200])


def test_resume_accepts_a_matching_identity():
    print("\n[6e] and it must still resume when the identity genuinely matches")
    import _harness as H
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "block.npz")
        _write_checkpoint(p, 4, 8, 1.0)
        err = {}
        br = None

        def again():
            nonlocal br
            br = H.BlockRun(p, nt=4, nx=8, dt_rec=1.0, manifest=dict(DECL))

        check("a matching identity does NOT raise", not _raises_msg(again, err),
              err.get("msg", "")[:120])
        check("and it resumes as done", br is not None and br.done is True,
              "done=%s" % (br.done if br else "<no object>"))


def test_provenance_block_records_the_build_link():
    print("\n[6f] the harness provenance block carries the build-manifest link")
    import _harness as H
    import _provenance as P
    pr = H.provenance(os.path.join(HERE, "test_record_corrections.py"), {"t": 1})
    for f in ("build_manifest_id", "build_link_status", "build_manifest_check",
              "build_manifest_path", "source_tree_digest"):
        check("provenance() records `%s`" % f, f in pr, str(sorted(pr))[:120])
    check("build_link_status is one of the four states",
          pr.get("build_link_status") in (P.LINK_LINKED, P.LINK_UNPROVEN,
                                          P.LINK_REFUTED, P.LINK_UNKNOWN),
          str(pr.get("build_link_status")))
    check("build_manifest_check is one of the three verdicts",
          pr.get("build_manifest_check") in (P.CHECK_MATCH, P.CHECK_MISMATCH,
                                             P.CHECK_UNKNOWN),
          str(pr.get("build_manifest_check")))
    check("the harness WROTE the manifest it reports",
          os.path.isfile(str(pr.get("build_manifest_path", ""))),
          str(pr.get("build_manifest_path")))


def _raises_msg(fn, out):
    try:
        fn()
    except Exception as exc:                                  # noqa: BLE001
        out["msg"] = "%s: %s" % (type(exc).__name__, exc)
        return True
    return False


# ===========================================================================
# 7. THE HANDOVER README  (task item 5)
# ===========================================================================
NPZ = os.path.join(os.path.dirname(HERE), "data", "sim40_checkpoints",
                   "sim40_full_2fK_eps7e-05.npz")


def test_handover_readme_corrected():
    print("\n[7] README_HANDOVER.md must match what the nine NPZs actually carry")
    keys = set(np.load(NPZ, allow_pickle=True).files)
    absent = [k for k in ("msat", "Ms", "MS", "aex", "AEX", "alpha", "ALPHA",
                          "NZ", "CY", "CZ", "pbc", "pbc_repetitions",
                          "magnetization", "m_init", "engine", "build")
              if k in keys]
    check("artefact: none of Ms/A/alpha/NZ/CY/CZ/PBC/init/build is in the npz",
          absent == [], "found %s" % absent)
    t = doc("README_HANDOVER.md")
    check("README_HANDOVER.md exists", bool(t))
    missing = sorted(k for k in keys if k not in t)
    check("README lists every real npz key", missing == [], "missing %s" % missing)
    check("README states Ms, A, alpha are NOT in the npz",
          re.search(r"\*\*not in the npz\*\*", t, re.I) is not None)
    # The phrase may appear ONLY as the claim being withdrawn, never as a claim.
    claims = [m.start() for m in re.finditer(r"full run configuration", t, re.I)]
    retracted = all(re.search(r"They do not|does not|replaces|instead of claiming",
                              t[max(0, i - 400):i + 400])
                    for i in claims)
    check("'full run configuration' appears only as a withdrawn claim",
          retracted, "%d occurrence(s)" % len(claims))
    check("README references src/sim40_eps_kresolved.py as the config of record",
          "sim40_eps_kresolved.py" in t)
    # Same class of defect as the deleted verdict greps: this used to require
    # that SOME 64-hex string appear anywhere in the README, which any 64-hex
    # string satisfies.  Now the README's own hash is compared to the file.
    cfg = os.path.join(os.path.dirname(HERE), "src", "sim40_eps_kresolved.py")
    real = (hashlib.sha256(open(cfg, "rb").read()).hexdigest()
            if os.path.isfile(cfg) else "<config file missing>")
    check("the sha256 the README gives for that file is the file's real sha256",
          real in t, "sha256(%s) = %s" % (os.path.basename(cfg), real[:16] + "..."))
    check("README states the packaged checkpoint hashes cannot be verified by a"
          " recipient without the originals",
          re.search(r"cannot be verified|not verifiable", t, re.I) is not None)
    check("README states the engine build is unrecoverable for these nine files",
          re.search(r"engine build", t, re.I) is not None and
          re.search(r"unrecoverable|NOT DETERMINABLE", t) is not None)


# ===========================================================================
# 8. THE CORRECTIONS INDEX  (task item 6)
# ===========================================================================
MIN_ITEMS = 10


def test_corrections_index():
    print("\n[8] CORRECTIONS_2026-09-18.md must be a complete old -> new index")
    # 2026-09-18, second round.  Every aggregate here used to be of the form
    # `count(x) >= len(items)` or `all(...)` over a list, which is TRUE on an
    # empty document: replacing the whole index with "# Corrections / none."
    # failed 2 of 9 checks and passed the other 7 on zero items.  Deleted: the
    # bare "file exists" check (subsumed -- a missing file fails everything
    # below) and the three per-tag OLD:/NEW:/WHERE: counts.  Every surviving
    # aggregate is now guarded by `n >= MIN_ITEMS` and by a non-empty
    # population, so no check in this test can pass on an empty index.
    t = doc("CORRECTIONS_2026-09-18.md")
    items = re.findall(r"^### C\d+\b.*$", t, re.M)
    n = len(items)
    enough = n >= MIN_ITEMS
    check("the corrections index is itemised with >= %d stable C-numbers"
          % MIN_ITEMS, enough, "%d items" % n)

    tags = {tag: t.count(tag) for tag in ("OLD:", "NEW:", "WHERE:")}
    check("every item carries OLD:, NEW: and WHERE:",
          enough and all(c >= n for c in tags.values()),
          "%s against %d items" % (tags, n))

    refs = re.findall(r"`([A-Za-z0-9_./-]+\.(?:md|py|txt|json)):(\d+)`", t)
    n_where = len(re.findall(r"^\s*\*\s*\*\*WHERE:\*\*", t, re.M))
    check("every item has a WHERE line and the WHERE lines point at file:line",
          enough and n_where >= n and len(refs) >= n,
          "%d WHERE lines, %d file:line refs, %d items" % (n_where, len(refs), n))

    root = os.path.dirname(HERE)                      # SAW-magnonics
    bad = []
    for f, ln in refs:
        for base in (HERE, root):
            p = os.path.join(base, f)
            if os.path.isfile(p):
                break
        else:
            bad.append("%s:%s (no such file)" % (f, ln))
            continue
        with io.open(p, encoding="utf-8", errors="replace") as fh:
            nl = sum(1 for _ in fh)
        if not 1 <= int(ln) <= nl:
            bad.append("%s:%s (file has %d lines)" % (f, ln, nl))
    check("every file:line reference resolves to a real line",
          enough and len(refs) >= n and bad == [],
          "%d refs, %d bad: %s" % (len(refs), len(bad), "; ".join(bad[:5])))

    # The retraction is HISTORY, kept in its own append-only file.  This checks
    # that the history exists and that the index points at it -- it does not
    # police the wording of either document.
    h = doc("VERDICT_HISTORY.md")
    check("the retraction is recorded in VERDICT_HISTORY.md and the index "
          "points at it",
          enough and "single_wave_sufficient" in h and "RETRACTED" in h
          and "VERDICT_HISTORY.md" in t,
          "history %d chars" % len(h))


def main():
    for fn in (test_formal_verdict_is_generated_from_one_record,
               test_three_category_legend,
               test_heldout_numbers_carry_the_heldout_tag,
               test_adequacy_numbers_are_labelled_train,
               test_exploratory_packet_section,
               test_aicc_parameter_count_corrected,
               test_percentile_not_probability,
               test_zero_of_500_upper_bound,
               test_pump_bin_convention_and_provenance,
               test_k0_share_reconciled,
               test_suite_count_reconciled,
               test_build_manifest_module_exists,
               test_build_manifest_write_and_check,
               test_manifest_honest_about_sim40,
               test_resume_refuses_on_dt_rec_change,
               test_resume_refuses_when_identity_absent,
               test_forward_provenance_is_in_the_run_manifest,
               test_resume_refuses_on_kernel_change,
               test_resume_accepts_a_matching_identity,
               test_provenance_block_records_the_build_link,
               test_handover_readme_corrected,
               test_corrections_index):
        try:
            fn()
        except Exception as exc:                              # noqa: BLE001
            check("%s raised" % fn.__name__, False, "%s: %s" % (type(exc).__name__, exc))

    print("\n" + "=" * 78)
    print("%d checks, %d FAILED" % (NCHECK, len(FAILS)))
    for f in FAILS:
        print("   FAIL  " + f)
    print("=" * 78)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
