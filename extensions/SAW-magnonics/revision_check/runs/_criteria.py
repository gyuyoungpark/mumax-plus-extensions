"""G9's declared rule, conditional error budget and generated documentation.

Delta <= K_SIGMA * hypot(sigma_stat, sigma_sys) + tol_phys.
Statistical errors are approximate overlap-adjusted OLS errors. The systematic
term is a conditional synthetic calibration, not an identified variance or a
physical false-positive rate. tol_phys is zero for a common eigenmode: unequal
free damping rates do not bound transient slopes of a superposition.
"""
import hashlib
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.dirname(HERE)
SPEC_PATH = os.path.join(RC, "G9_CRITERIA.json")
PREREG_PATH = os.path.join(RC, "PREREGISTRATION.md")
BEGIN = "<!-- BEGIN GENERATED FROM G9_CRITERIA.json -- do not edit by hand -->"
END = "<!-- END GENERATED FROM G9_CRITERIA.json -->"
NOT_DETERMINABLE = "NOT_DETERMINABLE"


class SpecError(RuntimeError):
    """Missing, malformed or unsupported decision criteria."""


def load(path=None):
    try:
        with open(path or SPEC_PATH, encoding="utf-8") as fh:
            spec = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SpecError("cannot load G9 criteria: %s" % exc) from exc
    if not isinstance(spec, dict) or spec.get("schema") != "saw_revision_check/g9_criteria/1":
        raise SpecError("unsupported G9 criteria schema")
    return spec


def _need(d, key, where):
    if not isinstance(d, dict) or key not in d:
        raise SpecError("%s is missing %s; no default is supplied" % (where, key))
    return d[key]


def get(*path, spec=None):
    cur = spec if spec is not None else load()
    for key in path:
        cur = _need(cur, key, "G9_CRITERIA.json")
    return cur


def estimator_fingerprint():
    """Invalidate calibration when the production estimator or rule changes."""
    digest = hashlib.sha256()
    for rel in ("runs/G9_idler_injection.py", "runs/_criteria.py",
                "growth_interval.py", "saw_analysis.py"):
        with open(os.path.join(RC, rel), encoding="utf-8") as fh:
            digest.update(rel.encode("ascii"))
            digest.update(fh.read().encode("utf-8"))
    return digest.hexdigest()


class RateComparison(dict):
    @property
    def compatible(self):
        return self["compatible"]


def compare_growth_rates(g1, g2, se1, se2, f1_Hz, f2_Hz, alpha,
                         seg_len=None, stride=None, spec=None, fit_valid=None):
    """Conditional compatibility; fit_valid comes from production diagnostics."""
    g = get("growth_rate_compatibility", spec=spec)
    try:
        K = float(_need(g, "K_SIGMA", "growth_rate_compatibility"))
        floor = _need(g, "systematic_relative_floor", "growth_rate_compatibility")
        f_sys = float(_need(floor, "value", "systematic_relative_floor"))
    except (TypeError, ValueError) as exc:
        raise SpecError("systematic floor must be MEASURED; invalid decision constant") from exc
    if not np.isfinite(K) or K <= 0 or not np.isfinite(f_sys) or f_sys < 0:
        raise SpecError("K_SIGMA must be positive and f_sys finite and nonnegative")
    phys = _need(g, "physical_tolerance", "growth_rate_compatibility")
    if phys.get("value") != 0.0 or phys.get("formula") != "0":
        raise SpecError("only tol_phys = 0 for common-eigenmode equality is justified")
    nan = float("nan")
    out = RateComparison(
        Gamma_1=nan, Gamma_2=nan, stderr_1=nan, stderr_2=nan, delta=nan,
        K_SIGMA=K, systematic_relative_floor=f_sys, sigma_stat=nan,
        sigma_sys=nan, tol_phys=0.0, allowance=nan, overlap_inflation=nan,
        relative_difference=nan, relative_allowance=nan,
        compatible=NOT_DETERMINABLE, conditional=True,
        uncertainty_scope=g["interpretation"], physical_scope=phys["scope"],
        seg_len=seg_len, stride=stride)

    def refuse(reason):
        out["reason"] = reason
        return out

    try:
        vals = np.asarray([g1, g2, se1, se2, f1_Hz, f2_Hz, alpha], dtype=float)
        if vals.shape != (7,) or not np.isfinite(vals).all():
            return refuse("non-finite rate, standard error, frequency or damping")
        g1, g2, se1, se2, f1_Hz, f2_Hz, alpha = vals.tolist()
        out.update(Gamma_1=g1, Gamma_2=g2, stderr_1=se1, stderr_2=se2,
                   delta=abs(g1-g2), f1_Hz=f1_Hz, f2_Hz=f2_Hz, alpha=alpha)
        if se1 < 0 or se2 < 0 or f1_Hz <= 0 or f2_Hz <= 0 or alpha < 0:
            return refuse("negative standard error/damping or nonpositive frequency")
        if (seg_len is None or stride is None or isinstance(seg_len, bool)
                or isinstance(stride, bool) or int(seg_len) != seg_len
                or int(stride) != stride or not 0 < stride <= seg_len):
            return refuse("missing or invalid Welch geometry")
    except (TypeError, ValueError, OverflowError):
        return refuse("malformed numeric comparison input or Welch geometry")
    if fit_valid is not True:
        return refuse("valid, non-transient GI fits on the same early interval are required")
    if floor.get("estimator_sha256") != estimator_fingerprint():
        return refuse("systematic calibration is absent or stale for this estimator")
    if floor.get("calibrated_K_SIGMA") != K:
        return refuse("systematic calibration used a different K_SIGMA")
    inflate = float(np.sqrt(seg_len / stride))
    stat = inflate * float(np.hypot(se1, se2))
    sys_err = f_sys * max(abs(g1), abs(g2))
    allowance = K * float(np.hypot(stat, sys_err))
    scale = max(abs(g1), abs(g2))
    out.update(overlap_inflation=inflate, sigma_stat=stat, sigma_sys=sys_err,
               allowance=allowance, compatible=bool(out["delta"] <= allowance),
               relative_difference=out["delta"] / scale if scale else nan,
               relative_allowance=allowance / scale if scale else nan,
               dominant_term="statistical" if stat >= sys_err else "systematic",
               reason="|dGamma| = %.4e %s allowance %.4e; tol_phys = 0; conditional on %s"
               % (out["delta"], "<=" if out["delta"] <= allowance else ">",
                  allowance, phys["scope"]))
    return out


def render_preregistration_section(spec=None):
    spec = spec if spec is not None else load()
    g = get("growth_rate_compatibility", spec=spec)
    floor, phys = g["systematic_relative_floor"], g["physical_tolerance"]
    L = [BEGIN, "", "## " + spec["document_section_title"], "",
         "Generated from G9_CRITERIA.json, version %s." % spec["version"],
         "Regenerate with `python runs/_criteria.py render`; verify with `python runs/_criteria.py check`.",
         "", "### G9.1 Growth-rate compatibility (P3)", "",
         "    " + g["rule"], "", g["interpretation"], "",
         "| term | value / formula | basis |", "|---|---|---|",
         "| `K_SIGMA` | `%s` | %s |" % (g["K_SIGMA"], g["K_SIGMA_justification"]),
         "| `sigma_stat` | `%s` | %s |" % (g["statistical_term"]["formula"], g["statistical_term"]["justification"]),
         "| `sigma_sys` | `%s`; f_sys = `%s` | %s |" % (g["systematic_term"]["formula"], floor["value"], floor["justification"]),
         "| `tol_phys` | `%s` | %s |" % (phys["formula"], phys["justification"]),
         "", "Physical scope: " + phys["scope"], "", g["indeterminate_policy"],
         "Both rates must be positive. No relative-error OR clause is allowed.",
         "", "Calibration: " + floor["how_measured"],
         "Evidence: %s; n = %s; measured UTC = %s."
         % (floor.get("evidence_file"), floor["n_controls"], floor["measured_utc"]),
         "", "Fit applicability: " + g["fit_applicability"]["statement"],
         "Linearity limit: %s." % g["fit_applicability"]["m_linear"],
         "Growth layout: %s samples. %s" % (g["fit_applicability"]["growth_seg_len"],
                                           g["fit_applicability"]["layout_scope"]),
         "", "### G9.2 Other predicates", "", "| symbol | value | meaning |", "|---|---|---|"]
    for k, v in sorted(spec["predicates"].items()):
        L.append("| `%s` | `%s` | %s |" % (k, v["value"], v["meaning"]))
    L += ["", "### G9.3 Prerequisites", "", spec["record_policy"], "",
          "Condition matching uses _conditions.compare on stamps and _control_match.match on declarations.",
          "Selection: " + spec["selection_policy"]["statement"]]
    for lbl, c in sorted(spec["required_controls"].items()):
        L += ["", "**%s**: %s" % (lbl, c["role"]),
              "Allowed declaration differences: %s." % ", ".join(c["allow_diff"]),
              "Allowed stamped differences: %s." % ", ".join(c["allow_condition_diff"])]
        L += ["- %s: %s" % (r["statement"], r["why"]) for r in c["must_satisfy"]]
    L += ["", "### G9.4 Coherence surrogate", "", spec["null"]["statement"],
          spec["null"]["rate_policy"], "P4 uses %s." % spec["null"]["p4_threshold_variant"]]
    for name in spec["null"]["variants_order"]:
        v = spec["null"]["variants"][name]
        L += ["", "**%s**: %s. %s" % (name, v["construction"], v["quotable_as"])]
    return "\n".join(L + ["", END, ""])


def write_preregistration_section(path=None):
    p = path or PREREG_PATH
    body = render_preregistration_section()
    with open(p, encoding="utf-8") as fh:
        doc = fh.read()
    if BEGIN in doc and END in doc:
        i, j = doc.index(BEGIN), doc.index(END) + len(END)
        new = doc[:i] + body.rstrip("\n") + doc[j:]
    else:
        new = doc.rstrip("\n") + "\n\n" + body
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new)
    return p


def preregistration_matches(path=None):
    try:
        with open(path or PREREG_PATH, encoding="utf-8") as fh:
            doc = fh.read()
    except OSError as exc:
        return False, str(exc)
    if BEGIN not in doc or END not in doc:
        return False, "generated G9 section missing"
    got = doc[doc.index(BEGIN):doc.index(END) + len(END)]
    ok = got.strip() == render_preregistration_section().strip()
    return ok, "generated G9 section matches" if ok else "regenerate G9 section from G9_CRITERIA.json"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "show":
        print(json.dumps(load(), indent=2))
    elif cmd == "render":
        print(write_preregistration_section())
    elif cmd == "check":
        ok, why = preregistration_matches()
        print(("OK: " if ok else "DRIFT: ") + why)
        sys.exit(0 if ok else 1)
    else:
        raise SystemExit("usage: _criteria.py [show|render|check]")
