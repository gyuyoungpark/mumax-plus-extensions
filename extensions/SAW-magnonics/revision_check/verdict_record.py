"""VERDICT.json -> the VERDICT section of model_comparison.md.

Why this file exists
--------------------
The formal verdict of the sim40 one-vs-two component comparison used to live as
prose in `model_comparison.md`, pinned by a fleet of whole-file substring greps.
That arrangement had two defects of the same class: prose could drift from the
recorded numbers without any check noticing, and an ADDED verdict paragraph was
invisible to every positive grep, so the retracted `single_wave_sufficient`
label could be reinstated with the suite fully green.

The mechanism replacing it:

  * `VERDICT.json` is the single source of truth.  It carries the operative
    verdict, the pre-registered tolerance, the four HELD-OUT values, the
    TRAIN-FIT adequacy triple and the frozen physical conclusions.  Every
    number names the evidence FILE and KEY that produces it.
  * this module GENERATES the VERDICT section of `model_comparison.md` from
    that record, between two markers named in the record itself.
  * the regression check is therefore ONE assertion -- "the generated section
    is byte-identical to what the document carries" -- plus one structural
    assertion, "the document declares exactly one VERDICT section and it is the
    generated one".  Neither can be satisfied by adding prose.
  * `--check` additionally re-reads every recorded number from its evidence
    file, so editing VERDICT.json itself does not silently move the record.

Past claims and their retraction are HISTORY, not record; they live in
`VERDICT_HISTORY.md` and nothing here polices that file.

Usage
-----
    python verdict_record.py --check     # verify record vs evidence vs document
    python verdict_record.py --render    # print the generated section
    python verdict_record.py --write     # splice the generated section in place
"""

import argparse
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = "VERDICT.json"

# A heading that DECLARES a verdict.  Used for the structural uniqueness check:
# the document must carry exactly one, and it must be the generated one.
VERDICT_HEADING_RE = re.compile(r"(?m)^#{1,6}\s*VERDICT\b.*$")


# ---------------------------------------------------------------- record I/O
def load(root=HERE):
    with io.open(os.path.join(root, RECORD), encoding="utf-8") as fh:
        return json.load(fh)


def _decimals(s):
    s = str(s)
    return len(s.split(".", 1)[1]) if "." in s else 0


def _dig(obj, key):
    """Walk a '/'-separated key path through nested dicts/lists."""
    cur = obj
    for part in key.split("/"):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


def evidence_value(ev, root=HERE, **fmt):
    """Read the value an evidence pointer names.  Raises on a bad pointer.

    For kind == "literal" the string looked for is BUILT from the recorded
    fields via ev["template"], never stored separately -- otherwise the recorded
    value and the string that verifies it could disagree.
    """
    path = os.path.join(root, ev["file"])
    kind = ev.get("kind", "number")
    if kind == "literal":
        want = ev.get("template", "{value}").format(**fmt)
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            return want if want in fh.read() else None
    with io.open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    return _dig(blob, ev["key"])


def verify_evidence(rec=None, root=HERE):
    """Re-read every recorded value from its evidence file.

    Returns a list of (name, ok, detail).  This is what stops VERDICT.json
    from becoming a second place a number can be wrong: the record states the
    value AND where it comes from, and the two are compared here.
    """
    rec = rec or load(root)
    out = []

    def num(name, recorded, ev):
        try:
            got = float(evidence_value(ev, root))
        except Exception as exc:                                  # noqa: BLE001
            out.append((name, False, "%s: %s" % (type(exc).__name__, exc)))
            return
        tol = 0.5 * 10.0 ** (-_decimals(recorded))
        ok = abs(got - float(recorded)) <= tol
        out.append((name, ok, "%s -> %s = %.10g (tol %g)"
                    % (ev["file"], ev["key"], got, tol)))

    v = rec["verdict"]
    for label, ev in (("preregistered", v["operative_evidence"]),
                      ("amended", v["operative_evidence_amended"])):
        try:
            got = evidence_value(ev, root)
        except Exception as exc:                                  # noqa: BLE001
            out.append(("operative verdict (%s) reads back from evidence" % label,
                        False, str(exc)))
            continue
        out.append(("operative verdict (%s) reads back from evidence" % label,
                    got == v["operative"],
                    "%s -> %s = %r" % (ev["file"], ev["key"], got)))

    # the retracted label must be absent from the evidence, not merely
    # annotated as retracted in prose.
    for r in v["retracted_labels"]:
        path = os.path.join(root, r["never_returned_by"])
        with io.open(path, encoding="utf-8", errors="replace") as fh:
            n = fh.read().count(r["label"])
        out.append(("retracted label %r appears %d times in %s"
                    % (r["label"], r["count_in_evidence"], r["never_returned_by"]),
                    n == r["count_in_evidence"], "counted %d" % n))
        out.append(("the operative verdict is not the retracted label %r" % r["label"],
                    v["operative"] != r["label"] and v["sub_label"] != r["label"],
                    "operative=%s sub=%s" % (v["operative"], v["sub_label"])))

    t = rec["tolerance"]
    want = t["evidence"].get("template", "{value}").format(name=t["name"],
                                                           value=t["value"])
    out.append(("%s = %s is in %s" % (t["name"], t["value"], t["evidence"]["file"]),
                evidence_value(t["evidence"], root,
                               name=t["name"], value=t["value"]) is not None,
                "looked for %r" % want))

    for h in rec["held_out"]:
        num("HELD-OUT %s (%s) = %s" % (h["symbol"], h["config"], h["value"]),
            h["value"], h["evidence"])

    a = rec["adequacy_train_fit"]
    for c in a["components"]:
        num("TRAIN-FIT adequacy N=%d = %s" % (c["n"], c["value"]),
            c["value"], c["evidence"])
    num("n_complex = %d" % a["n_complex"], str(a["n_complex"]), a["n_complex_evidence"])

    # the sub-label is a DERIVED statement: both held-out errors exceed tau.
    tau = float(t["value"])
    worst = min(float(h["value"]) for h in rec["held_out"])
    out.append(("sub-label %r follows from the held-out values against %s"
                % (rec["verdict"]["sub_label"], t["name"]),
                worst > tau,
                "smallest HELD-OUT error %.6f > tau_adeq %.2f" % (worst, tau)))
    return out


# ------------------------------------------------------------------- render
def _q(lines):
    return ["> " + ln if ln else ">" for ln in lines]


def render(rec=None, root=HERE):
    """The VERDICT section, generated.  Canonical newline is '\\n'."""
    rec = rec or load(root)
    r = rec["render"]
    v = rec["verdict"]
    t = rec["tolerance"]
    a = rec["adequacy_train_fit"]
    dec = a["display_decimals"]
    subs = {
        "operative": v["operative"],
        "sub_label": v["sub_label"],
        "retracted": v["retracted_labels"][0]["label"],
        "tau_name": t["name"],
        "tau": t["value"],
    }
    for c in a["components"]:
        subs["adeq%d" % c["n"]] = ("%." + str(dec) + "f") % float(c["value"])

    def sub(lines):
        return [ln.format(**subs) for ln in lines]

    L = [r["heading"], ""]
    body = list(_q(sub(r["lead"])))

    # --- the verdict table, built from the record, not typed
    configs = []
    for h in rec["held_out"]:
        if h["config"] not in configs:
            configs.append(h["config"])
    returned = "`%s / %s`" % (v["operative"], v["sub_label"])
    body += [">",
             "> | config | E_M1 (**HELD-OUT**) | E_M2 (**HELD-OUT**) | `%s` | returned |"
             % t["name"],
             "> |---|---:|---:|---:|---|"]
    for cfg in configs:
        cells = {h["symbol"]: h["value"] for h in rec["held_out"] if h["config"] == cfg}
        body.append("> | %s, **HELD-OUT** | **%s** | **%s** | %s | %s |"
                    % (cfg, cells["E_M1"], cells["E_M2"], t["value"], returned))
    body += [">"] + _q(sub(r["after_table"]))
    for para in r["paragraphs"]:
        body += [">"] + _q(sub(para))

    # --- frozen physical conclusions, verbatim from the record
    body += [">"] + _q(sub(r["frozen_intro"]))
    body += [">"] + _q(["* " + c for c in rec["frozen_physical_conclusions"]])

    L += body + ["", r["established_intro"], ""]
    for bullet in r["established"]:
        L.append("* " + bullet[0])
        L += ["  " + ln for ln in bullet[1:]]
    L += ["", "---", ""]
    return "\n".join(L) + "\n"


# ------------------------------------------------------- document splice/check
def _read(path):
    with io.open(path, "rb") as fh:
        return fh.read()


def _doc(rec, root=HERE):
    return os.path.join(root, rec["target"]["file"])


def extract(rec=None, root=HERE):
    """The section the document currently carries between the markers.

    Returns (text_or_None, reason).  text is newline-normalised to '\\n'.
    """
    rec = rec or load(root)
    tg = rec["target"]
    blob = _read(_doc(rec, root)).decode("utf-8", "replace").replace("\r\n", "\n")
    b, e = tg["begin_marker"], tg["end_marker"]
    nb, ne = blob.count(b), blob.count(e)
    if nb != 1 or ne != 1:
        return None, "marker count begin=%d end=%d (each must be 1)" % (nb, ne)
    i = blob.index(b) + len(b)
    j = blob.index(e)
    if j < i:
        return None, "END marker precedes BEGIN marker"
    return blob[i:j].lstrip("\n"), "ok"


def check_section(rec=None, root=HERE):
    """(ok, detail) -- the one assertion that replaces the substring greps."""
    rec = rec or load(root)
    got, why = extract(rec, root)
    if got is None:
        return False, why
    want = render(rec, root)
    if got.rstrip("\n") == want.rstrip("\n"):
        return True, "%d chars match" % len(want.rstrip("\n"))
    gl, wl = got.rstrip("\n").split("\n"), want.rstrip("\n").split("\n")
    for n, (x, y) in enumerate(zip(gl, wl), 1):
        if x != y:
            return False, ("first difference at generated line %d:\n"
                           "   doc: %r\n   gen: %r" % (n, x[:110], y[:110]))
    return False, ("document has %d lines, generator %d" % (len(gl), len(wl)))


def check_unique_heading(rec=None, root=HERE):
    """The document must declare exactly ONE verdict section, inside the markers.

    This is the structural half of the fix.  A superseding verdict block added
    ANYWHERE in the document -- before the tolerances section, at the end of
    the file, inside an appendix -- adds a second VERDICT heading and fails
    here, whether or not it lands inside the generated region.
    """
    rec = rec or load(root)
    tg = rec["target"]
    blob = _read(_doc(rec, root)).decode("utf-8", "replace").replace("\r\n", "\n")
    heads = [(m.start(), m.group(0).strip())
             for m in VERDICT_HEADING_RE.finditer(blob)]
    if len(heads) != 1:
        return False, ("%d verdict headings: %s"
                       % (len(heads), "; ".join(h[1][:60] for h in heads)))
    if tg["begin_marker"] not in blob or tg["end_marker"] not in blob:
        return False, "markers absent"
    i = blob.index(tg["begin_marker"])
    j = blob.index(tg["end_marker"])
    pos = heads[0][0]
    if not i < pos < j:
        return False, "the verdict heading is outside the generated region"
    return True, heads[0][1]


def write(root=HERE):
    """Splice the generated section into the document, preserving its EOL."""
    rec = load(root)
    tg = rec["target"]
    path = _doc(rec, root)
    raw = _read(path)
    crlf = b"\r\n" in raw
    blob = raw.decode("utf-8").replace("\r\n", "\n")
    b, e = tg["begin_marker"], tg["end_marker"]
    section = render(rec, root)
    if blob.count(b) == 1 and blob.count(e) == 1:
        i, j = blob.index(b), blob.index(e) + len(e)
        new = blob[:i] + b + "\n" + section + e + blob[j:]
    else:
        # first installation: replace the hand-written section in place
        m = VERDICT_HEADING_RE.search(blob)
        if not m:
            raise SystemExit("no VERDICT heading to replace and no markers")
        i = m.start()
        nxt = re.compile(r"(?m)^##\s+\d").search(blob, m.end())
        if not nxt:
            raise SystemExit("cannot find the section that follows VERDICT")
        j = nxt.start()
        new = blob[:i] + b + "\n" + section + e + "\n" + blob[j:]
    out = new.replace("\n", "\r\n") if crlf else new
    with io.open(path, "wb") as fh:
        fh.write(out.encode("utf-8"))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--write", action="store_true")
    ns = ap.parse_args(argv)
    if ns.render:
        try:                          # the section contains an em dash
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:                                         # noqa: BLE001
            pass
        sys.stdout.write(render())
        return 0
    if ns.write:
        print("wrote", write())
        return 0
    bad = 0
    for name, ok, detail in verify_evidence():
        print(("  ok    " if ok else "  BAD   ") + name + "   " + detail)
        bad += 0 if ok else 1
    for name, fn in (("generated section matches the document", check_section),
                     ("exactly one verdict section, and it is generated",
                      check_unique_heading)):
        ok, detail = fn()
        print(("  ok    " if ok else "  BAD   ") + name + "   " + detail)
        bad += 0 if ok else 1
    print("%s" % ("VERDICT RECORD CONSISTENT" if not bad
                  else "%d PROBLEM(S)" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
