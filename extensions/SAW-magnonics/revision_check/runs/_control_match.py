"""ONE comparison for "is this record a control OF that record", and nothing else.

WHY (audit 2026-09-18, item 3 / control completeness).
G9's control gate enforced each control's CONTENT with a bespoke if-statement:
"eps0 must not be non-zero" and "the spectator's bin must differ from
inj_res's".  Every condition that nobody had written an if for was therefore
unchecked, which is how a spectator injected at bin 2 -- adjacent to the
resonant bin 3, not at the largest pair mismatch REQUIRED_CONTROLS specifies --
passed as the null and the verdict still read DETECTED.

The class fix is that WHAT a control must satisfy is declared once in
G9_CRITERIA.json and evaluated by a generic loop, and that the field-by-field
comparison below is the ONLY comparison used, sharing its normalisation with the
checkpoint/certificate machinery (_gate.normalise, _gate.flatten, _gate.diff).
There is no second notion of "the same conditions" inside G9.

HANDOFF, stated so the next round does not duplicate this.  runs/_conditions.py
(priority-1, same round) is building the campaign-wide precondition mechanism:
stamped artifacts, a declared FIELD list, and compare()/require() over a named
`need`.  Once G9's checkpoints carry that stamp, `match()` below is the single
line that re-points at _conditions.compare(); the control declarations in
G9_CRITERIA.json and the generic loop in G9._control_gate do not change.  Until
the stamp is on G9 records, this compares the recorded run manifest with the
shared primitive rather than inventing a parallel field list.
"""

import math

import numpy as np

import _gate


def _both_nan(msg):
    """True for a _gate.diff message whose two sides are both NaN.

    NaN != NaN, so a field recorded as "not measured" in BOTH records is
    reported by _gate.diff as a difference.  Two records that both say "this
    quantity was not measured" are in the same condition, and it is the
    SUFFICIENCY checks (G9_CRITERIA.json -> must_satisfy) that refuse to certify
    a control on an unmeasured quantity, naming that reason instead of burying
    it in a field diff.  Folding NaN-vs-NaN out here therefore removes noise, not
    a check: nothing becomes unchecked, the refusal simply arrives with its
    reason attached.
    """
    if ": " not in msg:
        return False
    tail = msg.split(": ", 1)[1]
    if " -> " not in tail:
        return False
    a, b = tail.rsplit(" -> ", 1)
    try:
        return math.isnan(float(a)) and math.isnan(float(b))
    except (TypeError, ValueError):
        return False


def match(base, other, allow=()):
    """Fields in which `other` is not the same run condition as `base`.

    `allow` names top-level manifest fields a legitimate control may differ in.
    Returns a sorted list of 'field: base -> other' strings; empty means matched.
    """
    return [m for m in _gate.diff(base, other, allow=allow) if not _both_nan(m)]


def get_path(manifest, path, default=None):
    """manifest['a']['b'] addressed as 'a.b'; `default` when any step is absent."""
    cur = manifest
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def is_unmeasured(value):
    """True for a quantity recorded as 'not measured' (None or NaN)."""
    if value is None:
        return True
    try:
        return bool(np.isnan(float(value)))
    except (TypeError, ValueError):
        return False
