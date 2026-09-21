"""Explicit gate states and run-identity manifests for the GPU campaign.

Added 2026-09-18 in response to sections 8 and 10 of
SAW_handover_verification_2026-09-18.md.  It carries no physics: it exists so
that "the check could not run" and "the check ran and passed" can never again be
the same return value, and so that a checkpoint can never be reused for a run it
does not belong to.

Three states a gate can be in, and one thing they have in common:
    PASS               the check ran and the criterion held
    FAIL               the check ran and the criterion did not hold
    NOT_DETERMINABLE   the check could not run (missing engine, missing input,
                       degenerate fit); NOT a pass and NOT a refutation
    NOT_EVALUABLE      a verdict was requested but a required input (a control,
                       a prerequisite stage) is missing, incomplete or from
                       different conditions
Only PASS lets a campaign continue.  Everything else BLOCKS, loudly, naming the
reason -- never a silent default.
"""

import hashlib
import json

import numpy as np

PASS = "PASS"
FAIL = "FAIL"
NOT_DETERMINABLE = "NOT_DETERMINABLE"
NOT_EVALUABLE = "NOT_EVALUABLE"
BLOCKING = (FAIL, NOT_DETERMINABLE, NOT_EVALUABLE)


class GateHalt(RuntimeError):
    """A gate that is not PASS was used as a prerequisite."""


class RunIdentityMismatch(GateHalt):
    """A checkpoint was offered for a run whose manifest differs from it."""


class Gate:
    """One named check with an explicit state and a human-readable reason."""

    def __init__(self, name, state, reason="", detail=None, sub=None):
        if state not in (PASS, FAIL, NOT_DETERMINABLE, NOT_EVALUABLE):
            raise ValueError("unknown gate state %r" % (state,))
        self.name = str(name)
        self.state = state
        self.reason = str(reason)
        self.detail = detail if detail is not None else {}
        self.sub = list(sub or [])

    # -- states -------------------------------------------------------------
    @property
    def ok(self):
        return self.state == PASS

    @property
    def blocks(self):
        return self.state != PASS

    def require(self):
        """Raise unless the gate PASSED.  This is what stops a campaign."""
        if self.blocks:
            raise GateHalt("%s: %s -- %s" % (self.name, self.state,
                                             self.reason or "no reason given"))
        return self

    # -- records ------------------------------------------------------------
    def as_dict(self):
        return dict(name=self.name, state=self.state, reason=self.reason,
                    detail=self.detail,
                    sub=[g.as_dict() for g in self.sub])

    def to_json(self):
        return json.dumps(self.as_dict(), default=str, sort_keys=True)

    def __repr__(self):
        return "Gate(%s, %s, %r)" % (self.name, self.state, self.reason)

    def __str__(self):
        return "%s %s%s" % (self.name, self.state,
                            (" (%s)" % self.reason) if self.reason else "")


def combine(name, *gates):
    """Worst state wins, and every blocking sub-reason is carried up.

    FAIL beats NOT_DETERMINABLE/NOT_EVALUABLE only in the ORDER of the reason
    text; both block.  A pass requires every sub-gate to pass.
    """
    gates = [g for g in gates if g is not None]
    bad = [g for g in gates if g.blocks]
    if not bad:
        return Gate(name, PASS,
                    "; ".join(g.reason for g in gates if g.reason), sub=gates)
    order = {FAIL: 0, NOT_EVALUABLE: 1, NOT_DETERMINABLE: 2}
    worst = sorted(bad, key=lambda g: order[g.state])[0]
    return Gate(name, worst.state,
                "; ".join("%s=%s (%s)" % (g.name, g.state, g.reason)
                          for g in bad), sub=gates)


def state_from_record(d, key_state="gate_state", key_bool="gate_pass"):
    """Read a gate state out of an npz.  A record that carries only the old
    boolean is downgraded to NOT_DETERMINABLE when the boolean is False, since
    the old files cannot distinguish FAIL from "could not run"."""
    if key_state in d:
        s = str(d[key_state])
        if s in (PASS, FAIL, NOT_DETERMINABLE, NOT_EVALUABLE):
            return s
        return NOT_DETERMINABLE
    if key_bool in d:
        return PASS if bool(d[key_bool]) else FAIL
    return NOT_DETERMINABLE


# ----------------------------------------------------------------------------
# run identity
# ----------------------------------------------------------------------------

def normalise(obj):
    """Canonical form of a manifest value.

    Every number becomes a float, so an int 1 and a float 1.0 cannot hash
    differently; numpy scalars and arrays become plain Python; dict keys become
    strings.  Booleans stay booleans (True is not 1.0 here).
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (str, bytes)):
        return obj.decode() if isinstance(obj, bytes) else obj
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): normalise(v) for k, v in sorted(obj.items(),
                                                        key=lambda kv: str(kv[0]))}
    if isinstance(obj, np.ndarray):
        return [normalise(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple, set)):
        return [normalise(v) for v in obj]
    return str(obj)


def canonical_json(manifest):
    return json.dumps(normalise(manifest), sort_keys=True,
                      separators=(",", ":"))


def manifest_hash(manifest):
    return hashlib.sha256(canonical_json(manifest).encode()).hexdigest()


def flatten(obj, prefix=""):
    """{'box.NX': 1400.0, ...} so a mismatch can name the field."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, "%s.%s" % (prefix, k) if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, "%s[%d]" % (prefix, i)))
    else:
        out[prefix] = obj
    return out


def diff(stored, requested, allow=()):
    """Field-by-field difference of two manifests.

    allow : top-level field names permitted to differ (e.g. 'eps0' between a
            pumped run and its pump-off control).  Any OTHER difference, and
            any field present in one manifest and missing from the other, is
            reported.
    Returns a sorted list of 'field: stored -> requested' strings.
    """
    a = flatten(normalise(stored or {}))
    b = flatten(normalise(requested or {}))
    allow = set(allow)
    msgs = []
    for key in sorted(set(a) | set(b)):
        top = key.split(".")[0].split("[")[0]
        if top in allow:
            continue
        if key not in a:
            msgs.append("%s: MISSING -> %r" % (key, b[key]))
        elif key not in b:
            msgs.append("%s: %r -> MISSING" % (key, a[key]))
        elif a[key] != b[key]:
            msgs.append("%s: %r -> %r" % (key, a[key], b[key]))
    return msgs
