"""ONE precondition mechanism: condition sets, artifact stamps, stage needs.

Added 2026-09-18 (correction round 2) to replace three ad-hoc checks that each
failed the same way:

  * gate certificates were accepted on their recorded STATE alone, so a PASS
    pump certificate measured in the sim40 box (NX = 1024, pump on bin 9)
    authorised the NX = 1400 / bin 12 campaign;
  * checkpoints were accepted on a hash of whatever the call site volunteered,
    so XI / NY / NZ / CY / CZ / enable_mel / saw_direction could all change
    without moving it, and `manifest=None` volunteered nothing at all;
  * the build link was recorded as a field nobody read.

The common defect is not any of those three.  It is that each check invented its
own notion of "the same run", so a condition that one check happened not to name
was unchecked everywhere.  This module removes the class by making there be
exactly ONE notion:

    CONDITION SET   a normalised, hashable description of a run, whose FIELD
                    LIST lives in FIELDS below and nowhere else.  Grid, pump,
                    material, numerics, build.  A field the caller did not
                    declare is recorded as the explicit sentinel UNDECLARED and
                    still takes part in the digest -- silence is data, not a
                    default.

    STAMP           every artifact a stage writes (certificate, checkpoint,
                    summary) embeds the condition set it was produced under
                    AND the subset it actually verified.  `stamp()` is the only
                    writer; `read()` is the only reader.

    NEED            every stage declares, by name, the subset of fields it
                    requires of a prerequisite artifact.  `require()` is the
                    only comparison: expand(need) -> compare stored vs current,
                    field by field.  There is no positional, index-based or
                    state-only check, and a need naming a field outside FIELDS
                    raises instead of quietly checking nothing.

Refusals are loud and name the fields.  `require()` raises _gate.GateHalt;
`state()` is the same comparison returning a Gate so a caller can write a
NOT_EVALUABLE summary artifact instead of dying with a KeyError.

What this does NOT do, stated here so no reader infers more.  It is FORWARD
provenance only: it can refuse a mismatched artifact from now on, and it cannot
recover the conditions of an artifact written before it existed -- including the
published sim40 data, whose source-to-binary link stays unrecoverable
(../PROVENANCE_BUILD_MANIFEST.md).  An artifact carrying no stamp is therefore
refused, not trusted.
"""

import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gate                                                      # noqa: E402
import _provenance as _prov                                       # noqa: E402

SCHEMA = "saw_revision_check/conditions/1"

# The sentinel for "the caller did not declare this".  It is a value, it is
# hashed, and every gate that NEEDS such a field refuses it.
UNDECLARED = "UNDECLARED"

# npz keys.  Both names carry the SAME bytes; `run_identity` is retained only
# because records and tests already use that name.  One writer, one reader.
KEY = "conditions"
KEY_ALIAS = "run_identity"
KEY_DIGEST = "conditions_digest"


class PreconditionHalt(_gate.GateHalt):
    """A prerequisite artifact does not cover the conditions of this run."""


# ===========================================================================
# helpers used by the field resolvers
# ===========================================================================
def _get(d, path):
    """d['a']['b'] for path 'a.b', or None if any step is missing."""
    cur = d
    for part in str(path).split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _box(decl, env):
    """The box dict this run integrates on: declared inline, or by tag."""
    b = decl.get("box")
    if isinstance(b, dict) and b:
        return b
    tag = _first(decl.get("box_tag"), _get(decl, "box.tag"))
    if tag is not None:
        for cand in (env.get("boxes") or ()):
            if str(cand.get("tag")) == str(tag):
                return cand
    return {}


def _mat(decl, env):
    m = decl.get("material")
    if isinstance(m, dict) and m:
        return m
    tag = _first(decl.get("material_tag"), _get(decl, "material.tag"))
    if tag is not None:
        for cand in (env.get("materials") or ()):
            if str(cand.get("tag")) == str(tag):
                return cand
    return {}


def _bx(key):
    return lambda d, e: _box(d, e).get(key)


def _mt(key):
    return lambda d, e: _mat(d, e).get(key)


def _ev(key):
    return lambda d, e: e.get(key)


def _dc(*paths, **kw):
    """Declared value, first of `paths`; env key `env` as the documented default
    (build()'s own default, so the stamp describes what build() will do)."""
    env_key = kw.get("env")

    def resolve(d, e):
        for p in paths:
            v = _get(d, p)
            if v is not None:
                return v
        return e.get(env_key) if env_key else None
    return resolve


def _seed_k_cut_default(decl, env):
    """The campaign's seed band limit, 2q.

    Every stage passes it explicitly to make_seed and records it in its seed
    meta (G4 K_CUT = 2 Q, G4b 2 bx['q'], G6 and G9 2 Q), so this is the
    documented convention rather than an invented value; it applies only when a
    declaration names a box and a seed amplitude but no seed meta.
    """
    q = _q_of(decl, env)
    return None if q is None else 2.0 * float(q)


def _q_of(decl, env):
    """The pump wavevector.  Declared q if there is one, else 2 pi f / v."""
    q = _first(_get(decl, "q"), _get(decl, "box.q"), _box(decl, env).get("q"))
    if q is not None:
        return q
    f = _first(_get(decl, "f_saw"), _box(decl, env).get("f_saw"))
    v = _first(_get(decl, "v_saw"), _box(decl, env).get("v_saw"),
               env.get("V_SAW"))
    if f is None or v is None:
        return None
    return 2.0 * 3.141592653589793 * float(f) / float(v)


# ===========================================================================
# THE FIELD LIST.  One place.  Adding a parameter means adding a line HERE,
# and every stamp, digest, diff and stage need picks it up automatically.
# ===========================================================================
class Field(object):
    __slots__ = ("name", "group", "doc", "resolve")

    def __init__(self, group, name, doc, resolve):
        self.group, self.name, self.doc, self.resolve = group, name, doc, resolve

    @property
    def path(self):
        return "%s.%s" % (self.group, self.name)


FIELDS = (
    # ---- grid: the discretised volume the LLG is integrated on -------------
    Field("grid", "NX", "cells along x (box length / CX)", _bx("NX")),
    Field("grid", "NY", "cells along y", _ev("NY")),
    Field("grid", "NZ", "cells along z", _ev("NZ")),
    Field("grid", "CX", "cell size along x", lambda d, e: _first(
        _box(d, e).get("dx"), _box(d, e).get("CX"))),
    Field("grid", "CY", "cell size along y", _ev("CY")),
    Field("grid", "CZ", "cell size along z", _ev("CZ")),
    Field("grid", "PBC", "world pbc_repetitions", _ev("PBC")),
    # ---- pump: the SAW drive ----------------------------------------------
    Field("pump", "f_saw", "SAW frequency", lambda d, e: _first(
        _get(d, "f_saw"), _box(d, e).get("f_saw"))),
    Field("pump", "q", "SAW wavevector", _q_of),
    Field("pump", "eps_0", "strain amplitude (0 = SAW off)",
          _dc("eps0", "eps_0")),
    Field("pump", "xi", "Rayleigh ellipticity", _dc("xi", env="XI")),
    Field("pump", "phase", "SAW phase at t=0",
          _dc("saw_phase", "phase", env="SAW_PHASE")),
    Field("pump", "direction", "propagation sign along x",
          _dc("saw_direction", env="SAW_DIRECTION")),
    Field("pump", "enable_mel", "magnetoelastic channel on/off",
          _dc("enable_mel", env="ENABLE_MEL")),
    Field("pump", "enable_barnett", "Barnett channel on/off",
          _dc("enable_barnett", env="ENABLE_BARNETT")),
    # ---- material ---------------------------------------------------------
    Field("material", "Msat", "saturation magnetisation", _mt("MS")),
    Field("material", "Aex", "exchange stiffness", _mt("AEX")),
    Field("material", "alpha", "Gilbert damping", _mt("ALPHA")),
    Field("material", "B1", "magnetoelastic coupling", _mt("B1")),
    Field("material", "Kmr", "magnetorotation coupling", _mt("KMR")),
    Field("material", "B0", "bias field along x", _dc("B0")),
    Field("material", "temperature", "bath temperature", _dc("temperature")),
    # ---- numerics ---------------------------------------------------------
    Field("numerics", "dt_rec", "record cadence", _dc("dt_rec", "DT_REC")),
    Field("numerics", "dt_step", "fixed solver timestep",
          _dc("dt_step", "DT_STEP", env="DT_STEP")),
    Field("numerics", "nt", "number of records", _dc("nt")),
    Field("numerics", "block_records", "checkpoint block size",
          _dc("block_records")),
    Field("numerics", "seed_kind", "seed construction",
          _dc("seed.seed_kind", "seed_kind", env="SEED_KIND")),
    Field("numerics", "seed_amplitude", "per-mode seed amplitude",
          _dc("a_seed", "seed.a_mode")),
    Field("numerics", "seed_k_cut", "seed band limit",
          lambda d, e: _first(_get(d, "k_cut"), _get(d, "seed.k_cut"),
                              _seed_k_cut_default(d, e))),
    Field("numerics", "rng_seed", "explicit integer rng seed",
          _dc("rng_seed", "rng_seed_numpy", "seed.rng_seed")),
    # ---- build: which binary integrated it --------------------------------
    Field("build", "source_tree_digest", "sha256 over all recorded sources",
          _ev("source_tree_digest")),
    Field("build", "compiled_source_digest", "sha256 over compiled sources only",
          _ev("compiled_source_digest")),
    Field("build", "build_config_digest", "sha256 over the checked CMake config",
          _ev("build_config_digest")),
    Field("build", "engine_binary_sha256", "the .pyd python imports",
          _ev("engine_binary_sha256")),
    Field("build", "fp_precision", "FP_PRECISION of that build",
          _ev("fp_precision")),
    Field("build", "build_link_status", "source->binary link verdict",
          _ev("build_link_status")),
)

GROUPS = tuple(dict.fromkeys(f.group for f in FIELDS))
BY_PATH = {f.path: f for f in FIELDS}
ALL_FIELDS = tuple(f.path for f in FIELDS)


def expand(need):
    """Field paths a need names.  A group name expands to its whole group.

    An unknown name RAISES.  This is the anti-vacuity rule: a typo in a stage's
    need list must not silently check nothing.
    """
    out = []
    for name in ((need,) if isinstance(need, str) else tuple(need)):
        name = str(name)
        if name in BY_PATH:
            out.append(name)
        elif name in GROUPS:
            out.extend(f.path for f in FIELDS if f.group == name)
        else:
            raise KeyError(
                "unknown condition %r: it is neither a field %s nor a group %s"
                % (name, sorted(BY_PATH)[:3] + ["..."], list(GROUPS)))
    return tuple(dict.fromkeys(out))


# ===========================================================================
# the condition set
# ===========================================================================
def _canon(obj):
    return json.dumps(_gate.normalise(obj), sort_keys=True,
                      separators=(",", ":"))


def _source_map():
    try:
        return dict(_prov.link_summary()["compiled_source_map"])
    except Exception as exc:                                       # noqa: BLE001
        return {"NOT DETERMINABLE": str(exc)}


def _named_source_diffs(body, limit=8):
    """Which compiled sources differ between a stamp and the tree as it is now."""
    stored = body.get("source_map")
    if not isinstance(stored, dict) or not stored:
        return ["(the artifact records no per-file source map, so the changed "
                "file cannot be named)"]
    cur = _source_map()
    names = sorted(set(stored) | set(cur))
    out = []
    for n in names:
        a, b = stored.get(n), cur.get(n)
        if a != b:
            out.append("%s: %s -> %s" % (n, str(a)[:12], str(b)[:12]))
    if len(out) > limit:
        out = out[:limit] + ["(+%d more)" % (len(out) - limit)]
    return out


def _digest(obj):
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


class ConditionSet(object):
    """A normalised description of one run.  Hashable, diffable, stampable."""

    def __init__(self, values, source="", note=""):
        self.values = {g: dict(values.get(g, {})) for g in GROUPS}
        self.source = str(source)
        self.note = str(note)

    # -- construction -------------------------------------------------------
    @classmethod
    def of(cls, decl, env, source="", note=""):
        decl = dict(decl or {})
        env = dict(env or {})
        values = {g: {} for g in GROUPS}
        for f in FIELDS:
            try:
                v = f.resolve(decl, env)
            except Exception as exc:                              # noqa: BLE001
                v = "%s (resolver failed: %s)" % (UNDECLARED, exc)
            values[f.group][f.name] = (UNDECLARED if v is None
                                       else _gate.normalise(v))
        return cls(values, source=source, note=note)

    # -- access -------------------------------------------------------------
    def value(self, path):
        g, _, n = str(path).partition(".")
        return self.values.get(g, {}).get(n, UNDECLARED)

    def flat(self):
        return {f.path: self.value(f.path) for f in FIELDS}

    @property
    def undeclared(self):
        return tuple(p for p, v in sorted(self.flat().items())
                     if self.is_undeclared(v))

    @staticmethod
    def is_undeclared(v):
        return (v is None or (isinstance(v, str) and v.startswith(UNDECLARED))
                or (isinstance(v, float) and not math.isfinite(v)))

    def declared(self, need=ALL_FIELDS):
        """Fields in `need` that are UNDECLARED here."""
        return tuple(p for p in expand(need)
                     if self.is_undeclared(self.value(p)))

    @property
    def digest(self):
        return _digest(self.values)

    def group_digest(self, group):
        return _digest(self.values[group])

    # the build link, read as a gate input rather than a recorded string.
    @property
    def build_link_blocks(self):
        status = self.value("build.build_link_status")
        return str(status) != str(_prov.LINK_LINKED)

    def require_execution(self, where="execution", nx=None):
        """The same entry requirement for a fresh world and a cached run."""
        bad = [p + " is undeclared or nonfinite" for p in self.undeclared]
        if self.build_link_blocks:
            bad.append("build.build_link_status is " + str(self.value("build.build_link_status")))
        if nx is not None and self.value("grid.NX") != nx:
            bad.append("grid.NX does not match the allocated trace")
        for path in ("grid.NX", "grid.NY", "grid.NZ", "grid.CX", "grid.CY", "grid.CZ",
                     "pump.f_saw", "numerics.nt", "numerics.dt_rec", "numerics.dt_step",
                     "numerics.block_records"):
            value = self.value(path)
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not value > 0:
                bad.append(path + " must be positive")
        for path in ("build.source_tree_digest", "build.compiled_source_digest",
                     "build.build_config_digest", "build.engine_binary_sha256"):
            value = self.value(path)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                bad.append(path + " is not a SHA256 digest")
        if bad:
            raise PreconditionHalt(where + ": refusing execution\n    " + "\n    ".join(bad))
        return self

    # -- records ------------------------------------------------------------
    def stamp(self, verified=(), criteria=None, extra=None):
        """The npz keyword block for an artifact produced under these conditions.

        `verified` names what this artifact actually CHECKED (groups or fields),
        which is not the same as what it ran under: a pump-bin certificate runs
        under a material it never tested.  A later stage may only rely on the
        intersection of its need with this set.
        """
        vfields = expand(verified) if verified else ()
        body = dict(schema=SCHEMA, conditions=self.values,
                    digest=self.digest,
                    group_digests={g: self.group_digest(g) for g in GROUPS},
                    verified=list(vfields),
                    verified_declared=list(verified) if verified else [],
                    undeclared=list(self.undeclared),
                    criteria=_gate.normalise(criteria or {}),
                    # the per-file evidence behind build.compiled_source_digest.
                    # It is OUTSIDE `conditions` (the digest above does not cover
                    # it) and exists so that a refused resume can name the source
                    # file that moved, not only the digest that moved.
                    source_map=_source_map(),
                    source=self.source, note=self.note)
        if extra:
            body["extra"] = _gate.normalise(extra)
        blob = json.dumps(body, sort_keys=True, default=str)
        return {KEY: blob, KEY_ALIAS: blob, KEY_DIGEST: self.digest}

    def __repr__(self):
        return "ConditionSet(%s, %d undeclared)" % (self.digest[:12],
                                                    len(self.undeclared))


# ===========================================================================
# reading a stamp back
# ===========================================================================
def read(record):
    """(stamp_dict, reason).  Exactly one reader, and it never raises.

    `record` is an npz mapping, a dict, or a path.  A record with no stamp, an
    unparseable stamp, a stamp whose digest does not match its own contents
    (truncation, hand-editing) or a stamp of an unknown schema returns
    (None, reason) -- never a KeyError and never a default.
    """
    if isinstance(record, str):
        if not os.path.isfile(record):
            return None, "no such record: %s" % record
        try:
            import numpy as np                                    # noqa: PLC0415
            record = np.load(record, allow_pickle=True)
        except Exception as exc:                                  # noqa: BLE001
            return None, "unreadable record (%s)" % exc
    raw = None
    for key in (KEY, KEY_ALIAS):
        try:
            if key in getattr(record, "files", []) or key in record:
                raw = str(record[key])
                break
        except Exception:                                         # noqa: BLE001
            continue
    if not raw:
        return None, ("carries no condition stamp (%r), so the conditions it "
                      "was produced under are unknown; it cannot be shown to "
                      "belong to this run" % KEY)
    try:
        body = json.loads(raw)
    except Exception as exc:                                      # noqa: BLE001
        return None, "condition stamp is unparseable (%s)" % exc
    if not isinstance(body, dict):
        return None, ("condition stamp is a %s, not an object"
                      % type(body).__name__)
    if str(body.get("schema")) != SCHEMA:
        return None, ("condition stamp schema is %r, not %r"
                      % (body.get("schema"), SCHEMA))
    vals = body.get("conditions")
    if not isinstance(vals, dict):
        return None, "condition stamp carries no conditions object"
    if _digest({g: dict(vals.get(g, {})) for g in GROUPS}) != \
            str(body.get("digest")):
        return None, ("condition stamp digest does not match its own contents "
                      "(truncated or edited record)")
    return body, ""


def stored_set(body):
    """The ConditionSet a stamp records."""
    return ConditionSet(body.get("conditions") or {},
                        source=str(body.get("source", "")),
                        note=str(body.get("note", "")))


# ===========================================================================
# THE comparison
# ===========================================================================
def compare(body, current, need, allow=(), criteria=None):
    """Problems with using the stamped artifact as a prerequisite of `current`.

    Returns a list of strings, empty when the artifact covers `need`.  Three
    ways it can fail, all named:
      * the artifact did not VERIFY a field this stage needs;
      * the field is UNDECLARED on either side (a vacuous comparison is not a
        passing comparison);
      * the values differ.
    `allow` names fields (or groups) a legitimate prerequisite may differ in --
    e.g. the pump-bin certificate is measured at one eps_0 and consumed at
    another, because it tests WHERE the pump power lands, not how much.
    """
    want = [p for p in expand(need) if p not in set(expand(allow) if allow
                                                    else ())]
    stored = stored_set(body)
    verified = set(body.get("verified") or ())
    problems = []
    for path in want:
        if path not in verified:
            problems.append("%s: the artifact does not declare that it "
                            "verified this condition (verified = %s)"
                            % (path, sorted(verified) or "nothing"))
            continue
        a, b = stored.value(path), current.value(path)
        if ConditionSet.is_undeclared(a) or ConditionSet.is_undeclared(b):
            problems.append("%s: %s -> %s (an UNDECLARED condition cannot be "
                            "compared)" % (path, a, b))
        elif _canon(a) != _canon(b):
            problems.append("%s: %r -> %r" % (path, a, b))
            if path in ("build.compiled_source_digest",
                        "build.source_tree_digest"):
                for line in _named_source_diffs(body):
                    problems.append("    changed source  " + line)
    # CRITERIA: the thresholds the artifact's own verdict was taken at (a
    # tolerance, a required bin, a record length).  They are not conditions of
    # the run, but adopting a verdict measured at a looser threshold is the same
    # class of error, so they go through the same named comparison instead of a
    # bespoke per-stage check.
    if criteria:
        stored_crit = body.get("criteria")
        stored_crit = stored_crit if isinstance(stored_crit, dict) else {}
        for key in sorted(criteria):
            want_v = _gate.normalise(criteria[key])
            if key not in stored_crit:
                problems.append("criteria.%s: the artifact records no value "
                                "for this criterion (it records %s)"
                                % (key, sorted(stored_crit) or "none"))
            elif _canon(stored_crit[key]) != _canon(want_v):
                problems.append("criteria.%s: %r -> %r"
                                % (key, stored_crit[key], want_v))
    return problems


def state(gate_name, what, record, current, need, allow=(), criteria=None,
          require_build_link=False):
    """The comparison as a Gate.  Never raises on a bad record.

    NOT_EVALUABLE  the artifact is missing, unstamped, malformed or does not
                   cover `need`
    PASS           it covers `need` and every needed field agrees
    """
    body, why = read(record)
    if body is None:
        return _gate.Gate(gate_name, _gate.NOT_EVALUABLE,
                          "%s %s" % (what, why),
                          detail=dict(artifact=str(what), reason=why,
                                      need=list(expand(need))))
    problems = compare(body, current, need, allow=allow, criteria=criteria)
    build_link_blocks = current.build_link_blocks
    if require_build_link and build_link_blocks:
        problems.append(
            "build.build_link_status = %s: the installed engine binary is not "
            "shown to be the output of the selected build tree, so a run made "
            "with it cannot be attributed to these sources "
            "(see PROVENANCE_BUILD_MANIFEST.md)"
            % current.value("build.build_link_status"))
    if problems:
        return _gate.Gate(
            gate_name, _gate.NOT_EVALUABLE,
            "%s was produced under DIFFERENT conditions, %d of the %d this "
            "stage requires:\n    %s"
            % (what, len(problems), len(expand(need)), "\n    ".join(problems)),
            detail=dict(artifact=str(what), problems=problems,
                        need=list(expand(need)),
                        stored_digest=str(body.get("digest")),
                        current_digest=current.digest))
    return _gate.Gate(gate_name, _gate.PASS,
                      "%s covers all %d required conditions"
                      % (what, len(expand(need))),
                      detail=dict(artifact=str(what),
                                  need=list(expand(need)),
                                  stored_digest=str(body.get("digest")),
                                  current_digest=current.digest))


def require(gate_name, what, record, current, need, allow=(), criteria=None,
            require_build_link=False):
    """state(), but a non-PASS raises PreconditionHalt naming the fields."""
    g = state(gate_name, what, record, current, need, allow=allow,
              criteria=criteria, require_build_link=require_build_link)
    if g.blocks:
        raise PreconditionHalt("%s: %s -- %s" % (gate_name, g.state, g.reason))
    return g
