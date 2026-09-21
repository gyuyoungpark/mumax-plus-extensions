"""Forward provenance for the commensurate SAW-magnonics campaign.

Two things the 2026-09-18 audit found missing, and what this module does about
them.

(1) "Recording a source hash and a binary hash side by side does not link them."
    Correct.  `provenance()` in `_harness.py` writes sha256(chiralsawfield.cu)
    and sha256(_mumaxpluscpp*.pyd) into the same dict; nothing in that dict says
    the second was produced from the first.  `build_manifest()` below builds the
    link out of artefacts that actually connect the two:

      * the CMake build tree that produced the binary records, in its
        CMakeCache.txt, the source directory it was configured against
        (CMAKE_HOME_DIRECTORY), the floating-point precision (FP_PRECISION),
        the CUDA architectures, the generator and the compiler flags;
      * that same build tree still holds the .pyd it emitted, under
        build/lib.<plat>/;
      * if the INSTALLED .pyd is byte-identical to the build tree's .pyd, then
        the installed binary IS this build tree's output -- not a coincidence of
        two hashes written next to each other;
      * a forward receipt records a successful clean build with unchanged
        source hashes before/after compilation, configuration and output hash.
        Source mtimes alone never establish this connection.

    What that chain proves and what it does not is spelled out in
    ../PROVENANCE_BUILD_MANIFEST.md.  It is deliberately a four-state verdict,
    never a boolean, and `blocks_downstream` is True in every state except the
    one where the chain closes.

(2) "Make the forward provenance record actually participate in the skip/resume
    decision."  `run_identity()` produces the normalised identity of a run and
    `gate_resume()` is the single place where a stored checkpoint is compared
    against the run being requested.  `_harness.BlockRun` calls it; the P0-4
    run-manifest fix should call THIS function rather than adding a second,
    independent comparison.

    The identity is split in two on purpose:

      physics      nt, nx, dt_rec, the caller's physical parameters, the kernel
                   source hash, the engine binary hash and the build-manifest
                   link status.  A difference here REFUSES the checkpoint.
      bookkeeping  script / harness / analysis-module hashes, host, versions.
                   A difference here is RECORDED and PRINTED as a warning and
                   does not refuse, because a comment edit in a driver script
                   does not invalidate integrated magnetisation.

    Nothing here ever returns a silent pass: the states are RESUME_OK,
    REFUSED_IDENTITY_MISMATCH and REFUSED_IDENTITY_ABSENT, and the two refusals
    are designed to stop the caller.

No number in this module is invented.  Every field is read off a file, and
every lookup that can fail is guarded and recorded as "NOT DETERMINABLE".
"""

import glob
import hashlib
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))                  # .../runs
RC_DIR = os.path.abspath(os.path.join(HERE, ".."))                 # revision_check
EXT_DIR = os.path.abspath(os.path.join(RC_DIR, ".."))              # SAW-magnonics
SRC_DIR = os.path.join(EXT_DIR, "src")
REPO = os.path.abspath(os.path.join(EXT_DIR, "..", ".."))          # mumax-plus-dev
PHYSICS_DIR = os.path.join(REPO, "src", "physics")
KERNEL_CU = os.path.join(PHYSICS_DIR, "chiralsawfield.cu")

SCHEMA_BUILD = "saw_revision_check/build_manifest/1"
SCHEMA_IDENT = "saw_revision_check/run_identity/1"

# ---- the four link states.  There is no fifth, and none of them is a bool ----
LINK_LINKED = "LINKED_VIA_BUILD_TREE"
LINK_UNPROVEN = "CONSISTENT_BUT_UNPROVEN"
LINK_REFUTED = "REFUTED"
LINK_UNKNOWN = "NOT_DETERMINABLE"

# ---- the three check() verdicts --------------------------------------------
CHECK_MATCH = "MATCH"
CHECK_MISMATCH = "MISMATCH"
CHECK_UNKNOWN = "NOT_DETERMINABLE"

# ---- the three resume states ----------------------------------------------
RESUME_OK = "RESUME_OK"
RESUME_REFUSED_MISMATCH = "REFUSED_IDENTITY_MISMATCH"
RESUME_REFUSED_ABSENT = "REFUSED_IDENTITY_ABSENT"

NOT_DET = "NOT DETERMINABLE"


class ResumeRefused(RuntimeError):
    """Raised instead of silently re-running or silently trusting a checkpoint."""


# ===========================================================================
# small helpers
# ===========================================================================
def sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:                                       # noqa: BLE001
        return "%s (%s)" % (NOT_DET, exc)


def _stat(path):
    try:
        st = os.stat(path)
        return dict(size=int(st.st_size),
                    mtime_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(st.st_mtime)),
                    mtime_epoch=float(st.st_mtime))
    except Exception as exc:                                       # noqa: BLE001
        return dict(size=None, mtime_utc="%s (%s)" % (NOT_DET, exc),
                    mtime_epoch=None)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _digest(obj):
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()


def _norm_float(v):
    """Normalise a float so that a re-read from npz compares equal."""
    try:
        return float("%.12g" % float(v))
    except Exception:                                              # noqa: BLE001
        return v


# ===========================================================================
# which sources the manifest covers
# ===========================================================================
_COMPILED_EXT = (".cu", ".cuh", ".cpp", ".hpp", ".h", ".c", ".cc")
_BUILD_DESC = ("CMakeLists.txt", "pyproject.toml", "setup.py")


def compiled_source_files():
    """Every source the extension module is compiled FROM, by discovery.

    REWRITTEN 2026-09-18 (round 2).  The previous version hardcoded 18 paths
    against 131 files in src/physics and omitted magnetoelasticfield.cu -- the
    B_1 kernel these runs integrate with enable_mel=True by default -- plus
    exchange.cu, anisotropy.cu, zeeman.cu, demag.cpp, thermalnoise.cu and
    minimizer.cu.  A digest built from a hand-written list cannot move when a
    kernel outside the list is edited, so the resume gate accepted a stale
    checkpoint by construction.  That is the CLASS of defect: any enumeration
    of "the files that matter" is wrong the moment the integrated path changes.

    Discover the complete src/ tree and CMake-backed modules under extension/
    or extensions/, including headers and CMake includes. This is a conservative
    source set, not a claim that every discovered file is a translation unit.
    Legacy analysis archives without a build descriptor are not engine inputs.
    `build_tree_object_sources()` independently reads the selected build tree's
    project files. A compiled source outside these roots still BLOCKS linking
    rather than being silently absent from the receipt.

    PHYSICS_DIR is read at call time (not import time) so a test can point it
    at a copy of the tree.
    """
    seen = []
    roots = [PHYSICS_DIR, os.path.join(REPO, "src")]
    for layout in ("extension", "extensions"):
        parent = os.path.join(REPO, layout)
        if not os.path.isdir(parent):
            continue
        with os.scandir(parent) as entries:
            for entry in entries:
                if (entry.is_dir() and
                        os.path.isfile(os.path.join(entry.path, "CMakeLists.txt"))):
                    roots.append(entry.path)
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for f in sorted(files):
                if (f.lower().endswith(_COMPILED_EXT + (".cmake",)) or
                        f == "CMakeLists.txt"):
                    seen.append(os.path.join(dirpath, f))
    for name in _BUILD_DESC:
        p = os.path.join(REPO, name)
        seen.append(p)
    return sorted(dict.fromkeys(os.path.abspath(p) for p in seen))


def python_source_files():
    """Interpreted sources that shape a run but are NOT compiled into the .pyd.

    They belong in the run identity (editing saw_chiral.py changes the drive
    that is applied) but NOT in the binary-staleness comparison: a .py edit does
    not make the compiled binary stale, and treating it as such would refuse
    every checkpoint after a comment edit.
    """
    names = [os.path.join(SRC_DIR, "saw_chiral.py"),
             os.path.join(SRC_DIR, "saw.py"),
             os.path.join(RC_DIR, "saw_analysis.py"),
             os.path.join(HERE, "_harness.py")]
    names.extend(glob.glob(os.path.join(HERE, "G[0-9]*.py")))
    return [os.path.abspath(p) for p in names]


def source_files():
    """Every recorded source: compiled + interpreted.  Kept as the old name
    because callers and tests use it."""
    return compiled_source_files() + python_source_files()


def build_tree_object_sources(cache_path):
    """The sources the SELECTED build tree actually compiles, read out of it.

    MSBuild writes one .vcxproj per target with a <CudaCompile>/<ClCompile>
    Include= entry per translation unit; Makefile/Ninja generators write
    CMakeFiles/*.dir/**/*.o(bj).  Either is an independent statement of what was
    compiled, and it is compared against the glob so a divergence is visible.
    Returns (paths, how) and never raises.
    """
    if not cache_path or not os.path.isfile(cache_path):
        return [], "%s (no CMake cache)" % NOT_DET
    tree = os.path.dirname(os.path.abspath(cache_path))
    hits, how = set(), []
    try:
        import re                                                  # noqa: PLC0415
        pat = re.compile(r'Include="([^"]+\.(?:cu|cpp|cc|c))"', re.I)
        projs = glob.glob(os.path.join(tree, "**", "*.vcxproj"), recursive=True)
        for pr in projs:
            with open(pr, encoding="utf-8", errors="replace") as fh:
                for m in pat.finditer(fh.read()):
                    p = m.group(1)
                    if not os.path.isabs(p):
                        p = os.path.join(os.path.dirname(pr), p)
                    p = os.path.abspath(p)
                    # CMake's own compiler-identification probes live inside the
                    # build tree (CMakeFiles/<ver>/CompilerId*/...).  They are
                    # generated by CMake, are not repo sources and are not
                    # linked into the module, so they are not provenance.
                    if "CMakeFiles" in p.replace("\\", "/").split("/"):
                        continue
                    hits.add(p)
        if projs:
            how.append("%d vcxproj" % len(projs))
        objs = [p for p in glob.glob(os.path.join(tree, "**", "*.obj"),
                                     recursive=True)]
        objs += glob.glob(os.path.join(tree, "**", "*.o"), recursive=True)
        if objs:
            how.append("%d object files" % len(objs))
    except Exception as exc:                                       # noqa: BLE001
        return sorted(hits), "%s (%s)" % (NOT_DET, exc)
    if not hits:
        return [], "%s (no translation units found in %s)" % (NOT_DET, tree)
    return sorted(hits), "read from " + ", ".join(how)


def _rel(path):
    try:
        return os.path.relpath(path, REPO).replace("\\", "/")
    except Exception:                                              # noqa: BLE001
        return path.replace("\\", "/")


# ===========================================================================
# the build configuration, read out of the CMake build tree
# ===========================================================================
_CACHE_KEYS = ("CMAKE_HOME_DIRECTORY", "FP_PRECISION", "CMAKE_CUDA_ARCHITECTURES",
               "CMAKE_GENERATOR", "CMAKE_GENERATOR_PLATFORM", "CMAKE_CXX_FLAGS",
               "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CUDA_COMPILER",
               "CMAKE_CXX_COMPILER", "CMAKE_BUILD_TYPE",
               # where the tree says it puts what it emits: this is how the
               # binary is derived from the SELECTED cache instead of globbed
               "CMAKE_LIBRARY_OUTPUT_DIRECTORY",
               "CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE",
               "CMAKE_RUNTIME_OUTPUT_DIRECTORY")


def _parse_cmake_cache(path):
    out = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(("#", "//")):
                    continue
                name, _, value = line.partition("=")
                key = name.split(":", 1)[0]
                if key in _CACHE_KEYS:
                    out.setdefault(key, value)
    except Exception as exc:                                       # noqa: BLE001
        out["_error"] = "%s (%s)" % (NOT_DET, exc)
    return out


def _discover_build_trees():
    """Every CMake build tree in the repo, newest CMakeCache first."""
    explicit = os.environ.get("SAW_BUILD_CACHE")
    if explicit:
        return [os.path.abspath(explicit)] if os.path.isfile(explicit) else []
    hits = glob.glob(os.path.join(REPO, "build*", "**", "CMakeCache.txt"),
                     recursive=True)
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits


def _output_dirs(cache_path):
    """The output directories THIS cache declares, in its own words.

    CMAKE_LIBRARY_OUTPUT_DIRECTORY (and the _RELEASE variant) is where the tree
    puts the module it emits.  A relative value is resolved against the tree.
    """
    if not cache_path or not os.path.isfile(cache_path):
        return []
    tree = os.path.dirname(os.path.abspath(cache_path))
    cfg = _parse_cmake_cache(cache_path)
    out = []
    for key in ("CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE",
                "CMAKE_LIBRARY_OUTPUT_DIRECTORY",
                "CMAKE_RUNTIME_OUTPUT_DIRECTORY"):
        v = str(cfg.get(key, "")).strip().strip('"')
        if not v or v.startswith(NOT_DET):
            continue
        v = v.replace("\\", "/")
        p = v if os.path.isabs(v) else os.path.join(tree, v)
        out.append(os.path.abspath(p))
    if not out:
        out.append(tree)
    return list(dict.fromkeys(out))


def _build_tree_binary(cache_path, precision="single"):
    """The .pyd THIS build tree emitted, derived from ITS OWN output directory.

    FIXED 2026-09-18 (round 2).  The previous body was `_ = cache_path` followed
    by a glob of REPO/build/lib.*, so a cache from one tree and a binary from
    another still reported LINKED_VIA_BUILD_TREE.  The argument is now the only
    thing that decides: the cache names its output directory, and the module is
    looked for THERE (and, for multi-config generators, under Release/).
    Returns None when the tree's own output cannot be found -- which is a
    CONSISTENT_BUT_UNPROVEN link, not a pass.
    """
    hits = []
    for d in _output_dirs(cache_path):
        for sub in ("", "Release", "RelWithDebInfo", "Debug"):
            base = os.path.join(d, sub) if sub else d
            for ext in ("pyd", "so", "dll"):
                hits += glob.glob(os.path.join(
                    base, "_mumaxpluscpp_%s.*.%s" % (precision, ext)))
    hits = [h for h in dict.fromkeys(hits) if os.path.isfile(h)]
    if not hits:
        return None
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def _find_build_trees():
    """The build trees that are SELECTABLE as the provenance of a binary.

    A tree that emitted nothing cannot be the origin of the installed module, so
    it is not a candidate.  This is what closes the ambiguity the audit found:
    two FP_PRECISION=SINGLE trees existed, selection was by CMakeCache mtime
    alone, and build_test_single/ -- which declares the output directory
    `build_test_single/lib` and never created it -- could win.  Excluded trees
    are still listed in the manifest under `build_trees_excluded` with the
    reason, so nothing is hidden by being dropped.
    """
    keep = []
    for c in _discover_build_trees():
        if any(_build_tree_binary(c, p) for p in ("single", "double")):
            keep.append(c)
    return keep


def _excluded_build_trees():
    keep = set(_find_build_trees())
    return [dict(cmake_cache_path=c.replace("\\", "/"),
                 reason="emitted no _mumaxpluscpp module in its own declared "
                        "output directory %s" % _output_dirs(c))
            for c in _discover_build_trees() if c not in keep]


def _installed_binary(precision="single"):
    """The .pyd python WOULD IMPORT, not the newest one lying about.

    FIXED 2026-09-18 (round 2): the previous body globbed sys.path and sorted by
    mtime, so a stray newer copy anywhere on the path was hashed instead of the
    module that actually loads.  Resolution now goes through the import system
    (PathFinder over sys.path, in order), with a sys.path-ORDER glob as the
    fallback -- never mtime.
    """
    name = "_mumaxpluscpp_%s" % precision
    loaded = getattr(sys.modules.get(name), "__file__", None)
    if loaded and os.path.isfile(loaded):
        return loaded
    try:
        from importlib.machinery import PathFinder                 # noqa: PLC0415
        spec = PathFinder.find_spec(name, [d for d in sys.path if d])
        if spec is not None and spec.origin and os.path.isfile(spec.origin):
            return spec.origin
    except Exception:                                              # noqa: BLE001
        pass
    dirs = [d for d in sys.path if d]
    try:
        import sysconfig                                           # noqa: PLC0415
        sp = sysconfig.get_paths().get("purelib")
        if sp:
            dirs.append(sp)
    except Exception:                                              # noqa: BLE001
        pass
    for d in dirs:
        for ext in ("pyd", "so"):
            hits = sorted(glob.glob(os.path.join(d, "%s.*.%s" % (name, ext))))
            hits = [h for h in hits if os.path.isfile(h)]
            if hits:
                return hits[0]
    return None


# ===========================================================================
# THE BUILD MANIFEST
# ===========================================================================
def check_forward_receipt(cache, compiled_map, binary_sha):
    """Match a successful observed compilation, never infer it from mtimes."""
    path = os.path.join(os.path.dirname(cache or ""), "forward_build.json")
    try:
        with open(path, encoding="utf-8") as fh:
            receipt = json.load(fh)
        required = dict(schema="saw_forward_build/1", status="complete",
                        sources_before=compiled_map, sources_after=compiled_map,
                        cache_sha256=sha256_file(cache), binary_sha256=binary_sha,
                        configure_returncode=0, build_returncode=0)
        bad = [key for key, val in required.items() if receipt.get(key) != val]
        if "--clean-first" not in receipt.get("build_command", []):
            bad.append("clean compilation command")
        if not receipt.get("configure_command"):
            bad.append("configure command")
        if bad:
            return False, "forward receipt mismatch: " + ", ".join(bad), path
        return True, "source hashes before/after clean compilation and output agree", path
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return False, "missing or unreadable forward receipt: " + str(exc), path


def build_manifest(precision="single"):
    """Source files + build configuration -> binary, with an explicit verdict.

    Returns a dict.  `link_status` is one of LINK_LINKED / LINK_UNPROVEN /
    LINK_REFUTED / LINK_UNKNOWN and `blocks_downstream` is False only for
    LINK_LINKED.  Nothing in here is a default: every field that cannot be read
    is the string "NOT DETERMINABLE (<reason>)".
    """
    # ---- sources ----------------------------------------------------------
    compiled = set(os.path.abspath(p) for p in compiled_source_files())
    sources = {}
    newest = None
    newest_src = None
    for p in source_files():
        st = _stat(p)
        is_compiled = os.path.abspath(p) in compiled
        sources[_rel(p)] = dict(path=p.replace("\\", "/"),
                                sha256=sha256_file(p),
                                compiled_into_binary=bool(is_compiled), **st)
        if is_compiled and st["mtime_epoch"] is not None and \
                (newest is None or st["mtime_epoch"] > newest):
            newest, newest_src = st["mtime_epoch"], _rel(p)
    tree_digest = _digest({k: v["sha256"] for k, v in sources.items()})
    compiled_digest = _digest({k: v["sha256"] for k, v in sources.items()
                               if v["compiled_into_binary"]})

    # ---- build configuration ---------------------------------------------
    # Selection is no longer "newest CMakeCache with the right precision": the
    # tree that can be shown to have emitted the module python imports wins, and
    # ambiguity is recorded rather than resolved by mtime (audit sec.9, round 2).
    caches = _find_build_trees()
    inst_pre = _installed_binary(precision)
    inst_sha = sha256_file(inst_pre) if inst_pre else None
    right_precision = [c for c in caches
                       if _parse_cmake_cache(c).get("FP_PRECISION", "")
                       .strip().upper() == precision.upper()]
    matching = []
    for c in right_precision:
        b = _build_tree_binary(c, precision)
        if b and inst_sha and sha256_file(b) == inst_sha:
            matching.append(c)
    if matching:
        cache = matching[0]
    elif right_precision:
        cache = right_precision[0]
    elif caches:
        cache = caches[0]
    else:
        cache = None
    cfg = _parse_cmake_cache(cache) if cache else {}
    obj_sources, obj_how = build_tree_object_sources(cache)
    missed = sorted(_rel(p) for p in obj_sources
                    if os.path.abspath(p) not in compiled
                    and os.path.isfile(p))

    build_config = dict(
        requested_precision=precision,
        cmake_cache_path=(cache.replace("\\", "/") if cache else NOT_DET),
        cmake_cache_sha256=(sha256_file(cache) if cache else NOT_DET),
        cmake_home_directory=cfg.get("CMAKE_HOME_DIRECTORY", NOT_DET),
        fp_precision=cfg.get("FP_PRECISION", NOT_DET),
        cuda_architectures=cfg.get("CMAKE_CUDA_ARCHITECTURES", NOT_DET),
        generator=cfg.get("CMAKE_GENERATOR", NOT_DET),
        generator_platform=cfg.get("CMAKE_GENERATOR_PLATFORM", NOT_DET),
        cxx_flags=cfg.get("CMAKE_CXX_FLAGS", NOT_DET),
        cxx_flags_release=cfg.get("CMAKE_CXX_FLAGS_RELEASE", NOT_DET),
        cxx_compiler=cfg.get("CMAKE_CXX_COMPILER", NOT_DET),
        cuda_compiler=cfg.get("CMAKE_CUDA_COMPILER", NOT_DET),
        build_type=cfg.get("CMAKE_BUILD_TYPE", NOT_DET),
        n_build_trees_found=len(caches),
        n_build_trees_same_precision=len(right_precision),
        build_tree_output_dirs=[d.replace("\\", "/")
                                for d in _output_dirs(cache)] if cache else [],
        build_trees_selectable=[c.replace("\\", "/") for c in caches],
        build_trees_excluded=_excluded_build_trees(),
        selected_because=("its own output is byte-identical to the installed "
                          "module" if matching else
                          ("the only tree at FP_PRECISION=%s that emitted a "
                           "module" % precision if len(right_precision) == 1
                           else "NOT DETERMINABLE: no tree's output matches the "
                                "installed module")),
        object_sources_how=obj_how,
        n_object_sources=len(obj_sources),
        compiled_sources_missed_by_glob=missed,
    )
    # the fields a rebuild would change, as one comparable digest
    build_config["config_digest"] = _digest(
        {k: build_config[k] for k in sorted(build_config)
         if k not in ("build_trees_excluded", "build_trees_selectable",
                      "object_sources_how", "n_object_sources",
                      "n_build_trees_found", "n_build_trees_same_precision",
                      "selected_because", "config_digest")})

    # ---- binary ------------------------------------------------------------
    inst = _installed_binary(precision)
    bt = _build_tree_binary(cache, precision) if cache else None
    binary = dict(
        installed_path=(inst.replace("\\", "/") if inst else NOT_DET),
        installed_sha256=(sha256_file(inst) if inst else NOT_DET),
        build_tree_path=(bt.replace("\\", "/") if bt else NOT_DET),
        build_tree_sha256=(sha256_file(bt) if bt else NOT_DET),
    )
    if inst:
        binary.update({"installed_" + k: v for k, v in _stat(inst).items()})
    if bt:
        binary.update({"build_tree_" + k: v for k, v in _stat(bt).items()})

    # ---- THE LINK ----------------------------------------------------------
    # Staleness is asked of the COMPILED sources only: editing saw_chiral.py
    # does not make the .pyd stale, and treating it as stale would refuse every
    # checkpoint after a comment edit (a gate must refuse what invalidates the
    # run and no more).  Interpreted sources are still hashed into the identity.
    bin_mtime = binary.get("installed_mtime_epoch")
    stale = sorted(k for k, v in sources.items()
                   if v["compiled_into_binary"]
                   and v["mtime_epoch"] is not None and bin_mtime is not None
                   and v["mtime_epoch"] > bin_mtime)
    same = (isinstance(binary["installed_sha256"], str)
            and len(binary["installed_sha256"]) == 64
            and binary["installed_sha256"] == binary["build_tree_sha256"])
    home = str(build_config["cmake_home_directory"]).replace("\\", "/").rstrip("/")
    home_ok = bool(home) and os.path.normcase(os.path.abspath(home)) == \
        os.path.normcase(os.path.abspath(REPO))

    link = dict(
        installed_equals_build_tree_output=same,
        build_tree_configured_against_this_source_tree=home_ok,
        binary_newer_than_all_sources=(bin_mtime is not None and not stale),
        sources_changed_since_build=stale,
        newest_source=newest_src or NOT_DET,
        newest_source_mtime_utc=(time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime(newest))
                                 if newest else NOT_DET),
        binary_mtime_utc=binary.get("installed_mtime_utc", NOT_DET),
    )

    receipt_ok, receipt_reason, receipt_path = check_forward_receipt(
        cache, {k: v["sha256"] for k, v in sources.items() if v["compiled_into_binary"]},
        binary["installed_sha256"])
    link.update(forward_receipt_matches=receipt_ok, forward_receipt_reason=receipt_reason,
                forward_receipt_path=receipt_path,
                forward_receipt_sha256=sha256_file(receipt_path))

    # ---- verdict.  no default branch reaches LINK_LINKED. ------------------
    if inst is None or cache is None:
        status = LINK_UNKNOWN
        why = ("the installed binary or the CMake cache is not on this machine "
               "(installed=%s, cache=%s)"
               % (bool(inst), bool(cache)))
    elif bt is None:
        # the selected tree exists but its OWN output is gone, so the binary
        # cannot be derived from it.  Consistent, unproven, and it blocks.
        status = LINK_UNPROVEN
        why = ("the selected build tree %s declares its output directory as %s "
               "and no _mumaxpluscpp_%s module is there, so the installed "
               "binary cannot be derived from the tree that was configured; "
               "the link is unproven, not established"
               % (build_config["cmake_cache_path"],
                  build_config["build_tree_output_dirs"], precision))
    elif missed:
        status = LINK_UNPROVEN
        why = ("the selected build tree compiles %d source(s) that the recorded "
               "source list does not cover (%s), so the digest cannot be shown "
               "to cover the binary" % (len(missed), ", ".join(missed[:6])))
    elif not same:
        status = LINK_REFUTED
        why = ("the installed binary is not byte-identical to this build tree's "
               "output, so it was not produced by this build tree")
    elif not home_ok:
        status = LINK_REFUTED
        why = ("CMAKE_HOME_DIRECTORY is %r, not this source tree %r"
               % (home, REPO.replace("\\", "/")))
    elif not receipt_ok:
        status = LINK_UNPROVEN
        why = receipt_reason + "; timestamps do not prove source-to-build linkage"
    else:
        status = LINK_LINKED
        why = ("the loaded binary matches the recorded clean compilation; "
               "pre/post source hashes and CMake configuration match the current tree")

    m = dict(
        schema=SCHEMA_BUILD,
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=platform.node(), platform=platform.platform(),
        python=sys.version.split()[0],
        repo=REPO.replace("\\", "/"),
        sources=sources,
        source_tree_digest=tree_digest,
        compiled_source_digest=compiled_digest,
        n_sources=len(sources),
        n_compiled_sources=sum(1 for v in sources.values()
                               if v["compiled_into_binary"]),
        build_config=build_config,
        binary=binary,
        link=link,
        link_status=status,
        link_reason=why,
        blocks_downstream=(status != LINK_LINKED),
    )
    m["manifest_id"] = _digest({k: v for k, v in m.items()
                                if k not in ("utc", "manifest_id")})
    return m


def write_build_manifest(path, precision="single"):
    """Write the manifest as JSON.  Returns the manifest."""
    m = build_manifest(precision)
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".%d.tmp" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return m


# Fields whose difference is a genuine mismatch rather than a re-measurement.
# EVERY build_config field that describes the build is compared; the round-1
# list recorded build_type, cxx_compiler and cuda_compiler and then left them
# out of the comparison, so forging any of them returned MATCH.  The list is
# now derived from the recorded config itself (minus the fields that are
# descriptions of THIS measurement rather than of the build), so a field added
# to build_config cannot be silently unchecked.
_CHECK_FIELDS = ("source_tree_digest", "compiled_source_digest")
_CONFIG_NOT_COMPARED = ("build_trees_excluded", "build_trees_selectable",
                        "object_sources_how", "n_object_sources",
                        "n_build_trees_found", "n_build_trees_same_precision",
                        "selected_because", "requested_precision")
_CHECK_BINARY = ("installed_sha256", "build_tree_sha256")


def _config_fields(*cfgs):
    keys = set(_CHECK_CONFIG)
    for c in cfgs:
        if isinstance(c, dict):
            keys |= set(c)
    return tuple(sorted(k for k in keys if k not in _CONFIG_NOT_COMPARED))


# kept as a name because tests and docs refer to it; it is now the FLOOR of
# what is compared, not the whole of it
_CHECK_CONFIG = ("cmake_home_directory", "fp_precision", "cuda_architectures",
                 "generator", "generator_platform", "cxx_flags",
                 "cxx_flags_release", "cmake_cache_sha256", "build_type",
                 "cxx_compiler", "cuda_compiler", "config_digest",
                 "build_tree_output_dirs")


def check_build_manifest(path, precision="single"):
    """Compare a stored manifest against the tree as it is now.

    Returns dict(verdict, diffs, blocks_downstream, stored, current).
    A missing or unreadable manifest is CHECK_UNKNOWN and blocks; it is never
    treated as "nothing to check, therefore fine".
    """
    if not os.path.isfile(path):
        return dict(verdict=CHECK_UNKNOWN, diffs=[],
                    reason="no build manifest at %s" % path,
                    blocks_downstream=True, stored=None, current=None)
    try:
        with open(path, encoding="utf-8") as fh:
            stored = json.load(fh)
    except Exception as exc:                                       # noqa: BLE001
        return dict(verdict=CHECK_UNKNOWN, diffs=[],
                    reason="unreadable build manifest: %s" % exc,
                    blocks_downstream=True, stored=None, current=None)
    # A JSON null, list or string parses fine and is still not a manifest.  The
    # round-1 code went straight to stored.get() and died with AttributeError
    # where it promised NOT DETERMINABLE.
    if not isinstance(stored, dict):
        return dict(verdict=CHECK_UNKNOWN, diffs=[],
                    reason=("build manifest at %s is a JSON %s, not an object"
                            % (path, type(stored).__name__)),
                    blocks_downstream=True, stored=None, current=None)

    cur = build_manifest(precision)
    diffs = []
    if stored.get("schema") != cur["schema"]:
        diffs.append(("schema", stored.get("schema"), cur["schema"]))
    for f in _CHECK_FIELDS:
        if stored.get(f) != cur.get(f):
            diffs.append((f, stored.get(f), cur.get(f)))
    for f in _config_fields(stored.get("build_config"), cur["build_config"]):
        a = (stored.get("build_config") or {}).get(f)
        b = cur["build_config"].get(f)
        if a != b:
            diffs.append(("build_config." + f, a, b))
    for f in _CHECK_BINARY:
        a = (stored.get("binary") or {}).get(f)
        b = cur["binary"].get(f)
        if a != b:
            diffs.append(("binary." + f, a, b))
    # per-file, so a mismatch NAMES the file that moved
    s_src = stored.get("sources") or {}
    for k in sorted(set(s_src) | set(cur["sources"])):
        a = (s_src.get(k) or {}).get("sha256")
        b = (cur["sources"].get(k) or {}).get("sha256")
        if a != b:
            diffs.append(("sources.%s.sha256" % k, a, b))
    if stored.get("link_status") != cur["link_status"]:
        diffs.append(("link_status", stored.get("link_status"), cur["link_status"]))

    verdict = CHECK_MATCH if not diffs else CHECK_MISMATCH
    blocks = bool(diffs) or cur["blocks_downstream"]
    return dict(verdict=verdict, diffs=diffs,
                reason=("identical" if not diffs else
                        "%d field(s) differ" % len(diffs)),
                blocks_downstream=blocks, stored=stored, current=cur)


def default_manifest_path():
    return os.path.join(RC_DIR, "runs", "out", "build_manifest.json")


_LINK_CACHE = {}


def link_summary(precision="single", path=None, refresh=True):
    """Read the current build link without mutating the shared manifest.

    Compare against the last explicitly written snapshot. The compatibility
    argument `refresh` no longer requests a shared-file write. Every call reads
    current hashes; process-local cache entries are diagnostic only.
    On the very first call on a machine there is nothing to compare against and
    `check_verdict` is NOT_DETERMINABLE -- which is the truth, not a pass.

    Returns a small dict safe to embed in a provenance block or a run manifest:
        build_manifest_id, build_link_status, build_link_blocks,
        build_manifest_check, build_manifest_path, source_tree_digest,
        engine_binary_sha256, fp_precision
    """
    key = (precision, path or default_manifest_path())
    # Recheck each boundary: a process cache must not hide changed inputs.
    p = path or default_manifest_path()
    pre = check_build_manifest(p, precision)
    # Reading identity must not mutate a shared manifest (parallel campaign
    # processes can race with its writer). Explicit write_build_manifest calls
    # archive the snapshot at run boundaries.
    man = pre["current"] or build_manifest(precision)
    out = dict(
        build_manifest_path=p.replace("\\", "/"),
        build_manifest_id=man["manifest_id"],
        build_link_status=man["link_status"],
        build_link_blocks=bool(man["blocks_downstream"]),
        build_link_reason=man["link_reason"],
        build_manifest_check=pre["verdict"],
        build_manifest_check_reason=pre["reason"],
        source_tree_digest=man["source_tree_digest"],
        compiled_source_digest=man["compiled_source_digest"],
        # the per-file evidence BEHIND compiled_source_digest, so a refusal can
        # name the source that moved instead of only the digest
        compiled_source_map={k: v["sha256"] for k, v in man["sources"].items()
                             if v.get("compiled_into_binary")},
        build_config_digest=man["build_config"].get("config_digest", NOT_DET),
        engine_binary_sha256=man["binary"].get("installed_sha256", NOT_DET),
        fp_precision=man["build_config"].get("fp_precision", NOT_DET),
    )
    _LINK_CACHE[key] = dict(out)
    return out


def run_manifest_fields(precision="single"):
    """The forward-provenance fields that MUST take part in skip/resume.

    Folded into `_harness.BlockRun`'s run manifest so that the one existing
    comparison (`_gate.diff` on the normalised manifest) covers them too.  No
    second comparison is introduced: a rebuilt kernel changes
    `sha256_chiralsawfield_cu`, `engine_binary_sha256` and `source_tree_digest`,
    the manifest hash moves, and the stored checkpoint is refused by the same
    code path that refuses a changed dt_rec.

    `build_manifest_check` is deliberately NOT included: it describes this
    process's comparison against the previous manifest, not the run's identity,
    and including it would refuse every checkpoint after any rebuild anywhere.
    """
    s = link_summary(precision)
    return dict(
        sha256_chiralsawfield_cu=sha256_file(KERNEL_CU),
        engine_binary_sha256=s["engine_binary_sha256"],
        source_tree_digest=s["source_tree_digest"],
        compiled_source_digest=s["compiled_source_digest"],
        build_config_digest=s["build_config_digest"],
        fp_precision=s["fp_precision"],
        build_link_status=s["build_link_status"],
    )


def build_conditions(precision="single"):
    """The `build` group of a condition set (see runs/_conditions.py).

    One place produces it, so certificates, checkpoints and summaries all record
    the same five build facts, and a stage that needs the build group compares
    them field by field like any other condition.
    """
    s = link_summary(precision)
    return dict(
        source_tree_digest=s["source_tree_digest"],
        compiled_source_digest=s["compiled_source_digest"],
        build_config_digest=s["build_config_digest"],
        engine_binary_sha256=s["engine_binary_sha256"],
        fp_precision=s["fp_precision"],
        build_link_status=s["build_link_status"],
    )


# ===========================================================================
# RUN IDENTITY, and the skip/resume gate
# ===========================================================================
def run_identity(nt, nx, dt_rec, params=None, precision="single",
                 script_path=None, manifest=None):
    """The normalised identity of one integrated block run.

    `physics` fields refuse a mismatched checkpoint; `bookkeeping` fields only
    warn.  See the module docstring for why the split exists.
    """
    m = manifest if manifest is not None else build_manifest(precision)
    phys = dict(
        nt=int(nt), nx=int(nx), dt_rec=_norm_float(dt_rec),
        params=_canon(params if params is not None else {}),
        sha256_chiralsawfield_cu=sha256_file(KERNEL_CU),
        engine_binary_sha256=m["binary"].get("installed_sha256", NOT_DET),
        fp_precision=m["build_config"].get("fp_precision", NOT_DET),
        build_link_status=m["link_status"],
        build_manifest_id=m["manifest_id"],
    )
    book = dict(
        schema=SCHEMA_IDENT,
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        host=platform.node(),
        python=sys.version.split()[0],
        sha256_harness=sha256_file(os.path.join(HERE, "_harness.py")),
        sha256_provenance=sha256_file(os.path.join(HERE, "_provenance.py")),
        sha256_saw_analysis_py=sha256_file(os.path.join(RC_DIR,
                                                        "saw_analysis.py")),
        sha256_script=(sha256_file(script_path) if script_path else NOT_DET),
        script=(os.path.abspath(script_path).replace("\\", "/")
                if script_path else NOT_DET),
    )
    ident = dict(schema=SCHEMA_IDENT, physics=phys, bookkeeping=book)
    ident["physics_digest"] = _digest(phys)
    return ident


def compare_identity(stored, want):
    """-> (physics_diffs, bookkeeping_diffs), each a list of (field, old, new)."""
    sp = (stored or {}).get("physics") or {}
    wp = (want or {}).get("physics") or {}
    sb = (stored or {}).get("bookkeeping") or {}
    wb = (want or {}).get("bookkeeping") or {}
    pd = [(k, sp.get(k), wp.get(k)) for k in sorted(set(sp) | set(wp))
          if sp.get(k) != wp.get(k)]
    # host/utc/script differ by construction on a legitimate resume
    ignore = {"utc", "host", "python", "script"}
    bd = [(k, sb.get(k), wb.get(k)) for k in sorted(set(sb) | set(wb))
          if k not in ignore and sb.get(k) != wb.get(k)]
    return pd, bd


def gate_resume(stored_npz, want_identity):
    """The ONE place a stored checkpoint is cleared for skip/resume.

    `stored_npz` is an open npz mapping (or any dict-like with `.files` or
    `__contains__`).  Returns (state, physics_diffs, bookkeeping_diffs) where
    state is RESUME_OK, RESUME_REFUSED_MISMATCH or RESUME_REFUSED_ABSENT.

    A checkpoint that carries no `run_identity` is REFUSED, not trusted: the
    campaign inherited files written before this record existed, and accepting
    them would let a new provenance block claim attribution over a stretch of
    integration it did not produce.
    """
    raw = None
    try:
        if "run_identity" in getattr(stored_npz, "files", []) or \
                "run_identity" in stored_npz:
            raw = str(stored_npz["run_identity"])
    except Exception:                                              # noqa: BLE001
        raw = None
    if not raw:
        return RESUME_REFUSED_ABSENT, [("run_identity", None, "required")], []
    try:
        stored = json.loads(raw)
    except Exception as exc:                                       # noqa: BLE001
        return (RESUME_REFUSED_ABSENT,
                [("run_identity", "unparseable: %s" % exc, "required")], [])
    pd, bd = compare_identity(stored, want_identity)
    if pd:
        return RESUME_REFUSED_MISMATCH, pd, bd
    return RESUME_OK, [], bd


def describe_diffs(diffs, limit=6):
    out = []
    for k, a, b in diffs[:limit]:
        sa, sb = str(a), str(b)
        if len(sa) > 24:
            sa = sa[:21] + "..."
        if len(sb) > 24:
            sb = sb[:21] + "..."
        out.append("%s: stored=%s requested=%s" % (k, sa, sb))
    if len(diffs) > limit:
        out.append("(+%d more)" % (len(diffs) - limit))
    return "; ".join(out)


if __name__ == "__main__":
    p = default_manifest_path()
    man = write_build_manifest(p)
    print("wrote %s" % p)
    print("  link_status       : %s" % man["link_status"])
    print("  blocks_downstream : %s" % man["blocks_downstream"])
    print("  reason            : %s" % man["link_reason"])
    print("  manifest_id       : %s" % man["manifest_id"])
    r = check_build_manifest(p)
    print("  self-check        : %s (%s)" % (r["verdict"], r["reason"]))
