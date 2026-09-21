# The build manifest — linking sources and build configuration to the binary

> HISTORICAL IMPLEMENTATION NOTES, superseded during the 2026-09-18 resume.
> The timestamp-based `LINKED_VIA_BUILD_TREE` claims below were insufficient.
> Current acceptance requires an observed clean-build receipt matching source,
> configuration and loaded binary hashes: see `PROVENANCE_FORWARD_RECEIPT.md`.
> The old notes and numerical snapshots remain here as history, not as the
> active build contract. Historical sim40 provenance is still unrecovered.

Written 2026-09-18 in response to section 9 of
`SAW_handover_verification_2026-09-18.md`:

> 소스 해시와 binary 해시를 나란히 적는 것만으로 해당 소스에서 그 binary가
> 만들어졌다는 연결을 증명하지는 않는다. 전체 관련 소스·빌드 설정과 binary를
> 잇는 build manifest가 필요하다. 과거 sim40에 대한 이 연결은 현재도 복구되지
> 않았다.

The auditor is right on both halves, and the two halves have different answers.

* **Forward, from the next run onward**: fixable, and fixed here.
* **Backward, for the existing sim40 data**: **not** fixable. Nothing in this
  document recovers it, and nothing in the code pretends to.

Implementation: `runs/_provenance.py`.
Pinned by: `test_record_corrections.py` sections `[5a]`, `[5b]`, `[5c]`, `[6c]`,
`[6d]`, `[6f]`.

---

## 1. Why the old record did not link anything

`_harness.provenance()` wrote, into the same dict:

```
sha256_chiralsawfield_cu : 56a7d57c...   (a source file)
engine.cpp_module_sha256 : 11a972e4...   (the loaded .pyd)
```

Two true facts, adjacent on the page, with no relation between them. A reader
who rebuilt the kernel from a *different* source, or who had two `.pyd` files
installed, or who edited the `.cu` after building, would get the same shape of
record with the same confident look. Adjacency is not causation.

## 2. What actually links them

Three artefacts on disk connect the source tree to the loaded binary. None of
them is a hash written next to another hash.

1. **The CMake build tree** that produced the binary records, in its
   `CMakeCache.txt`, *which source tree it was configured against*
   (`CMAKE_HOME_DIRECTORY`), the floating-point precision the binary was built
   for (`FP_PRECISION`), the CUDA architectures, the generator and the compiler
   flags. On this machine, for the single-precision build:

   ```
   cmake_cache_path      D:/mumax-plus-dev/build/temp.win-amd64-cpython-314/Release/single/CMakeCache.txt
   cmake_home_directory  D:/mumax-plus-dev
   fp_precision          SINGLE
   cuda_architectures    52
   generator             Visual Studio 17 2022  (x64)
   cxx_flags_release     /O2 /Ob2 /DNDEBUG
   ```

2. **That build tree still holds the artefact it emitted**, in the output
   directory *the selected cache itself declares*
   (`CMAKE_LIBRARY_OUTPUT_DIRECTORY` = `build/lib.win-amd64-cpython-314` here).

   **CORRECTED 2026-09-18 (round 2).** `_build_tree_binary()` used to discard
   its `cache_path` argument and glob `REPO/build/lib.*`, so a cache from one
   tree and a binary from another still reported `LINKED_VIA_BUILD_TREE`; and
   selection among the four build trees in this repo was by `CMakeCache.txt`
   mtime, which a second `FP_PRECISION=SINGLE` tree (`build_test_single/`, whose
   declared output directory does not exist) could win. Now the binary is
   derived from the selected cache's own output directory, a tree that emitted
   nothing is not selectable at all (and is listed with its reason under
   `build_config.build_trees_excluded`), and the tree whose output is
   byte-identical to the module python imports is the one selected
   (`build_config.selected_because`). If the output cannot be derived the
   verdict is `CONSISTENT_BUT_UNPROVEN`, which blocks.

3. **The installed binary is byte-identical to it.** Measured, not assumed:

   ```
   build/lib.win-amd64-cpython-314/_mumaxpluscpp_single.cp314-win_amd64.pyd
       sha256 11a972e4fc64bcc6097ad0cea645a42c6b2d75a5848012851954fcc80cf0740c  5321216 B
   <site-packages>/_mumaxpluscpp_single.cp314-win_amd64.pyd
       sha256 11a972e4fc64bcc6097ad0cea645a42c6b2d75a5848012851954fcc80cf0740c  5321216 B
   ```

   Equal hashes *between a build output and an installed file* are a link, in a
   way that equal-length hashes of a source and a binary never are: the
   installed file **is** this build tree's product.

The manifest records all three, plus every source file whose change would
change the integrated physics, plus the check that **no recorded source is
newer than the binary**.

## 3. The four link states

`link_status` is never a boolean and there is no fifth state. `blocks_downstream`
is `False` in exactly one of them.

| state | meaning | blocks? |
|---|---|---|
| `LINKED_VIA_BUILD_TREE` | the installed binary is byte-identical to this build tree's output, that tree was configured against this source tree, and no recorded source is newer than the binary | no |
| `REFUTED` | one of those three failed — a different binary is installed, CMake was configured against another tree, or a source is newer than the binary | **yes** |
| `CONSISTENT_BUT_UNPROVEN` | reserved for a record that agrees everywhere it can be compared but is missing one leg of the chain | **yes** |
| `NOT_DETERMINABLE` | the installed binary, the build-tree artefact or the CMake cache is not on this machine | **yes** |

`check_build_manifest()` returns its own three-way verdict — `MATCH`,
`MISMATCH`, `NOT_DETERMINABLE` — and a **missing manifest is
`NOT_DETERMINABLE` and blocks**. It is never read as "nothing to check,
therefore fine".

## 4. What the link proves, and what it does not

**It proves**, for the run being recorded:

* the installed binary is the artefact a CMake build tree on this machine
  emitted, byte for byte;
* that tree was configured against this source tree, at this precision, with
  these flags and this CUDA architecture;
* the recorded source files hashed to these values at the moment of the record,
  and none of them had a modification time later than the binary's.

**It does not prove**:

* that the recorded source files are the ones the compiler *read*. The build
  tree's object files are not re-hashed against their translation units here, so
  a source edited and re-saved with an older timestamp would not be caught.
  Modification times are filesystem metadata and can be set arbitrarily; they
  are corroboration, not evidence.
* ~~that the source list is complete~~ — **CORRECTED 2026-09-18 (round 2).**
  The sentence this bullet used to carry ("it is explicit and covers the chiral
  SAW drive, ... widening the list is a one-line change") described the defect
  rather than a limitation. The explicit list held 18 paths against 131 files in
  `src/physics` and did not contain `magnetoelasticfield.cu`, the `B_1` kernel
  every one of these runs integrates with `enable_mel=True`; nor `exchange.cu`,
  `anisotropy.cu`, `zeeman.cu`, `demag.cpp`, `thermalnoise.cu` or `minimizer.cu`.
  A digest built from a hand-written list cannot move when a kernel outside the
  list is edited, so the resume gate accepted a stale checkpoint *by
  construction*. Nothing is enumerated now: `compiled_source_files()` walks
  `src/physics`, `src/core`, `src/bindings`, `src/cudautil`, `src/linsolver` and
  `src/cmd` plus the build description (334 recorded files on this machine, 331
  of them compiled), and `build_tree_object_sources()` reads the translation
  units out of the selected build tree's own project files (111 on this machine)
  and reports any the glob missed, which makes the link
  `CONSISTENT_BUT_UNPROVEN` instead of leaving the gap silent.
* anything about the *correctness* of the kernel. The manifest is provenance,
  not validation. The MR coefficient check the auditor ran
  (5,000 finite-difference and complex-step comparisons, max relative error
  4.61e-15) is a separate and independent thing.
* that the compiler and CUDA toolkit versions are what they were, where the
  cache does not name them. Whatever the cache *does* name is now compared:
  `build_type`, `cxx_compiler` and `cuda_compiler` were recorded in round 1 and
  left out of `_CHECK_CONFIG`, so forging any of them returned `MATCH`. The
  compared set is now derived from the recorded configuration itself, so a field
  that is added to the manifest cannot be silently unchecked.
* **anything backwards.** Everything here is FORWARD provenance: it can refuse a
  mismatched artifact from now on. It does not recover the conditions of any
  artifact written before it existed, and in particular the source-to-binary
  link of the existing sim40 data stays unrecoverable (section 5). A record with
  no condition stamp is therefore refused, not trusted.

## 5. The backward direction: sim40 stays unrecoverable

The nine `data/sim40_checkpoints/*.npz` files used throughout this revision were
written in May 2026 by `src/sim40_eps_kresolved.py`. They carry no provenance
block at all (see `README_HANDOVER.md` for their complete key list), and the
build that produced them is gone: the current `_mumaxpluscpp_single*.pyd` was
rebuilt on 2026-09-17, and the `.old_*` and `.INUSE_*` copies beside it are
dated 2026-06-08 and 2026-06-11 — all of them *after* the data.

Therefore:

> **The link from the sim40 data to the binary that produced it is
> unrecoverable, and this manifest does not recover it.** No re-derivation from
> modification times, file sizes or present-day hashes can establish it, and
> none is attempted. The existing sim40 results are used in this revision as
> data whose *analysis* is reproducible from the stored `my_xt` arrays, not as
> results with attributable provenance.

The manifest changes what is true of runs made **from now on**, and only that.

## 6. How the record participates in skip/resume

Writing a provenance record that nothing reads is the same failure one level up.
So the forward record is part of the identity that decides whether a checkpoint
may be reused.

The run-manifest gate itself is `runs/_gate.py` + `_harness.BlockRun` (audit
section 8, P0-4). This work does **not** add a second comparison beside it.
`_provenance.run_manifest_fields()` returns

```
sha256_chiralsawfield_cu   engine_binary_sha256   source_tree_digest
fp_precision               build_link_status
```

and `BlockRun.__init__` folds them into the reserved part of the run manifest.
They are then hashed and diffed by the one existing mechanism, so a rebuilt
kernel is refused by the *same* code path that refuses a changed `dt_rec`:

```
RunIdentityMismatch: block.npz was recorded for a DIFFERENT run: 1 field(s)
differ (stored -> requested)
    sha256_chiralsawfield_cu: 'ffff...' -> '56a7d57c...'
```

Anything extending this should add fields to `run_manifest_fields()`, never a
parallel check.

`build_manifest_check` is deliberately **not** in the run manifest. It records
this process's comparison against the previously stored manifest — useful in the
provenance block, but it is a statement about the machine's history, not about
the run's identity, and putting it in the manifest would invalidate every
checkpoint after any unrelated rebuild.

## 7. Running it

```
python revision_check/runs/_provenance.py
```

writes `runs/out/build_manifest.json`, prints the link status and reason, and
self-checks. `_harness.provenance()` calls the same code on every campaign
output, so each npz now carries `build_manifest_id`, `build_link_status`,
`build_manifest_check` and `source_tree_digest` alongside the old hashes.

Observed on this machine, 2026-09-18:

```
link_status       : LINKED_VIA_BUILD_TREE
blocks_downstream : False
reason            : the installed binary is byte-identical to the artefact this
                    build tree emitted, that tree was configured against this
                    source tree, and no recorded source is newer than the binary
self-check        : MATCH
```

One source in the list, `src/physics/energy.cu`, is present; there is no
`energy.cpp` in this tree, and the manifest keeps any missing path in the record
with a `NOT DETERMINABLE` hash rather than dropping it, so a missing file cannot
make the manifest look cleaner than it is.


## 8. Round 2 (2026-09-18): the manifest is one half of ONE mechanism

The build manifest no longer stands on its own. Its five facts —
`source_tree_digest`, `compiled_source_digest`, `build_config_digest`,
`engine_binary_sha256`, `fp_precision` and `build_link_status` — are the `build`
group of the condition set in `runs/_conditions.py`, which is the single
description of "what a run was performed under" that certificates, checkpoints
and summaries all carry and that every stage compares. `blocks_downstream` is
read by that mechanism (`ConditionSet.build_link_blocks`), so a link that is not
`LINKED_VIA_BUILD_TREE` blocks a stage that asks for the build group instead of
being a recorded string. `PRECONDITIONS_ROUND2.md` describes the whole of it.

Observed on this machine after the round-2 changes, 2026-09-18:

```
link_status       : LINKED_VIA_BUILD_TREE
blocks_downstream : False
selected_because  : its own output is byte-identical to the installed module
n_sources         : 334   (331 compiled, 3 interpreted)
object sources    : 111, read from 26 vcxproj + 121 object files; 0 missed by the glob
excluded trees    : build_test_single/ and build_test/ (emitted no module in
                    their own declared output directories)
```

Staleness is asked of the compiled sources only. Editing `saw_chiral.py` does
not make the `.pyd` stale, and treating it as stale would refuse every
checkpoint after a comment edit; the interpreted sources are still hashed into
`source_tree_digest`, so they still move the run identity.
