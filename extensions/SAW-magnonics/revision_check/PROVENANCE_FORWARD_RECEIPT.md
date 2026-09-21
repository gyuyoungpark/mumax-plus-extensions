# Forward Build Contract

Current as of the resumed 2026-09-18 correction. This supersedes the linkage
claims in `PROVENANCE_BUILD_MANIFEST.md`; it does not recover historical sim40
build attribution.

`runs/build_forward.py` compiles an isolated engine with `--clean-first` and
records the full enumerated compiled-source hash map before and after the
successful configure/build commands, the CMake cache hash, and output hash.
The source maps must be identical. The receipt is checkpointed after configure
and build; a completed matching build is reused, an incomplete one is rebuilt.

`_provenance.build_manifest()` requires that receipt to match the current source
map, current CMake cache and actually imported (or import-resolved) extension.
A matching binary in a directory plus old source mtimes is not sufficient.
Source timestamps remain diagnostic metadata, not an acceptance criterion.
These are local forward records, not tamper-proof compiler attestations; the
build assumes no transient source mutation between the before/after snapshots.

The new build is isolated at `C:/Users/kist_gyuyoung/SAW-build-20260918`.
`forward_build.json`, `configure.log`, and `build.log` are retained there.
The installed engine and other running GPU work were not replaced or stopped.

Use a fresh process, so an older already-loaded DLL cannot win import resolution:

```powershell
$env:SAW_BUILD_CACHE='C:\Users\kist_gyuyoung\SAW-build-20260918\CMakeCache.txt'
$env:PYTHONPATH='C:\Users\kist_gyuyoung\SAW-build-20260918\lib'
python runs/tests/integration_resume_20260918.py
```

The run identity additionally hashes the interpreted SAW wrappers, shared
harness, and campaign drivers. A Python driver change does not require C++
recompilation, but conservatively invalidates reuse of an old checkpoint.
Identity reads do not rewrite a shared manifest; explicit run-boundary writes
use process-specific temporary files and atomic replacement. Hashes are checked
again at each run boundary rather than trusted from a process-local cache.

Only the recorded T=0, x-directed, Barnett-off execution path is exercised by
the short integration. Kernel y-direction covariance and Barnett-sign issues
remain out of that path, not reasons to block it. Thermal partial restart is
blocked because a spin/time checkpoint alone does not restore the engine RNG.
