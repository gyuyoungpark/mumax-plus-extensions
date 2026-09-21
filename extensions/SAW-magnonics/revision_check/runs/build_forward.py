"""Compile an isolated engine and record a forward-only source/build receipt."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import _provenance as P

SCHEMA = "saw_forward_build/1"


def snapshot():
    return {P._rel(p): P.sha256_file(p) for p in P.compiled_source_files()}


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-dir", required=True)
    args = parser.parse_args()
    root = Path(args.build_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    record = root / "forward_build.json"
    before = snapshot()
    if record.exists():
        old = json.loads(record.read_text(encoding="utf-8"))
        binary = Path(old.get("binary_path", "missing"))
        if (old.get("status") == "complete" and old.get("sources_after") == before
                and binary.is_file() and old.get("binary_sha256") == P.sha256_file(binary)
                and old.get("cache_sha256") == P.sha256_file(root / "CMakeCache.txt")):
            print("Already complete: " + str(record), flush=True)
            return
    elif any(root.iterdir()):
        raise SystemExit("Refusing an existing build tree without a forward record")
    cmake = shutil.which("cmake") or str(Path(sys.executable).parent / "Scripts" / "cmake.exe")
    output = root / "lib"
    configure = [cmake, "-S", P.REPO, "-B", str(root), "-G", "Visual Studio 17 2022",
                 "-A", "x64", "-DFP_PRECISION=SINGLE", "-DMUMAX_MODULE_NAME=_mumaxpluscpp_single",
                 "-DCMAKE_CUDA_ARCHITECTURES=89", "-DPYTHON_EXECUTABLE=" + sys.executable,
                 "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + str(output),
                 "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=" + str(output)]
    build = [cmake, "--build", str(root), "--config", "Release", "--clean-first", "--parallel", "4"]
    state = dict(schema=SCHEMA, status="configuring", repo=str(Path(P.REPO).resolve()),
                 started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 sources_before=before, configure_command=configure, build_command=build)
    write(record, state)
    for name, command in (("configure", configure), ("build", build)):
        state["status"] = name
        write(record, state)
        print(name + ": " + subprocess.list2cmdline(command), flush=True)
        with (root / (name + ".log")).open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        state[name + "_returncode"] = proc.returncode
        write(record, state)
        if proc.returncode:
            raise SystemExit("Failed " + name + "; see " + str(root / (name + ".log")))
    after = snapshot()
    if after != before:
        state["status"] = "source_changed_during_build"
        write(record, state)
        raise SystemExit("Sources changed during compilation; no receipt issued")
    cache = root / "CMakeCache.txt"
    binary = P._build_tree_binary(str(cache), "single")
    if not binary:
        raise SystemExit("Build returned success without an extension module")
    state.update(status="complete", sources_after=after, cache_sha256=P.sha256_file(cache),
                 binary_path=str(Path(binary).resolve()), binary_sha256=P.sha256_file(binary),
                 finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    write(record, state)
    print("Forward receipt: " + str(record), flush=True)
    print("Binary: " + binary, flush=True)


if __name__ == "__main__":
    main()
