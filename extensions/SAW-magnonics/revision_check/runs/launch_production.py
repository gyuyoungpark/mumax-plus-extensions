"""Run the existing G4 stages sequentially, preserving logs and gate failures."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--arm", choices=("YIGlit", "sim40set"), default="YIGlit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    job = Path(args.job_dir).resolve()
    job.mkdir(parents=True, exist_ok=True)
    build = Path(args.build_dir).resolve()
    env = dict(os.environ, SAW_BUILD_CACHE=str(build / "CMakeCache.txt"),
               PYTHONPATH=str(build / "lib"), PYTHONUNBUFFERED="1")
    state = dict(pid=os.getpid(), arm=args.arm, build_dir=str(build),
                 output_dir=str(root / "runs" / "out" / "G4"), stages=[])

    def update(**fields):
        state.update(fields, updated_utc=datetime.now(timezone.utc).isoformat())
        temporary = job / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, job / "status.json")

    def run_stage(name, arguments):
        command = [sys.executable, "-u", *arguments]
        log_path = job / (name + ".log")
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\nCOMMAND: " + subprocess.list2cmdline(command) + "\n")
            log.flush()
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
            update(status="running", stage=name, child_pid=child.pid,
                   current_log=str(log_path))
            print(name, "started", child.pid, flush=True)
            result = child.wait()
        state["stages"].append(dict(stage=name, returncode=result, log=str(log_path)))
        update(child_pid=None)
        if result:
            raise RuntimeError("%s exited %d; see %s" % (name, result, log_path))

    try:
        update(status="starting", stage="preflight", child_pid=None)
        run_stage("preflight", ["-c",
            "import sys,json;sys.path.insert(0,'runs');import _provenance as p;"
            "m=p.write_build_manifest(" + repr(str(job / "build_manifest.json")) + ");"
            "print(json.dumps({k:m[k] for k in ('link_status','link_reason','manifest_id','blocks_downstream')},indent=2));"
            "sys.exit(2 if m['blocks_downstream'] else 0)"])
        # The smoke test runs in its own output tree; production parameters are unchanged.
        smoke_root = job / "smoke"
        smoke_root.mkdir(exist_ok=True)
        run_stage("engine_smoke", ["-c",
            "import sys;sys.path.insert(0,'runs/tests');"
            "import integration_resume_20260918 as t;"
            "t.H.OUT_DIR=" + repr(str(smoke_root)) + ";t.main()"])
        for stage in ("pump", "disp", "onset", "analyze"):
            run_stage(stage, ["runs/G4_commensurate_onset.py", stage,
                              "--arm", args.arm])
        run_stage("result_gate", ["-c",
            "import numpy as np,sys;"
            "d=np.load('runs/out/G4/G4_summary.npz',allow_pickle=False);"
            "s=str(d['gate_state']);print('G4.3',s);"
            "sys.exit(0 if s=='PASS' else 2)"])
        update(status="complete", stage="complete", child_pid=None)
        return 0
    except Exception as exc:
        update(status="stopped", reason=str(exc), child_pid=None)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
