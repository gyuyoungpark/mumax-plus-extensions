"""Read-only diagnosis of G4's infinite-film Kittel reference.

Static engine evaluations only. No campaign certificate is issued or changed.
Checkpoints preserve each demag axis / linear-operator column and reject a
different source, binary, input, or perturbation amplitude on resume.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RC = HERE.parent
sys.path.insert(0, str(RC / "runs"))
import _harness as H


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(path, identity, **data):
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, identity=json.dumps(identity, sort_keys=True), **data)
    os.replace(tmp, path)


def resume(path, identity):
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved["identity"])) != identity:
            raise RuntimeError(f"Checkpoint identity mismatch: {path}")
        return {k: saved[k].copy() for k in saved.files if k != "identity"}


def measure(pbc_y, B0, perturbation):
    H.PBC = (2, pbc_y, 0)
    box = H.BOX_PRIMARY
    decl = dict(box=box, material=H.YIG_LIT, B0=B0, eps0=0.0,
                f_saw=6e9, temperature=0.0, rng_seed=0, a_seed=perturbation,
                k_cut=0.0)
    cs = H.conditions(decl, source="static k0 dispersion diagnosis",
                      nt=2, dt_rec=1e-12, block_records=2)
    cs.require_execution("static k0 diagnosis", nx=box["NX"])
    identity = dict(script_sha256=sha(__file__), declaration=decl,
                    condition_stamp=cs.stamp(), perturbation=perturbation,
                    certificate_sha256=sha(RC / "runs/out/G4/G4_B0star.npz"))
    path = HERE / (f"reference_pbc{pbc_y}_B{B0:.12f}_h{perturbation:.0e}.npz")
    saved = resume(path, identity)
    if bool(saved.get("done", False)):
        return json.loads(str(saved["result"]))
    mag0 = np.zeros((3, H.NZ, H.NY, box["NX"]))
    mag0[0] = 1.0
    world, magnet, _ = H.build(box, H.YIG_LIT, B0, 0.0, 6e9, mag0, conditions=cs)
    demag = saved.get("demag", np.full((3, 3, H.NY), np.nan))
    matrix = saved.get("matrix", np.full((2 * H.NY, 2 * H.NY), np.nan))
    leak = saved.get("leak", np.full(2 * H.NY, np.nan))
    for axis in range(3):
        if np.isfinite(demag[axis]).all():
            continue
        state = np.zeros_like(mag0)
        state[axis] = 1
        magnet.magnetization = state
        field = magnet.demag_field.eval().astype(float)
        demag[axis] = field.mean(axis=(1, 3)) / (-H.MU0 * H.YIG_LIT["MS"])
        checkpoint(path, identity, demag=demag, matrix=matrix, leak=leak, done=False)
    magnet.magnetization = mag0
    base_torque = magnet.llg_torque.eval().astype(float)
    for column in range(2 * H.NY):
        if np.isfinite(matrix[:, column]).all():
            continue
        component, row = 1 + column // H.NY, column % H.NY
        responses = []
        for sign in (1, -1):
            state = mag0.copy()
            state[component, 0, row, :] = sign * perturbation
            state /= np.linalg.norm(state, axis=0)
            magnet.magnetization = state
            responses.append(magnet.llg_torque.eval().astype(float))
        derivative = (responses[0] - responses[1]) / (2 * perturbation)
        projected = derivative[1:, 0].mean(axis=-1)
        matrix[:, column] = projected.ravel()
        remainder = derivative[1:, 0] - projected[..., None]
        leak[column] = np.linalg.norm(remainder) / np.linalg.norm(derivative[1:, 0])
        checkpoint(path, identity, demag=demag, matrix=matrix, leak=leak, done=False)
    values, vectors = np.linalg.eig(matrix)
    candidates = np.flatnonzero(values.imag > 0)
    weights = np.array([sum(abs(vectors[c*H.NY:(c+1)*H.NY, j].sum())**2
                           for c in (0, 1)) / H.NY for j in candidates])
    mode = candidates[np.argmax(weights)]
    N = np.array([demag[c, c].mean() for c in range(3)])
    P = B0 + H.MU0 * H.YIG_LIT["MS"] * (N[1] - N[0])
    Q = B0 + H.MU0 * H.YIG_LIT["MS"] * (N[2] - N[0])
    result = dict(pbc=list(H.PBC), B0=B0, perturbation=perturbation,
                  diagonal_N=N.tolist(), N_trace=float(N.sum()),
                  kittel_infinite_Hz=float(H.kittel_freq(B0, H.YIG_LIT["MS"])),
                  kittel_mean_tensor_Hz=float(H.GAMMA * np.sqrt(P*Q)/(2*np.pi)),
                  projected_llg_frequency_Hz=float(values[mode].imag/(2*np.pi)),
                  projected_llg_decay_per_s=float(values[mode].real),
                  uniform_overlap=float(weights.max()),
                  max_column_projection_leakage=float(leak.max()),
                  equilibrium_torque_rms_per_s=float(np.sqrt(np.mean(base_torque**2))),
                  gamma_engine=float(np.mean(magnet.gamma.eval())),
                  caveat="k_x=0 projection retains all y rows; not a pumped instability test")
    checkpoint(path, identity, demag=demag, matrix=matrix, leak=leak,
               eigenvalues=values, result=json.dumps(result), done=True)
    return result


def main():
    with np.load(RC / "runs/out/G4/G4_B0star.npz", allow_pickle=False) as data:
        history = json.loads(str(data["history"]))
    cases = [(2, h["B0"], 1e-3) for h in history]
    cases += [(2, history[-1]["B0"], 1e-4),
              (8, history[-1]["B0"], 1e-3),
              (32, history[-1]["B0"], 1e-3)]
    results = []
    for case in cases:
        result = measure(*case)
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(purpose="check the reference that halted G4; no pairing verdict",
                  script_sha256=sha(__file__), recorded_dispersion=history, results=results)
    tmp = HERE / "dispersion_reference.tmp"
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(tmp, HERE / "dispersion_reference.json")


if __name__ == "__main__":
    main()
