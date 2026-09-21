"""Conditional G9 estimator calibration and independent synthetic validation.

All arrays are analytical test signals, not engine or material results.
Both calibration and validation call G9.pair_rates and G9.rate_comparison.
The finite-set allowance is fitted only on calibration seeds. Held cases are
counted separately. Validation counts are conditional synthetic compatibility
counts, never physical false-positive/false-negative rates.

--output names a NEW result. A sibling checkpoint resumes after each record.
--write writes a proposed criteria JSON beside that result, never PREREGISTRATION.
"""
import argparse
import copy
import datetime
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _criteria as CRIT
import G9_idler_injection as G9

DT_REC = G9.DT_REC
F_SAW = G9.F_SAW
N = G9.N


def synth_pair(nx, dx, nt, j, gamma, f1_target, amp_ratio, noise_rel, seed,
               offset_bins=0.0, gamma_ratio=1.0):
    """SYNTHETIC commensurate waves, including a declared frequency-bin offset."""
    dk = 2 * np.pi / (nx * dx)
    x = (np.arange(nx) + 0.5) * dx
    t = np.arange(nt) * DT_REC
    df = 1.0 / ((nt - nt // 2) * DT_REC)
    f1 = (round(f1_target / df) + offset_bins) * df
    f2 = F_SAW - f1
    amp = 1e-4
    m = amp * np.exp(gamma*t)[:, None] * np.cos(
        j*dk*x[None, :] - 2*np.pi*f1*t[:, None] + 0.3)
    m += amp*amp_ratio*np.exp(gamma*gamma_ratio*t)[:, None] * np.cos(
        (N-j)*dk*x[None, :] - 2*np.pi*f2*t[:, None] - 0.3)
    m += amp*noise_rel*np.random.default_rng(seed).standard_normal(m.shape)
    return m.astype(np.float32), dk


def one(nx, dx, nt, j, gamma, f1, amp_ratio, noise_rel, seed,
        offset_bins=0.0, gamma_ratio=1.0):
    my, dk = synth_pair(nx, dx, nt, j, gamma, f1, amp_ratio, noise_rel,
                        seed, offset_bins, gamma_ratio)
    pr = G9.pair_rates(my, dx, DT_REC, j*dk, (N-j)*dk, 2*np.pi*F_SAW)
    # Calibration reads the statistical term through the production rule with
    # a declared zero systematic term. It does not implement another estimator.
    spec = copy.deepcopy(G9.CRITERIA)
    floor = spec["growth_rate_compatibility"]["systematic_relative_floor"]
    floor.update(value=0.0, estimator_sha256=CRIT.estimator_fingerprint(),
                 calibrated_K_SIGMA=spec["growth_rate_compatibility"]["K_SIGMA"])
    cmp = G9.rate_comparison(pr, spec)
    return dict(nt=nt, j=j, gamma=gamma, f1_target=f1, amp_ratio=amp_ratio,
                noise_rel=noise_rel, seed=seed, offset_bins=offset_bins,
                gamma_ratio=gamma_ratio, pair_rates=pr,
                eligible=cmp["compatible"] != CRIT.NOT_DETERMINABLE,
                rel_diff=cmp["relative_difference"],
                rel_stat=(cmp["sigma_stat"]/max(abs(pr["gamma_signal"]),
                                               abs(pr["gamma_idler"]))
                          if cmp["compatible"] != CRIT.NOT_DETERMINABLE else float("nan")))


def configurations(seeds):
    base = dict(nt=1500, gamma=1.52e8, amp_ratio=0.01, f1=2e9, j=3,
                noise_rel=1e-4, seed=seeds[0])
    axes = dict(nt=(750, 1500, 3000), gamma=(1e8, 1.52e8, 3e8),
                amp_ratio=(0.01, 0.1, 1.0), noise_rel=(1e-6, 1e-4, 1e-2),
                seed=seeds)
    configs = [dict(base, **dict(zip(axes, vals)))
               for vals in itertools.product(*axes.values())]
    for key, vals in (("f1", (2.5e9, 3e9)), ("j", (4, 5))):
        for val in vals:
            configs.append(dict(base, **{key: val}))
    return configs


def summarise(rows, K=None):
    """Finite calibration coverage requirement; not a measured variance."""
    K = float(K if K is not None else G9.CRITERIA["growth_rate_compatibility"]["K_SIGMA"])
    good = [r for r in rows if r["eligible"] and
            np.isfinite(r["rel_diff"]) and np.isfinite(r["rel_stat"])]
    if not good:
        raise RuntimeError("no eligible calibration fits; cannot measure a floor")
    need = [np.sqrt(max(0.0, (r["rel_diff"]/K)**2-r["rel_stat"]**2)) for r in good]
    # Round upward so floating-point evaluation does not reject the boundary.
    floor = float(np.nextafter(max(need), np.inf))
    return dict(K=K, n_total=len(rows), n_eligible=len(good),
                n_indeterminate=len(rows)-len(good), f_sys=floor,
                raw_max=max(r["rel_diff"] for r in good))


def counts(rows, spec):
    result = dict(total=len(rows), compatible=0, incompatible=0, indeterminate=0)
    for r in rows:
        cmp = G9.rate_comparison(r["pair_rates"], spec)
        r["comparison"] = cmp
        key = ("indeterminate" if cmp["compatible"] == CRIT.NOT_DETERMINABLE
               else "compatible" if cmp["compatible"] is True else "incompatible")
        result[key] += 1
    return result


def save_atomic(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)
        fh.write("\n")
    # Windows scanners can briefly hold an otherwise writable checkpoint.
    for attempt in range(6):
        try:
            os.replace(path + ".tmp", path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="new G9-focused result JSON")
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--write", action="store_true", help="emit proposed criteria beside results")
    a = ap.parse_args()
    if a.nx <= 2*N:
        ap.error("nx must resolve all synthetic grid modes")
    out = os.path.abspath(a.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit("preserving existing result: " + out)
    batches = [("calibration", configurations((7, 11))),
               ("validation_equal", configurations((101, 103)))]
    base = dict(nt=1500, gamma=1.52e8, amp_ratio=0.01, f1=2e9, j=3,
                noise_rel=1e-4, seed=107)
    batches.append(("validation_offgrid", [dict(base, offset_bins=v) for v in (0.23, 0.47)]))
    batches.append(("validation_unequal", [dict(base, gamma_ratio=v) for v in (0.99, 0.91)]))
    jobs = [(group, cfg) for group, configs in batches for cfg in configs]
    with open(__file__, "rb") as fh:
        script_sha = hashlib.sha256(fh.read()).hexdigest()
    identity = dict(estimator_sha256=CRIT.estimator_fingerprint(), script_sha256=script_sha,
                    nx=a.nx, dt_rec=DT_REC, dx=G9.BOX["dx"], jobs=jobs,
                    K_SIGMA=G9.CRITERIA["growth_rate_compatibility"]["K_SIGMA"],
                    fit_applicability=G9.CRITERIA["growth_rate_compatibility"]["fit_applicability"])
    # JSON normalization makes tuples stable over a checkpoint reload.
    identity = json.loads(json.dumps(identity))
    ckpt = out + ".checkpoint.json"
    payload = dict(identity=identity, synthetic=True, rows=[], next_index=0)
    if os.path.exists(ckpt):
        with open(ckpt, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload["identity"] != identity or payload["next_index"] != len(payload["rows"]):
            raise SystemExit("checkpoint differs from this estimator/configuration")
    for idx in range(payload["next_index"], len(jobs)):
        group, cfg = jobs[idx]
        row = one(a.nx, G9.BOX["dx"], **cfg)
        row["group"] = group
        payload["rows"].append(row)
        payload["next_index"] = idx+1
        save_atomic(ckpt, payload)
        if (idx+1) % 25 == 0:
            print("%d/%d synthetic records" % (idx+1, len(jobs)), flush=True)
    rows = payload["rows"]
    summary = summarise([r for r in rows if r["group"] == "calibration"])
    spec = copy.deepcopy(G9.CRITERIA)
    floor = spec["growth_rate_compatibility"]["systematic_relative_floor"]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    floor.update(value=summary["f_sys"], n_controls=summary["n_eligible"],
                 measured_utc=now, estimator_sha256=identity["estimator_sha256"],
                 calibrated_K_SIGMA=summary["K"],
                 evidence_file=os.path.relpath(out, CRIT.RC).replace("\\", "/"),
                 how_measured="G9.pair_rates and G9.rate_comparison with shared GI fits; "
                              "%d/%d calibration records eligible. Calibration seeds 7,11; "
                              "independent validation seeds 101,103,107. Reduced nx=%d; "
                              "transfer to the campaign grid and physical noise is unvalidated."
                              % (summary["n_eligible"], summary["n_total"], a.nx))
    payload.update(measured_utc=now, summary=summary,
                   interpretation="Conditional synthetic estimator counts; no physical error-rate claim.",
                   counts={group: counts([r for r in rows if r["group"] == group], spec)
                           for group, _ in batches})
    floor["validation_counts"] = payload["counts"]
    save_atomic(out, payload)
    if a.write:
        proposal = out + ".criteria.json"
        if os.path.exists(proposal):
            raise SystemExit("preserving existing criteria proposal: " + proposal)
        save_atomic(proposal, spec)
        print("Criteria proposal: " + proposal)
        print("After adopting the proposal, main must run: python runs/_criteria.py render")
    print(json.dumps(dict(summary=summary, counts=payload["counts"]), indent=2))


if __name__ == "__main__":
    main()
