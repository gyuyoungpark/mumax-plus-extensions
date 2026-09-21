"""Small uniform-Zeeman AFM/cavity ringdown using the native GPU solver.

Build this fork, then run from its root with double precision:
    python examples/afm_cavity_ringdown.py --mumaxplus-fp-precision DOUBLE

The cavity variable stores energy-normalized real quadratures (q,p,0),
not a photon annihilation amplitude. This demonstration is not an EP fit.
No data files are read or written unless --output is explicitly supplied.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import mumaxplus as mp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-ps', type=float, default=20.0)
    parser.add_argument('--output', type=Path, help='Optional new CSV output path')
    parser.add_argument('--mumaxplus-fp-precision', default='DOUBLE')
    args = parser.parse_args()
    if mp.FP_PRECISION != 'DOUBLE':
        parser.error('This example requires a DOUBLE build/precision selection.')
    if not np.isfinite(args.duration_ps) or args.duration_ps <= 0:
        parser.error('--duration-ps must be finite and positive')
    if args.output and args.output.exists():
        parser.error('--output already exists; choose a new path')

    gamma, ms, lattice = 1.7595e11, 624e3, 3.13e-10
    be = 4 * 0.84e-12 / (ms * lattice**2)
    ba = 2 * 245.4e3 / ms
    world = mp.World(cellsize=(40e-9, 40e-9, 10e-9))
    afm = mp.Antiferromagnet(world, mp.Grid((1, 1, 1)))
    for sub, direction in ((afm.sub1, 1), (afm.sub2, -1)):
        sub.msat, sub.gamma, sub.alpha = ms, gamma, 0.001
        sub.aex, sub.ku1, sub.anisU = 0.0, ba * ms / 2, (0, 0, 1)
        sub.enable_demag, sub.enable_openbc = False, True
        sub.magnetization = (0, 0, direction)
    afm.afmex_cell, afm.afmex_nn, afm.latcon = -be * ms * lattice**2 / 4, 0.0, lattice
    impl = afm._impl
    if not hasattr(impl, 'cavity_energy_field'):
        raise RuntimeError('Rebuild this fork: the loaded module lacks the energy-derived cavity.')
    impl.cavity_energy_field = be
    impl.cavity_omega = gamma * np.sqrt(ba * (2 * be + ba))
    impl.cavity_kappa = impl.cavity_omega / 1000  # Q=500, viscous p damping
    impl.cavity_h0 = 0.1  # tesla per unit q; shared by forward/backward coupling
    impl.enable_cavity_afm = True
    impl.enable_aux_mode = False
    cavity = mp.Variable(impl.cavity_amplitude())
    cavity.set((1e-6, 0, 0))
    world._impl.reset_timesolver_equations()
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = 12.5e-15
    count = max(1, int(round(args.duration_ps * 1e-12 / (16 * 12.5e-15))))
    rows, energies = [], []
    for i in range(count + 1):
        if i:
            world.timesolver.steps(16)
        a = np.asarray(afm.sub1.magnetization.average())
        b = np.asarray(afm.sub2.magnetization.average())
        q, p, _ = cavity.average()
        mx = a[0] + b[0]
        # Total coupled energy above collinear equilibrium, divided by ms*V*BE.
        # The library's spin total_energy quantity does not include cavity energy.
        energy = (0.5 * np.dot(a + b, a + b)
                  + ba / (2 * be) * (np.dot(a[:2], a[:2]) + np.dot(b[:2], b[:2]))
                  + q*q + p*p - impl.cavity_h0 / be * q * mx)
        rows.append((world.timesolver.time, q, p, mx))
        energies.append(energy)
    data, energies = np.asarray(rows), np.asarray(energies)
    if not np.isfinite(data).all() or not np.isfinite(energies).all():
        raise RuntimeError('Nonfinite GPU trajectory')
    print(json.dumps({'samples': len(data), 'duration_s': float(data[-1, 0]),
                      'max_abs_Mx': float(np.max(np.abs(data[:, 3]))),
                      'coupled_energy_final_over_initial': float(energies[-1] / energies[0]),
                      'precision': mp.FP_PRECISION}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            np.savetxt(stream, data, delimiter=',', header='t_s,q,p,Mx', comments='')


if __name__ == '__main__':
    main()
