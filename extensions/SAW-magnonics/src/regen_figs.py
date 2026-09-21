"""Regenerate sim11 figures only (no simulation, uses cached data)."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('..', 'src'))
sys.path.insert(0, '..')
import matplotlib
matplotlib.use('Agg')
import numpy as np

from sim11_angle_dependence import (
    part_a_analytic, fig_coupling_vs_angle,
    fig_material_comparison, fig_simulation_validation
)

print("Computing analytic results...")
results = part_a_analytic()

print("Generating fig_angle_coupling...")
fig_coupling_vs_angle(results)

print("Generating fig_angle_materials...")
fig_material_comparison(results)

print("Generating fig_angle_validation...")
sim_data = dict(np.load('../data/sim11_angle_sweep.npz'))
fig_simulation_validation(sim_data, results)

print("All figures regenerated!")
