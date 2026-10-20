"""Explicit classical energy API, distributed as a derived source-only port.

The original frozen sources had finite mathematical/software controls and one
lossless uniform CPU FM pilot with two references. Edited public source hashes
have not been physically rerun. Native, AFM, spatial, multimode and dissipative
implementation validation is not implied by that original finite evidence.

Importing this package imports only Python standard-library code. It neither
loads mumax+, numpy/scipy, nor runs a model, test, fit or simulation. This is not
a drop-in replacement for the old CavityMagnon / CavityMagnonAFM APIs.
"""
from .energy import (
    EnergyCavityModel, EnergyMode, MagneticSource, OverlapSnapshot,
    QuadratureRate, Quadratures, UnsupportedCoupling, reject_legacy_coupling,
)
from .normalization import FmRwaMapping, fm_rwa_mapping, moment_from_homogeneous_material

__all__ = [
    "EnergyCavityModel", "EnergyMode", "MagneticSource", "OverlapSnapshot",
    "QuadratureRate", "Quadratures", "UnsupportedCoupling", "reject_legacy_coupling",
    "FmRwaMapping", "fm_rwa_mapping", "moment_from_homogeneous_material",
]
