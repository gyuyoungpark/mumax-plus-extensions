#pragma once

#include <cmath>
#include <stdexcept>

// Host-only input validation. This file has no CUDA, integration, or model
// evaluation dependency, so its refusal paths can be tested without a GPU.
// It does NOT test positive definiteness of the coupled spin-cavity energy.
namespace cavity {
struct UniformAfmConfiguration {
  double omega, kappa, field, energyField, driveQ, driveP;
  bool auxiliary;
  double auxOmega, auxKappa, auxField, auxJ;
};

inline void finite(double value, const char* message) {
  if (!std::isfinite(value)) throw std::invalid_argument(message);
}

inline void positive(double value, const char* message) {
  if (!std::isfinite(value) || value <= 0.0)
    throw std::invalid_argument(message);
}

inline void nonnegative(double value, const char* message) {
  if (!std::isfinite(value) || value < 0.0)
    throw std::invalid_argument(message);
}

inline void validateQuadratures(double q, double p) {
  finite(q, "Cavity q coordinate must be finite");
  finite(p, "Cavity p coordinate must be finite");
}

inline void validate(const UniformAfmConfiguration& c, double ms1, double ms2,
                     int occupiedCells) {
  positive(c.omega, "cavity_omega must be finite and positive (rad/s)");
  nonnegative(c.kappa, "cavity_kappa must be finite and nonnegative (rad/s)");
  positive(c.energyField,
           "cavity_energy_field must be finite and positive (tesla)");
  finite(c.field, "cavity_h0 must be finite (tesla per q)");
  finite(c.driveQ, "cavity_drive_re must be finite (1/s)");
  finite(c.driveP, "cavity_drive_im must be finite (1/s)");
  positive(ms1, "Native uniform cavity requires finite positive sublattice Ms");
  positive(ms2, "Native uniform cavity requires finite positive sublattice Ms");
  if (ms1 != ms2)
    throw std::invalid_argument(
        "Native uniform cavity requires equal sublattice Ms; unequal moments "
        "need a moment-weighted cavity source");
  if (occupiedCells <= 0)
    throw std::invalid_argument("Native uniform cavity requires occupied magnetic cells");
  if (c.auxiliary) {
    positive(c.auxOmega, "aux_omega must be finite and positive (rad/s)");
    nonnegative(c.auxKappa, "aux_kappa must be finite and nonnegative (rad/s)");
    finite(c.auxField, "aux_h0 must be finite (tesla per q)");
    finite(c.auxJ, "aux_j must be finite (rad/s)");
  }
}
}  // namespace cavity
