#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Antiferromagnet;
class Field;

bool cavityAfmAssuredZero(const Ferromagnet* fm);

Field evalCavityAfmField(const Ferromagnet* fm);
FM_FieldQuantity cavityAfmFieldQuantity(const Ferromagnet* fm);

// Primary energy-normalized oscillator, stored as (q_a, p_a, unused).
// dq_a/dt = omega_a p_a; dp_a/dt = -omega_a q_a - 2 kappa_a p_a
// + omega_a H_a <m1x+m2x>/(2 B_E) - 2 J sqrt(omega_a/omega_d) q_d.
// The source and uniform spin field follow the same Zeeman energy.
class CavityAfmRHSQuantity : public FieldQuantity {
 public:
  explicit CavityAfmRHSQuantity(const Antiferromagnet* afm);
  CavityAfmRHSQuantity* clone() { return new CavityAfmRHSQuantity(afm_); }

  int ncomp() const override { return 3; }
  std::shared_ptr<const System> system() const override;
  std::string name() const override { return "cavity_afm_rhs"; }
  std::string unit() const override { return "1/s"; }
  Field eval() const override;

 private:
  const Antiferromagnet* afm_;
};

// Auxiliary energy-normalized oscillator, stored as (q_d, p_d, unused).
// Its reciprocal photon and spin sources are derived from the same energy;
// viscous damping acts on p_d as -2 kappa_d p_d.
class AuxModeRHSQuantity : public FieldQuantity {
 public:
  explicit AuxModeRHSQuantity(const Antiferromagnet* afm);
  AuxModeRHSQuantity* clone() { return new AuxModeRHSQuantity(afm_); }

  int ncomp() const override { return 3; }
  std::shared_ptr<const System> system() const override;
  std::string name() const override { return "aux_mode_rhs"; }
  std::string unit() const override { return "1/s"; }
  Field eval() const override;

 private:
  const Antiferromagnet* afm_;
};
