#include "cavityafm.hpp"
#include "cavityconfig.hpp"
#include "antiferromagnet.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"
#include <cmath>
#include <stdexcept>

// Revision energy: E/E0 = q_a^2+p_a^2+q_d^2+p_d^2
// - (H_a/B_norm) q_a <m1x+m2x> - (H_d/B_norm) q_d <m1x+m2x>
// + 4 J/sqrt(omega_a omega_d) q_a q_d + E_spin/E0.
// E0 = M_s V_occupied B_norm for ONE equal, uniform-Ms sublattice;
// B_norm=cavityEnergyField is chosen as B_E in the supplied examples.
// The state stores normalized q,p,
// not dimensionless photon annihilation amplitudes. Baths: dot p=-2 kappa p.
// Native mumax+ Gilbert torques use the SAME field on both sublattices.
// Legacy g, phase, Bogoliubov, and circular-backward settings are not used.
bool cavityAfmAssuredZero(const Ferromagnet* fm) {
  if (!fm->isSublattice()) return true;
  const auto* host=dynamic_cast<const Antiferromagnet*>(fm->hostMagnet());
  return !host || !host->enableCavityAfm();
}

Field evalCavityAfmField(const Ferromagnet* fm) {
  Field h(fm->system(),3); h.makeZero();
  if(cavityAfmAssuredZero(fm)) return h;
  const auto* host=dynamic_cast<const Antiferromagnet*>(fm->hostMagnet());
  host->validateCavityConfiguration();
  auto a=fieldAverage(host->cavityAmplitude()->field());
  cavity::validateQuadratures(a[0],a[1]);
  real hx=host->cavityH0*a[0];
  if(host->enableAuxMode()) {
    auto d=fieldAverage(host->auxAmplitude()->field());
    cavity::validateQuadratures(d[0],d[1]);
    hx+=host->auxH0*d[0];
  }
  h.setUniformComponent(0,hx);
  return h;
}

FM_FieldQuantity cavityAfmFieldQuantity(const Ferromagnet* fm) {
  return FM_FieldQuantity(fm,evalCavityAfmField,3,"cavity_afm_field","T");
}

static real totalMx(const Antiferromagnet* afm) {
  const real mx=fieldComponentAverage(afm->sub1()->magnetization()->field(),0)
       + fieldComponentAverage(afm->sub2()->magnetization()->field(),0);
  cavity::finite(mx,"Cavity magnetic source must be finite");
  return mx;
}

CavityAfmRHSQuantity::CavityAfmRHSQuantity(const Antiferromagnet* afm): afm_(afm) {}
std::shared_ptr<const System> CavityAfmRHSQuantity::system() const {
  return afm_->cavityAmplitudeSystem();
}
Field CavityAfmRHSQuantity::eval() const {
  Field rhs(system(),3); rhs.makeZero();
  if(!afm_->enableCavityAfm()) return rhs;
  afm_->validateCavityConfiguration();
  auto a=fieldAverage(afm_->cavityAmplitude()->field());
  cavity::validateQuadratures(a[0],a[1]);
  real w=afm_->cavityOmega;
  real dq=w*a[1];
  real dp=-w*a[0]-2*afm_->cavityKappa*a[1]
          +w*afm_->cavityH0/(2*afm_->cavityEnergyField)*totalMx(afm_);
  if(afm_->enableAuxMode()) {
    auto d=fieldAverage(afm_->auxAmplitude()->field());
    cavity::validateQuadratures(d[0],d[1]);
    dp-=2*afm_->auxJ*sqrt(w/afm_->auxOmega)*d[0];
  }
  rhs.setUniformComponent(0,dq+afm_->cavityDriveRe);
  rhs.setUniformComponent(1,dp+afm_->cavityDriveIm);
  return rhs;
}

AuxModeRHSQuantity::AuxModeRHSQuantity(const Antiferromagnet* afm): afm_(afm) {}
std::shared_ptr<const System> AuxModeRHSQuantity::system() const {
  return afm_->auxAmplitudeSystem();
}
Field AuxModeRHSQuantity::eval() const {
  Field rhs(system(),3); rhs.makeZero();
  if(!afm_->enableCavityAfm() || !afm_->enableAuxMode()) return rhs;
  afm_->validateCavityConfiguration();
  auto a=fieldAverage(afm_->cavityAmplitude()->field());
  auto d=fieldAverage(afm_->auxAmplitude()->field());
  cavity::validateQuadratures(a[0],a[1]);
  cavity::validateQuadratures(d[0],d[1]);
  real w=afm_->auxOmega;
  real dq=w*d[1];
  real dp=-w*d[0]-2*afm_->auxKappa*d[1]
          +w*afm_->auxH0/(2*afm_->cavityEnergyField)*totalMx(afm_)
          -2*afm_->auxJ*sqrt(w/afm_->cavityOmega)*a[0];
  rhs.setUniformComponent(0,dq);
  rhs.setUniformComponent(1,dp);
  return rhs;
}
