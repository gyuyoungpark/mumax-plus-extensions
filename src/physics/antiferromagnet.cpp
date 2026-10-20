#include "antiferromagnet.hpp"

#include <algorithm>
#include <memory>
#include <math.h>
#include <cfloat>
#include <vector>

#include "fieldquantity.hpp"
#include "cavityconfig.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "relaxer.hpp"

Antiferromagnet::Antiferromagnet(std::shared_ptr<System> system_ptr,
                                 std::string name)
    : HostMagnet(system_ptr, name),
      sub1_(Ferromagnet(system_ptr, name + ":sublattice_1", this)),
      sub2_(Ferromagnet(system_ptr, name + ":sublattice_2", this)),
      cavityAmplitude_(Variable(
          std::make_shared<System>(system_ptr->world(), Grid(int3{1, 1, 1}),
                                  GpuBuffer<bool>(), GpuBuffer<unsigned int>()),
          3, name + ":cavity_amplitude", "1")),
      auxAmplitude_(Variable(
          std::make_shared<System>(system_ptr->world(), Grid(int3{1, 1, 1}),
                                  GpuBuffer<bool>(), GpuBuffer<unsigned int>()),
          3, name + ":aux_amplitude", "1")) {
        addSublattice(&sub1_);
        addSublattice(&sub2_);
        cavityAmplitude_.set(real3{0, 0, 0});
        auxAmplitude_.set(real3{0, 0, 0});
      }
      
Antiferromagnet::Antiferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry,
                         GpuBuffer<unsigned int> regions)
    : Antiferromagnet(std::make_shared<System>(world, grid, geometry, regions), name) {}

const Ferromagnet* Antiferromagnet::sub1() const {
  return &sub1_;
}

const Ferromagnet* Antiferromagnet::sub2() const {
  return &sub2_;
}

std::shared_ptr<const System> Antiferromagnet::cavityAmplitudeSystem() const {
  return cavityAmplitude_.system();
}

std::shared_ptr<const System> Antiferromagnet::auxAmplitudeSystem() const {
  return auxAmplitude_.system();
}

void Antiferromagnet::validateCavityConfiguration(bool enabled, bool auxiliary) const {
  if (!enabled) return;
  // A plain sum of volume averages is reciprocal only for equal, uniform,
  // static moments. Do not silently apply it to ferrimagnets or varying Ms.
  if (sub1_.msat.isDynamic() || sub2_.msat.isDynamic() ||
      !sub1_.msat.isUniform() || !sub2_.msat.isUniform())
    throw std::invalid_argument(
        "Native uniform cavity requires static spatially uniform Ms on both sublattices");
  cavity::validate(
      {cavityOmega, cavityKappa, cavityH0, cavityEnergyField,
       cavityDriveRe, cavityDriveIm, auxiliary,
       auxOmega, auxKappa, auxH0, auxJ},
      sub1_.msat.getUniformValue(), sub2_.msat.getUniformValue(),
      system()->cellsInGeo());
}

void Antiferromagnet::validateCavityConfiguration() const {
  validateCavityConfiguration(enableCavityAfm_, enableAuxMode_);
}

void Antiferromagnet::setEnableCavityAfm(bool enabled) {
  validateCavityConfiguration(enabled, enableAuxMode_);
  if (enabled == enableCavityAfm_) return;
  const bool previous = enableCavityAfm_;
  enableCavityAfm_ = enabled;
  try {
    mumaxWorld()->resetTimeSolverEquations();
  } catch (...) {
    enableCavityAfm_ = previous;
    // setEquations preserves the old equations and dt if RHS admission fails.
    throw;
  }
}

void Antiferromagnet::setEnableAuxMode(bool enabled) {
  // Auxiliary settings may be staged while the primary cavity is disabled.
  // They become active and are validated when the primary is enabled.
  validateCavityConfiguration(enableCavityAfm_, enabled);
  if (enabled == enableAuxMode_) return;
  const bool previous = enableAuxMode_;
  enableAuxMode_ = enabled;
  try {
    mumaxWorld()->resetTimeSolverEquations();
  } catch (...) {
    enableAuxMode_ = previous;
    throw;
  }
}

void Antiferromagnet::minimize(real tol, int nSamples) {
  if (enableCavityAfm_)
    throw std::invalid_argument("Disable the cavity before spin-only minimize()");
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void Antiferromagnet::relax(real tol) {
  if (enableCavityAfm_)
    throw std::invalid_argument("Disable the cavity before spin-only relax()");
  std::vector<real> threshold = {sub1()->RelaxTorqueThreshold,
                                 sub2()->RelaxTorqueThreshold};
    // If only one sublattice has a user-set threshold, then both
    // sublattices are relaxed using the same threshold.
    if (threshold[0] > 0.0 && threshold[1] <= 0.0)
      threshold[1] = threshold[0];
    else if (threshold[0] <= 0.0 && threshold[1] > 0.0)
      threshold[0] = threshold[1];

    Relaxer relaxer(this, threshold, tol);
    relaxer.exec();
}
