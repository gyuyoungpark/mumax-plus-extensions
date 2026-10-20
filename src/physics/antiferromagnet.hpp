#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ferromagnet.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "hostmagnet.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include "system.hpp"
#include "variable.hpp"

class Antiferromagnet : public HostMagnet {
 public:
  Antiferromagnet(std::shared_ptr<System> system_ptr,
                  std::string name);

  Antiferromagnet(MumaxWorld* world,
         Grid grid,
         std::string name,
         GpuBuffer<bool> geometry,
         GpuBuffer<unsigned int> regions);
         
  /** Empty destructor
   * Sublattices are destroyed automatically. They are not pointers.
   */
  ~Antiferromagnet() override {};
  
 const Ferromagnet* sub1() const;
 const Ferromagnet* sub2() const;
 
 void minimize(real tol = 1e-6, int nSamples = 20);
 void relax(real tol);

 // Energy-normalized uniform-x cavity coordinates (q, p, unused).
 // E0 = M_s V_occupied * cavityEnergyField, for ONE equal, static,
 // spatially uniform-Ms sublattice. The two sublattices share geometry.
 bool enableCavityAfm() const { return enableCavityAfm_; }
 void setEnableCavityAfm(bool);
 bool enableAuxMode() const { return enableAuxMode_; }
 void setEnableAuxMode(bool);
 void validateCavityConfiguration() const;
 real cavityOmega = 0.0;
 real cavityKappa = 0.0;
 real cavityH0 = 0.0; // Tesla multiplying q in the field on BOTH sublattices.
 real cavityEnergyField = 1.0; // Positive normalization field (chosen as B_E in examples).
 real cavityDriveRe = 0.0; // Additive dq/dt drive, in 1/s.
 real cavityDriveIm = 0.0; // Additive dp/dt drive, in 1/s.
 real auxOmega = 0.0;
 real auxKappa = 0.0;
 real auxJ = 0.0;
 real auxH0 = 0.0;

 const Variable* cavityAmplitude() const { return &cavityAmplitude_; }
 const Variable* auxAmplitude() const { return &auxAmplitude_; }
 std::shared_ptr<const System> cavityAmplitudeSystem() const;
 std::shared_ptr<const System> auxAmplitudeSystem() const;

 private:
  bool enableCavityAfm_ = false;
  bool enableAuxMode_ = false;
  void validateCavityConfiguration(bool enabled, bool auxiliary) const;
  Ferromagnet sub1_;
  Ferromagnet sub2_;
  Variable cavityAmplitude_;
  Variable auxAmplitude_;
};
