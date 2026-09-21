#include "antiferromagnet.hpp"

#include <algorithm>
#include <memory>
#include <math.h>
#include <cfloat>
#include <vector>

#include "fieldquantity.hpp"
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

void Antiferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void Antiferromagnet::relax(real tol) {
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
