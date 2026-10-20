// Input-contract tests only. No CUDA, trajectory, eigensolve, FFT, or fit.
// Example: c++ -std=c++17 -Isrc/physics test/cavity_config_host.cpp -o cavity_config_host
#include "cavityconfig.hpp"

#include <iostream>
#include <limits>

int main() {
  int checks = 0;
  auto require = [&](bool condition) {
    if (!condition) throw std::runtime_error("Host cavity configuration contract failed");
    ++checks;
  };
  auto refused = [&](auto action) {
    try { action(); } catch (const std::invalid_argument&) { return true; }
    return false;
  };
  const cavity::UniformAfmConfiguration valid{
      1.0, 0.0, -0.1, 2.0, 0.0, 0.0, true, 2.0, 0.0, 0.1, -0.2};
  cavity::validate(valid, 1.0, 1.0, 1);
  ++checks;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  // No field/coupling-sign constraint is invented; only rates/scales have
  // positivity restrictions. These tests do not assert physical stability.
  for (int member = 0; member < 10; ++member) {
    auto c = valid;
    double* entries[] = {&c.omega, &c.kappa, &c.field, &c.energyField,
                         &c.driveQ, &c.driveP, &c.auxOmega, &c.auxKappa,
                         &c.auxField, &c.auxJ};
    *entries[member] = nan;
    require(refused([&] { cavity::validate(c, 1.0, 1.0, 1); }));
    *entries[member] = inf;
    require(refused([&] { cavity::validate(c, 1.0, 1.0, 1); }));
  }
  for (int member = 0; member < 5; ++member) {
    auto c = valid;
    double* entries[] = {&c.omega, &c.kappa, &c.energyField, &c.auxOmega, &c.auxKappa};
    *entries[member] = -1.0;
    require(refused([&] { cavity::validate(c, 1.0, 1.0, 1); }));
  }
  for (int member = 0; member < 3; ++member) {
    auto c = valid;
    double* entries[] = {&c.omega, &c.energyField, &c.auxOmega};
    *entries[member] = 0.0;
    require(refused([&] { cavity::validate(c, 1.0, 1.0, 1); }));
  }
  require(refused([&] { cavity::validate(valid, 1.0, 2.0, 1); }));
  require(refused([&] { cavity::validate(valid, 0.0, 0.0, 1); }));
  require(refused([&] { cavity::validate(valid, nan, 1.0, 1); }));
  require(refused([&] { cavity::validate(valid, 1.0, inf, 1); }));
  require(refused([&] { cavity::validate(valid, 1.0, 1.0, 0); }));
  require(refused([&] { cavity::validate(valid, 1.0, 1.0, -1); }));
  require(refused([&] { cavity::validateQuadratures(nan, 0.0); }));
  require(refused([&] { cavity::validateQuadratures(0.0, inf); }));
  auto disabledAux = valid;
  disabledAux.auxiliary = false;
  disabledAux.auxOmega = 0.0;
  disabledAux.auxKappa = nan;
  cavity::validate(disabledAux, 1.0, 1.0, 1);  // inactive coordinates are irrelevant
  ++checks;
  std::cout << "PASS " << checks << " host-only configuration checks\n";
}
