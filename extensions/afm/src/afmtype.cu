#include <stdexcept>

#include "afmtype.hpp"
#include "cudalaunch.hpp"
#include "dmi.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "hostmagnet.hpp"
#include "inter_parameter.hpp"
#include "parameter.hpp"
#include "world.hpp"

__global__ void k_anisotropicAfmExchangeField(CuField hField,
                                              const CuField sisterMagnetization,
                                              const CuVectorParameter afmexDir,
                                              const CuInterParameter interExch,
                                              const CuInterParameter scaleExch,
                                              const CuParameter msat,
                                              const CuParameter sisterMsat,
                                              const Grid mastergrid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
  if (msat.valueAt(idx) == 0 || sisterMsat.valueAt(idx) == 0)
    return;

  const auto system = hField.system;
  const int3 coo = system.grid.index2coord(idx);
  const real3 m = sisterMagnetization.vectorAt(idx);
  const real3 stiffness = afmexDir.vectorAt(idx);
  const unsigned int region = system.getRegionIdx(idx);
  real3 h{0, 0, 0};

  // Each bond contributes A_d * (delta m_i).(delta m_j) / cellsize_d^2.
  // All four endpoint moments must be active for a reciprocal bond.
#pragma unroll
  for (int3 offset : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                      int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    const int3 neighbor = mastergrid.wrap(coo + offset);
    if (!hField.cellInGeometry(neighbor))
      continue;
    const int neighborIdx = system.grid.coord2index(neighbor);
    if (msat.valueAt(neighborIdx) == 0 || sisterMsat.valueAt(neighborIdx) == 0)
      continue;

    const int3 normal = offset * offset;
    const real a = dot(normal, stiffness);
    const real b = dot(normal, afmexDir.vectorAt(neighborIdx));
    const unsigned int neighborRegion = system.getRegionIdx(neighborIdx);
    real inter = 0;
    real scale = 1;
    if (region != neighborRegion) {
      inter = interExch.valueBetween(region, neighborRegion);
      scale = scaleExch.valueBetween(region, neighborRegion);
    }
    const real coupling = getExchangeStiffness(inter, scale, a, b);
    const real delta = dot(offset, system.cellsize);
    h += coupling * (sisterMagnetization.vectorAt(neighborIdx) - m) /
         (delta * delta);
  }
  hField.setVectorInCell(idx, hField.vectorAt(idx) + h / msat.valueAt(idx));
}

bool anisotropicAfmExchangeAssuredZero(const Ferromagnet* magnet) {
  const auto* host = magnet->hostMagnet();
  if (!host || host->afmexNNDir.assuredZero() || magnet->msat.assuredZero())
    return true;
  for (auto* sub : host->getOtherSublattices(magnet)) {
    if (!sub->msat.assuredZero())
      return false;
  }
  return true;
}

Field evalAnisotropicAfmExchangeField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3, real3{0, 0, 0});
  if (anisotropicAfmExchangeAssuredZero(magnet))
    return hField;

  const auto* host = magnet->hostMagnet();
  for (auto* sub : host->sublattices()) {
    if (!sub->enableOpenBC)
      throw std::invalid_argument(
          "Directional AFM exchange requires enable_openbc=True on all "
          "sublattices. Periodic neighbors remain coupled.");
  }
  for (auto* sub : host->getOtherSublattices(magnet)) {
    cudaLaunch(hField.grid().ncells(), k_anisotropicAfmExchangeField,
               hField.cu(), sub->magnetization()->field().cu(),
               host->afmexNNDir.cu(), host->interAfmExchNN.cu(),
               host->scaleAfmExchNN.cu(), magnet->msat.cu(), sub->msat.cu(),
               magnet->world()->mastergrid());
  }
  return hField;
}

Field evalAnisotropicAfmExchangeEnergyDensity(const Ferromagnet* magnet) {
  if (anisotropicAfmExchangeAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);
  return evalEnergyDensity(magnet, evalAnisotropicAfmExchangeField(magnet),
                           0.5);
}

real evalAnisotropicAfmExchangeEnergy(const Ferromagnet* magnet) {
  if (anisotropicAfmExchangeAssuredZero(magnet))
    return 0;
  const real edens =
      anisotropicAfmExchangeEnergyDensityQuantity(magnet).average()[0];
  return energyFromEnergyDensity(magnet, edens);
}

FM_FieldQuantity anisotropicAfmExchangeFieldQuantity(
    const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAnisotropicAfmExchangeField, 3,
                          "anisotropic_afm_exchange_field", "T");
}

FM_FieldQuantity anisotropicAfmExchangeEnergyDensityQuantity(
    const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalAnisotropicAfmExchangeEnergyDensity, 1,
                          "anisotropic_afm_exchange_energy_density", "J/m3");
}

FM_ScalarQuantity anisotropicAfmExchangeEnergyQuantity(
    const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalAnisotropicAfmExchangeEnergy,
                           "anisotropic_afm_exchange_energy", "J");
}
