#include "cudalaunch.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "ncafm.hpp"
#include "oct_k6.hpp"
#include "parameter.hpp"

__global__ void k_octK6Field(CuField hField,
                             const CuField magnetization,
                             const CuParameter k6Oct,
                             const CuParameter msat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!hField.cellInGeometry(idx)) {
    if (hField.cellInGrid(idx))
      hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
  const real ms = msat.valueAt(idx);
  if (ms == 0) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
  const real3 m = magnetization.vectorAt(idx);
  const real u = 3 * m.x * m.x - m.y * m.y;
  const real v = m.x * m.x - m.y * m.y;
  const real factor = k6Oct.valueAt(idx) / ms;
  hField.setVectorInCell(idx, real3{-12 * factor * m.x * m.y * m.y * u,
                                    -6 * factor * m.y * u * v, 0});
}

__global__ void k_octK6EnergyDensity(CuField edens,
                                     const CuField magnetization,
                                     const CuParameter k6Oct,
                                     const CuParameter msat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!edens.cellInGeometry(idx)) {
    if (edens.cellInGrid(idx))
      edens.setValueInCell(idx, 0, 0);
    return;
  }
  if (msat.valueAt(idx) == 0) {
    edens.setValueInCell(idx, 0, 0);
    return;
  }
  const real3 m = magnetization.vectorAt(idx);
  const real u = 3 * m.x * m.x - m.y * m.y;
  edens.setValueInCell(idx, 0, k6Oct.valueAt(idx) * m.y * m.y * u * u);
}

bool octK6AssuredZero(const Ferromagnet* magnet) {
  const auto* host = magnet->hostMagnet();
  const auto* ncafm = host ? host->asNcAfm() : nullptr;
  return !ncafm || ncafm->k6Oct.assuredZero() || magnet->msat.assuredZero();
}

Field evalOctK6Field(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3, real3{0, 0, 0});
  if (octK6AssuredZero(magnet))
    return hField;
  cudaLaunch(hField.grid().ncells(), k_octK6Field, hField.cu(),
             magnet->magnetization()->field().cu(),
             magnet->hostMagnet()->asNcAfm()->k6Oct.cu(), magnet->msat.cu());
  return hField;
}

Field evalOctK6EnergyDensity(const Ferromagnet* magnet) {
  Field edens(magnet->system(), 1, 0.0);
  if (octK6AssuredZero(magnet))
    return edens;
  cudaLaunch(edens.grid().ncells(), k_octK6EnergyDensity, edens.cu(),
             magnet->magnetization()->field().cu(),
             magnet->hostMagnet()->asNcAfm()->k6Oct.cu(), magnet->msat.cu());
  return edens;
}

real evalOctK6Energy(const Ferromagnet* magnet) {
  if (octK6AssuredZero(magnet))
    return 0;
  return energyFromEnergyDensity(
      magnet, octK6EnergyDensityQuantity(magnet).average()[0]);
}

FM_FieldQuantity octK6FieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalOctK6Field, 3, "oct_k6_field", "T");
}

FM_FieldQuantity octK6EnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalOctK6EnergyDensity, 1,
                          "oct_k6_energy_density", "J/m3");
}

FM_ScalarQuantity octK6EnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalOctK6Energy, "oct_k6_energy", "J");
}
