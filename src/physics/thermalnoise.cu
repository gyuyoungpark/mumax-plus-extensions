#include <curand.h>
#include <stdexcept>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "parameter.hpp"
#include "thermalnoise.hpp"
#include "world.hpp"

#if FP_PRECISION == SINGLE
const auto& generateRandNormal = curandGenerateNormal;
#elif FP_PRECISION == DOUBLE
const auto& generateRandNormal = curandGenerateNormalDouble;
#endif

bool thermalNoiseAssuredZero(const Ferromagnet* magnet) {
  return magnet->temperature.assuredZero();
}

__global__ void k_thermalNoise(CuField noiseField,
                               const CuParameter msat,
                               const CuParameter alpha,
                               const CuParameter gamma,
                               const CuParameter temperature,
                               real preFactor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!noiseField.cellInGeometry(idx)) {
    if (noiseField.cellInGrid(idx))
      noiseField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  if (!noiseField.cellInGrid(idx))
    return;
  if (msat.valueAt(idx) == 0) {
    noiseField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real Ms = msat.valueAt(idx);
  real T = temperature.valueAt(idx);
  real a = alpha.valueAt(idx);
  real g = gamma.valueAt(idx);

  real3 noise = noiseField.vectorAt(idx);
  noise *= sqrt(preFactor * g * a * T / ((1 + a * a) * Ms));
  noiseField.setVectorInCell(idx, noise);
}

Field evalThermalNoise(const Ferromagnet* magnet) {
  Field noise(magnet->system(), 3);

  if (thermalNoiseAssuredZero(magnet)) {
    noise.makeZero();
    return noise;
  }
  
  int N = noise.grid().ncells();
  real mean = 0.0;
  real stddev = 1.0;
  // cuRAND normal generation requires an even element count. A one-cell
  // macrospin (or any odd-sized grid) must not leave uninitialized noise.
  const int paddedN = N + (N % 2);
  GpuBuffer<real> scratch(N % 2 ? paddedN : 0);
  for (int c = 0; c < 3; c++) {
    real* destination = N % 2 ? scratch.get() : noise.device_ptr(c);
    if (generateRandNormal(magnet->randomGenerator, destination, paddedN,
                           mean, stddev) != CURAND_STATUS_SUCCESS)
      throw std::runtime_error("Thermal Gaussian generation failed");
    if (N % 2 && cudaMemcpy(noise.device_ptr(c), destination, N * sizeof(real),
                           cudaMemcpyDeviceToDevice) != cudaSuccess)
      throw std::runtime_error("Thermal Gaussian buffer copy failed");
  }

  auto msat = magnet->msat.cu();
  auto alpha = magnet->alpha.cu();
  auto gamma = magnet->gamma.cu();
  auto temp = magnet->temperature.cu();
  real cellVolume = magnet->world()->cellVolume();
  real preFactor = 2 * KB / cellVolume;
  cudaLaunch(N, k_thermalNoise, noise.cu(), msat, alpha, gamma, temp, preFactor);
  return noise;
}

FM_FieldQuantity thermalNoiseQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalThermalNoise, 3, "thermalNoise", "");
}
