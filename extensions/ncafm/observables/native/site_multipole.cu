// Generic linear site-moment contraction. Coefficients include geometry,
// moments and counting weights; output components are declared by the caller.
// Layout: coefficients[output][site][xyz], moments[cell][site][xyz].
#include <cuda_runtime.h>
#include <cstddef>

__global__ void siteMultipoleKernel(const double* coefficients,
                                  const double* moments, double* output,
                                  int cells, int sites, int components) {
  const std::size_t index = blockIdx.x*blockDim.x+threadIdx.x;
  if (index >= static_cast<std::size_t>(cells)*components) return;
  const int cell = index/components;
  const int component = index%components;
  double sum = 0;
  for (int site=0; site<sites; ++site)
    for (int axis=0; axis<3; ++axis)
      sum += coefficients[(static_cast<std::size_t>(component)*sites+site)*3+axis]*
             moments[(static_cast<std::size_t>(cell)*sites+site)*3+axis];
  output[index] = sum;
}

// Arrays are device-resident. Stream lifetime is owned by the caller.
extern "C" cudaError_t launchSiteMultipole(const double* coefficients,
    const double* moments, double* output, int cells, int sites,
    int components, cudaStream_t stream) {
  if (!coefficients || !moments || !output || cells<=0 || sites<=0 ||
      components<=0) return cudaErrorInvalidValue;
  const std::size_t count = static_cast<std::size_t>(cells)*components;
  if (count > static_cast<std::size_t>(2147483647)*128)
    return cudaErrorInvalidValue;
  siteMultipoleKernel<<<(count+127)/128,128,0,stream>>>(
      coefficients, moments, output, cells, sites, components);
  return cudaGetLastError();
}
