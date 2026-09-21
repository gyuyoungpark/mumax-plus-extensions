#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "field.hpp"
#include "ncafm.hpp"
#include "octupole.hpp"

__device__ static real3 rotateInPlane(real3 m, real cosAngle, real sinAngle) {
  return real3{cosAngle * m.x - sinAngle * m.y, sinAngle * m.x + cosAngle * m.y,
               m.z};
}

__global__ void k_octupolevector(CuField octupole,
                                 const CuField mag1,
                                 const CuField mag2,
                                 const CuField mag3,
                                 const CuParameter msat1,
                                 const CuParameter msat2,
                                 const CuParameter msat3) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!octupole.cellInGeometry(idx)) {
    if (octupole.cellInGrid(idx))
      octupole.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m1 = mag1.vectorAt(idx);
  real3 m2 = mag2.vectorAt(idx);
  real3 m3 = mag3.vectorAt(idx);
  real ms1 = msat1.valueAt(idx);
  real ms2 = msat2.valueAt(idx);
  real ms3 = msat3.valueAt(idx);

  const real totalMs = ms1 + ms2 + ms3;
  if (totalMs == 0) {
    octupole.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }
  // The rotation is attached to the sublattice index, not inferred from spins.
  const real sin120 = 0.86602540378443864676;
  const real3 m2Rot = rotateInPlane(m2, -0.5, -sin120);
  const real3 m3Rot = rotateInPlane(m3, -0.5, sin120);
  octupole.setVectorInCell(idx,
                           (m1 * ms1 + m2Rot * ms2 + m3Rot * ms3) / totalMs);
}

Field evalOctupoleVector(const NcAfm* magnet) {
  // Fixed-index order parameter with saturation-magnetization weights.
  Field octupole(magnet->system(), 3);

  if (magnet->sub1()->msat.assuredZero() &&
      magnet->sub2()->msat.assuredZero() &&
      magnet->sub3()->msat.assuredZero()) {
    octupole.makeZero();
    return octupole;
  }
  cudaLaunch(octupole.grid().ncells(), k_octupolevector, octupole.cu(),
             magnet->sub1()->magnetization()->field().cu(),
             magnet->sub2()->magnetization()->field().cu(),
             magnet->sub3()->magnetization()->field().cu(),
             magnet->sub1()->msat.cu(), magnet->sub2()->msat.cu(),
             magnet->sub3()->msat.cu());
  return octupole;
}

NcAfm_FieldQuantity octupoleVectorQuantity(const NcAfm* magnet) {
  return NcAfm_FieldQuantity(magnet, evalOctupoleVector, 3, "octupole_vector",
                             "");
}
