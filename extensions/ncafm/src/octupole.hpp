#pragma once

#include "quantityevaluator.hpp"

// Fixed-index, lab-frame sublattice order parameter, normalized by total Ms.
// This is not a complete coordinate-dependent cluster multipole tensor.
Field evalOctupoleVector(const NcAfm*);
NcAfm_FieldQuantity octupoleVectorQuantity(const NcAfm*);
