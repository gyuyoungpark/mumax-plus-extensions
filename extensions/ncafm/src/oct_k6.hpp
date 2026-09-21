#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

// Per-sublattice energy: k6_oct * my^2 * (3*mx^2 - my^2)^2 (J/m3).
// The planar rigid 120-degree barrier is 3*abs(k6_oct). Off plane this
// polynomial is not a pure sixth-order Stevens anisotropy.
bool octK6AssuredZero(const Ferromagnet*);
Field evalOctK6Field(const Ferromagnet*);
Field evalOctK6EnergyDensity(const Ferromagnet*);
real evalOctK6Energy(const Ferromagnet*);

FM_FieldQuantity octK6FieldQuantity(const Ferromagnet*);
FM_FieldQuantity octK6EnergyDensityQuantity(const Ferromagnet*);
FM_ScalarQuantity octK6EnergyQuantity(const Ferromagnet*);
