#pragma once

#include "quantityevaluator.hpp"

class Ferromagnet;
class Field;

// Additive cross-gradient stiffnesses along the Cartesian grid axes (J/m).
// These continuum coefficients do not specify an atomistic A/C/G bond graph.
bool anisotropicAfmExchangeAssuredZero(const Ferromagnet*);
Field evalAnisotropicAfmExchangeField(const Ferromagnet*);
Field evalAnisotropicAfmExchangeEnergyDensity(const Ferromagnet*);
real evalAnisotropicAfmExchangeEnergy(const Ferromagnet*);

FM_FieldQuantity anisotropicAfmExchangeFieldQuantity(const Ferromagnet*);
FM_FieldQuantity anisotropicAfmExchangeEnergyDensityQuantity(
    const Ferromagnet*);
FM_ScalarQuantity anisotropicAfmExchangeEnergyQuantity(const Ferromagnet*);
