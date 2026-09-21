"""Declare physical inputs before predictions; no coupled result is read."""
from pathlib import Path
import json,math,hashlib
W=Path(__file__).resolve().parent
mu=624e3*(40e-9)*(40e-9)*(10e-9);hbar=1.054571817e-34
gamma=1.7595e11;unit=2*math.pi*1e9
# Used ONLY to prescribe the old common-G diagnostic. The material-input-free EP predictor
# never uses these nominal r values or G to define its predicted coupling.
nominal={'A':.084287616,'B':.126,'C':.99}
points={key:{'B_star_T':.459542044187*unit/math.sqrt(gamma*mu/(2*hbar)*r),
             'wc_GHz':261.318803319,'kappa_GHz':.261365756187} for key,r in nominal.items()}
settings={'scope':'Native Gilbert, zero bias, uniform Zeeman, viscous cavity bath. Nine physical-field EP predictions and one separately declared common-G response test.',
 'mu_s_J_per_T':mu,'hbar_J_s':hbar,'eta_external':.5,
 'kappa_GHz':[.1306828780935,.261365756187,.522731512374],
 'common_G_operating_points':points,
 'common_G_provenance':'The preparer used nominal r and the energy-defined Zeeman overlap to set these physical B_star values at common G_U/(2pi)=459.542044187 MHz. This diagnostic is distinct from the material-input-free physical-field EP prediction and is not a claim of inferring the field overlap from no material information.',
 'common_G_gamma_rad_s_T':gamma,'no_coupled_data_read':True}
(W/'prediction_settings.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
print('Public physical settings prepared; no material r values are exported to predictor.')
