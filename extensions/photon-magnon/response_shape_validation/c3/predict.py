"""Free-data-only physical-field predictions, frozen before coupled validation.

Reads only the public measured-coefficient export and physical operating inputs.
No material fields, alpha, squeezing ratio or coupled matrices are available here.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import mpmath as mp
mp.mp.dps=70
M=mp.mpf
W=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dec(x):return mp.nstr(x,55)
def enc(z):return {'real':dec(mp.re(z)),'imag':dec(mp.im(z))}
public=W/'calibration_public_frozen.json'
settings=W/'prediction_settings.json'
cals=json.loads(public.read_text(encoding='utf-8'))
spec=json.loads(settings.read_text(encoding='utf-8'))
u=2*mp.pi*M('1e9')
F=2*M(str(spec['mu_s_J_per_T']))/(M(str(spec['hbar_J_s']))*u)
reference=next(x for x in cals['cases'] if x['case_id']=='A')
rho_ref=M(str(reference['A_tilde_per_T']))/M(str(reference['B_tilde_per_T']))
freq=[M(str(reference['Omega_bar']))-4+M(8)*i/800 for i in range(801)]

def ep(gam,om,bb,rho,kap):
    delta=gam-kap
    L=mp.sqrt((1-rho*gam)**2+(rho*om)**2)
    x=2*delta*rho*om*om/(L+1-rho*gam)
    wc=mp.sqrt(om*om+kap*kap+x)
    H=om*om*delta*delta+gam*delta*x-x*x/4
    B=mp.sqrt(H/(wc*F*bb))
    lam=-(gam+kap)/2-mp.j*mp.sqrt(om*om-delta*delta/4+x/2)
    return {'B_star_T':dec(B),'wc_GHz':dec(wc),'pole':enc(lam),'Q':dec(wc/(2*kap))}

def response(gam,om,aa,bb,kap,wc,bstar):
    ans=[]
    for f in freq:
        z=-mp.j*f;D=z*z+2*gam*z+om*om+gam*gam
        P=D*(z*z+2*kap*z+wc*wc)-wc*F*bstar*bstar*(aa*z+bb)
        ans.append(enc(-2*M(str(spec['eta_external']))*kap*z*D/P))
    return ans

rows=[]
assert sorted(c['case_id'] for c in cals['cases'])==['A','B','C']
for c in sorted(cals['cases'],key=lambda c:c['case_id']):
    gam,om,aa,bb=[M(str(c[k])) for k in ['Gamma_bar','Omega_bar','A_tilde_per_T','B_tilde_per_T']]
    for kapstr in spec['kappa_GHz']:
        kap=M(str(kapstr));exact=ep(gam,om,bb,aa/bb,kap);scalar=ep(gam,om,bb,rho_ref,kap)
        wc=M(exact['wc_GHz']);bf=M(exact['B_star_T'])
        rows.append({'case_id':c['case_id'],'kappa_GHz':dec(kap),'shape_prediction':exact,'strength_scalar_prediction':scalar,
          'response_operating_point':'same frozen shape-predicted physical B_star and wc for both models',
          'response_shape':response(gam,om,aa,bb,kap,wc,bf),
          'response_strength_scalar':response(gam,om,bb*rho_ref,bb,kap,wc,bf)})
common=[]
for c in cals['cases']:
    gam,om,aa,bb=[M(str(c[k])) for k in ['Gamma_bar','Omega_bar','A_tilde_per_T','B_tilde_per_T']]
    point=spec['common_G_operating_points'][c['case_id']]
    kap,wc,bf=[M(str(point[k])) for k in ['kappa_GHz','wc_GHz','B_star_T']]
    common.append({'case_id':c['case_id'],'physical_operating_point':point,
      'response_shape':response(gam,om,aa,bb,kap,wc,bf),
      'response_strength_scalar':response(gam,om,bb*rho_ref,bb,kap,wc,bf)})
out={'scope':spec['scope'],'frozen_utc':datetime.now(timezone.utc).isoformat(),
 'read_files':{public.name:sha(public),settings.name:sha(settings)},'predictor_sha256':sha(Path(__file__)),
 'truth_or_coupled_data_read':False,'frequency_GHz':[dec(f) for f in freq],
 'rows':rows,'common_G_responses':common,'rho_ref_bar':dec(rho_ref),
 'note':'EP prediction uses only fitted physical susceptibility and physical B_star; no conversion through truth r. The separately prescribed common-G field settings use nominal energy normalization, as disclosed in settings provenance.'}
outpath=W/'predictions_frozen.json'
assert not outpath.exists(),'Preserve frozen predictions; start another explicitly named run if free calibration changes.'
outpath.write_text(json.dumps(out,indent=2),encoding='utf-8')
(W/'predictions_frozen.sha256').write_text(sha(outpath)+'  predictions_frozen.json\n',encoding='ascii')
print(json.dumps({'frozen_sha256':sha(outpath),'predicted_EP_conditions':len(rows),'common_G_cases':len(common)}))
