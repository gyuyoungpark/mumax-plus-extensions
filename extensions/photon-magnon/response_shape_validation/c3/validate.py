"""Post-freeze original-sublattice energy validation; never alters predictions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import mpmath as mp
import numpy as np
mp.mp.dps=75
M=mp.mpf
W=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dec(x):return mp.nstr(x,55)
def enc(z):return {'real':dec(mp.re(z)),'imag':dec(mp.im(z))}
pred=read(W/'predictions_frozen.json')
assert sha(W/'predictions_frozen.json')==(W/'predictions_frozen.sha256').read_text().split()[0]
frozen_hash=sha(W/'predictions_frozen.json');started=datetime.now(timezone.utc).isoformat()
assert pred['frozen_utc']<started
spec=read(W/'prediction_settings.json')
for name,digest in pred['read_files'].items():assert sha(W/name)==digest
# Private truth is opened for the first time in the coupled pipeline here.
private=read(W.parent/'c2/private/truth_by_case.json')
if 'cases' in private:
    private=private['cases']
if isinstance(private,list):private={v['case_id']:v for v in private}
u=2*mp.pi*M('1e9');mu=M(str(spec['mu_s_J_per_T']));hb=M(str(spec['hbar_J_s']))
etaext=M(str(spec['eta_external']))
frequencies=np.array(pred['frequency_GHz'],dtype=float)

def matrix(cid,bstar,wc,kap):
    d=private[cid];gg=M(str(d['gamma_rad_s_T']))/u
    be=M(str(d['BE_T']));ba=M(str(d['BA_T']));al=M(str(d['alpha']))
    eta=1/(1+al*al);K=mp.zeros(6);J=mp.zeros(6)
    for i in range(4):K[i,i]=gg*(be+ba)
    K[0,2]=K[2,0]=K[1,3]=K[3,1]=gg*be
    K[4,4]=K[5,5]=wc
    h=bstar*mp.sqrt(2*gg*u*mu/hb)/u
    K[0,4]=K[4,0]=K[2,4]=K[4,2]=-h
    J[0,1]=-eta;J[1,0]=eta;J[2,3]=eta;J[3,2]=-eta
    J[4,5]=1;J[5,4]=-1
    R=mp.diag([eta*al]*4+[0,2*kap/wc]);A=(J-R)*K
    return A,K,J,R

def free_coefficients(cid):
    A,K,J,R=matrix(cid,M(0),M(261),M('.2'))
    A=A[:4,:4];J=J[:4,:4];R=R[:4,:4]
    gam=-sum(A[i,i] for i in range(4))/4
    A2=A*A;om=mp.sqrt(gam*gam-sum(A2[i,i] for i in range(4))/4)
    C=mp.matrix([1,0,1,0]);gg=M(str(private[cid]['gamma_rad_s_T']))/u
    drive=-(J-R)*C*gg
    ch=lambda z:(C.T*mp.lu_solve(z*mp.eye(4)-A,drive))[0]
    bb=(om*om+gam*gam)*ch(0)
    aa=(1+2*gam+om*om+gam*gam)*ch(1)-bb
    return {'Gamma_bar':dec(gam),'Omega_bar':dec(om),'A_tilde_per_T':dec(aa),'B_tilde_per_T':dec(bb),'rho_s':dec(aa/bb/u)}

free={cid:free_coefficients(cid) for cid in ['A','B','C']}
(W/'free_response_truth.json').write_text(json.dumps({'scope':'Post-prediction validation only: coefficients reconstructed by two direct free4D resolvent solves, not fitting input.','cases':free},indent=2),encoding='utf-8')

def coupling(cid,bf):
    d=private[cid];rr=M(str(d.get('truth_r',d.get('r'))))
    return bf*mp.sqrt(M(str(d['gamma_rad_s_T']))*mu/(2*hb))*mp.sqrt(rr)/u

def solve_full_ep(cid,kap,seed):
    om=M(free[cid]['Omega_bar'])
    def eq(bmicro,wc,er,ei):
        A,*_=matrix(cid,bmicro*M('1e-6'),wc,kap);z=mp.mpc(er,ei)
        fun=lambda z:mp.det(z*mp.eye(6)-A)
        f=fun(z)/om**6;d=mp.diff(fun,z)/om**5
        return f.real,f.imag,d.real,d.imag
    start=(M(seed['B_star_T'])*M('1e6'),M(seed['wc_GHz']),M(seed['pole']['real']),M(seed['pole']['imag']))
    sol=mp.findroot(eq,start,tol=M('1e-58'),maxsteps=35)
    bf,wc=sol[0]*M('1e-6'),sol[1];z=mp.mpc(sol[2],sol[3])
    assert bf>0 and wc>0 and z.real<0 and abs(z.imag)>1
    A,K,J,R=matrix(cid,bf,wc,kap);F=z*mp.eye(6)-A
    det=lambda v:mp.det(v*mp.eye(6)-A)
    resid=max(abs(x) for x in eq(*sol));dd=abs(mp.diff(det,z,2))/om**4
    # Any nonzero order-five minor plus a double root proves nullity one.
    mi=max(abs(mp.det(mp.matrix([[F[i,j] for j in range(6) if j!=c] for i in range(6) if i!=r])))/om**5 for r in range(6) for c in range(6))
    emin=min(mp.eigsy(K,eigvals_only=True));rmin=min(mp.eigsy(R,eigvals_only=True))
    assert resid<M('1e-45') and dd>M('1e-8') and mi>M('1e-8') and emin>0 and rmin>=0
    return {'B_star_T':dec(bf),'wc_GHz':dec(wc),'G_U_GHz':dec(coupling(cid,bf)),'pole':enc(z),'Q':dec(wc/(2*kap)),
      'certificate':{'normalized_p_and_dp_max':dec(resid),'normalized_second_derivative':dec(dd),'max_normalized_5_minor':dec(mi),'energy_min_eigenvalue':dec(emin),'dissipation_min_eigenvalue':dec(rmin)}}

def full_response(cid,bf,wc,kap):
    A,*_=matrix(cid,bf,wc,kap);ar=np.array(A.tolist(),dtype=float)
    rhs=np.array([0,0,0,0,0,1.],dtype=complex)
    return np.array([-2*float(etaext*kap)*np.linalg.solve(-1j*f*np.eye(6)-ar,rhs)[5] for f in frequencies])
def values(items):return np.array([complex(float(x['real']),float(x['imag'])) for x in items])
def compare_response(row,bf,wc,kap):
    full=full_response(row['case_id'],bf,wc,kap);ps=values(row['response_shape']);pc=values(row['response_strength_scalar'])
    peak=float(max(abs(full)));es=float(max(abs(ps-full)));ec=float(max(abs(pc-full)))
    return {'shape_abs':es,'scalar_abs':ec,'peak':peak,'shape_normalized':es/peak,'scalar_normalized':ec/peak,'error_ratio_shape_to_scalar':es/ec if ec>1e-20 else None},full

solved={};rows=[];curves={}
for row in pred['rows']:
    cid=row['case_id'];kap=M(row['kappa_GHz']);sp=row['shape_prediction']
    ep=solve_full_ep(cid,kap,sp);solved[(cid,row['kappa_GHz'])]=ep
    # Independent physical reference device: reference susceptibility strength
    # is rescaled by B_star; no damping term is deleted.
    ref=solved[('A',row['kappa_GHz'])]
    btarget=M(free[cid]['B_tilde_per_T']);bref=M(free['A']['B_tilde_per_T'])
    scalarB=M(ref['B_star_T'])*mp.sqrt(bref/btarget)
    sc={'B_star_T':dec(scalarB),'wc_GHz':ref['wc_GHz'],'G_U_GHz':dec(coupling(cid,scalarB))}
    errs={'fc_MHz':float((M(sp['wc_GHz'])-M(ep['wc_GHz']))*1000),
          'G_kHz':float((coupling(cid,M(sp['B_star_T']))-M(ep['G_U_GHz']))*M('1e6')),
          'B_T':float(M(sp['B_star_T'])-M(ep['B_star_T']))}
    frozen_scalar=row['strength_scalar_prediction']
    scalar_errors={'fc_MHz':float((M(frozen_scalar['wc_GHz'])-M(ep['wc_GHz']))*1000),
          'G_kHz':float((coupling(cid,M(frozen_scalar['B_star_T']))-M(ep['G_U_GHz']))*M('1e6')),
          'B_T':float(M(frozen_scalar['B_star_T'])-M(ep['B_star_T']))}
    eff={'fc_MHz':float((M(sc['wc_GHz'])-M(ep['wc_GHz']))*1000),
         'G_kHz':float((M(sc['G_U_GHz'])-M(ep['G_U_GHz']))*M('1e6')),
         'B_T':float(M(sc['B_star_T'])-M(ep['B_star_T']))}
    resp,full=compare_response(row,M(sp['B_star_T']),M(sp['wc_GHz']),kap)
    rows.append({'case_id':cid,'kappa_GHz':float(kap),'full_EP':ep,'shape_prediction_errors':errs,'strength_prediction_errors':scalar_errors,'strength_shape_effect':eff,'strength_matched_full_reference_EP':sc,'response':resp})
    curves[f'{cid}_{float(kap):.12f}']=full
    print(cid,float(kap),errs,resp,flush=True)
common=[]
for row in pred['common_G_responses']:
    p=row['physical_operating_point'];bf,wc,kap=[M(str(p[k])) for k in ['B_star_T','wc_GHz','kappa_GHz']]
    resp,full=compare_response(row,bf,wc,kap)
    common.append({'case_id':row['case_id'],'physical_operating_point':p,'reported_G_U_GHz':dec(coupling(row['case_id'],bf)),'response':resp})
    curves['common_'+row['case_id']]=full
np.savez_compressed(W/'full_response_curves.npz',frequency_GHz=frequencies,**curves)
assert sha(W/'predictions_frozen.json')==frozen_hash
out={'status':'completed','frozen_prediction_sha256':frozen_hash,'frozen_utc':pred['frozen_utc'],'validation_started_utc':started,
 'validator_sha256':sha(Path(__file__)),'scope':'Independent original-sublattice6D validation after public free-data-only prediction; numerical resolution gates evaluated separately in C4.',
 'units':'fc errorsMHz; per-branch G errorskHz; physical B in tesla. Rates labelled GHz mean angular rate/(2pi1GHz).',
 'rows':rows,'common_G_response':common,'coupled_refit_of_free_coefficients':False,'experimental_claim':False}
(W/'validation_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'EP_cases':len(rows),'common_cases':len(common)}))
