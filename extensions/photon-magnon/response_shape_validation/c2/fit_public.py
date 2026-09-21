"""Fit physical free-response coefficients using PUBLIC observations only.

No r, exchange, anisotropy, damping coefficient, target pole or analytic
numerator is supplied. Poles come from two independent free spin records.
Pulse coefficients come from a known-field convolution least-squares fit.
"""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
U=2*np.pi*1e9
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
READ=[]
PUB=None
OUT=None
def read_record(name):
    meta_path=PUB/(name+'.json');raw_path=PUB/(name+'.npz')
    meta=json.loads(meta_path.read_text(encoding='utf-8'));raw=np.load(raw_path,allow_pickle=False)
    assert sha(raw_path)==meta['raw_sha256']
    READ.extend([{'path':meta_path.relative_to(PUB).as_posix(),'sha256':sha(meta_path)},{'path':raw_path.relative_to(PUB).as_posix(),'sha256':sha(raw_path)}])
    return meta,{k:raw[k] for k in raw.files}

POLE_WINDOWS=[(0,.04e-9),(.005e-9,.032e-9),(.01e-9,.04e-9)]
FIT_WINDOWS=[(0,.08e-9),(.002e-9,.064e-9),(.01e-9,.08e-9)]
def fit_poles(records,window):
    minus=[];plus=[];ranks=[]
    for _,raw in records:
        x=raw['m6'][:,[0,1,3,4]];t=raw['t_s'];mask=(t>=window[0]-1e-24)&(t<=window[1]+1e-24)
        x=x[mask];scale=np.linalg.norm(x[0]);x=x/scale
        minus.append(x[:-1].T);plus.append(x[1:].T)
        ranks.append(int(np.linalg.matrix_rank(x,tol=np.linalg.norm(x)*1e-9)))
    X=np.concatenate(minus,axis=1);Y=np.concatenate(plus,axis=1)
    uu,sv,vh=np.linalg.svd(X,full_matrices=False)
    rank=int(np.count_nonzero(sv>sv[0]*1e-9));assert rank==4,(rank,sv)
    P=(Y@vh.T)@np.diag(1/sv)@uu.T
    dt=records[0][1]['t_s'][1]-records[0][1]['t_s'][0]
    poles=np.log(np.linalg.eigvals(P))/(dt*U)
    omega=float(np.mean(abs(poles.imag)));gamma=float(np.mean(-poles.real))
    assert omega>0 and gamma>0
    return {'window_s':list(window),'Gamma_bar':gamma,'Omega_bar':omega,
       'poles_bar':[[float(z.real),float(z.imag)] for z in poles],
       'joined_data_rank':rank,'individual_trace_ranks':ranks,'state_data_singular_values':sv.tolist(),
       'state_data_condition':float(sv[0]/sv[-1]),
       'relative_transition_residual':float(np.linalg.norm(Y-P@X)/np.linalg.norm(Y))}

def pulse_and_derivative_tau(tau,meta):
    t=tau/U;wave=meta['waveform'];x=(t-wave['start_s'])/wave['duration_s']
    if not 0<x<1:return 0.,0.
    f=wave['carrier_Hz'];duration=wave['duration_s'];amp=wave['amplitude_T']
    ss=np.sin(np.pi*x);cc=np.cos(np.pi*x);angle=2*np.pi*f*(t-wave['start_s'])
    B=amp*ss**4*np.cos(angle)
    dBdt=amp*(4*ss**3*cc*np.pi/duration*np.cos(angle)-ss**4*2*np.pi*f*np.sin(angle))
    return B,dBdt/U

def convolution_design(meta,raw,gamma,omega):
    # Green-function convolution realized as two driven second-order systems.
    # There is no differentiation of the observed magnetization.
    S=gamma*gamma+omega*omega
    def rhs(tau,q):
        B,dB=pulse_and_derivative_tau(tau,meta)
        return [q[1],B-2*gamma*q[1]-S*q[0],q[3],dB-2*gamma*q[3]-S*q[2]]
    tau=raw['t_s']*U
    sol=solve_ivp(rhs,(tau[0],tau[-1]),np.zeros(4),t_eval=tau,method='DOP853',rtol=2e-12,atol=1e-24,max_step=.0005)
    assert sol.success
    computed_B=np.array([pulse_and_derivative_tau(v,meta)[0] for v in tau])
    assert np.max(abs(computed_B-raw['B_T']))<1e-17
    return np.column_stack([sol.y[2],sol.y[0]])

def coefficients(design,raw,window):
    t=raw['t_s'];mask=(t>=window[0]-1e-24)&(t<=window[1]+1e-24)
    X=design[mask];y=raw['Mx'][mask];scale=np.linalg.norm(X,axis=0)
    normalized=X/scale;beta,_,rank,sv=np.linalg.lstsq(normalized,y,rcond=None);assert rank==2
    coef=beta/scale;res=y-X@coef;sigma2=float(res@res/(len(y)-2))
    covariance=np.linalg.inv(normalized.T@normalized)*sigma2/scale[:,None]/scale[None,:]
    A,B=coef;rho=A/B/U
    jac=np.array([1/B,-A/B**2])/U
    return {'window_s':list(window),'A_tilde_per_T':float(A),'B_tilde_per_T':float(B),
       'a_tilde_per_T_s':float(A*U),'b_tilde_per_T_s2':float(B*U*U),'rho_s':float(rho),
       'relative_residual_L2':float(np.linalg.norm(res)/np.linalg.norm(y)),
       'max_absolute_magnetization_residual':float(np.max(abs(res))),
       'design_column_norms':scale.tolist(),'normalized_design_singular_values':sv.tolist(),
       'normalized_design_condition':float(sv[0]/sv[-1]),'raw_design_condition':float(np.linalg.cond(X)),
       'formal_ols_covariance_A_B':covariance.tolist(),
       'formal_ols_rho_standard_error_s':float(np.sqrt(max(0,jac@covariance@jac))),
       'formal_uncertainty_scope':'Deterministic-fit residual diagnostic; not experimental noise or a confidence interval.'}

def main():
    global PUB,OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='Common run root; fit outputs are under response_shape_validation/c2/fit.')
    p.add_argument('--public',type=Path,help='Optional isolated public observation directory; no private inputs are read.')
    p.add_argument('--allow-partial',action='store_true',help='Development fit only; incomplete results cannot be frozen.')
    args=p.parse_args()
    root=args.output.resolve()/'response_shape_validation'/'c2'
    PUB=args.public.resolve() if args.public else root/'public'
    OUT=root/'fit'
    if not PUB.is_dir():raise RuntimeError('Public observations not found; run run_gpu.py first.')
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.allow_partial:
        required=[]
        for cid in ('A','B','C'):
            required += [f'{cid}_{v}_{k}' for v in ('halfstep','quarterstep') for k in ('seedA','seedB','pulse')]
            if cid in ('A','B'):
                required += [f'{cid}_quarter_halfamp_pulse']
                required += [f'{cid}_{v}_{k}' for v in ('eighthstep','eighthseedhalf') for k in ('seedA','seedB')]
        missing=[name for name in required if not all((PUB/(name+ext)).is_file() for ext in ('.json','.npz'))]
        if missing:raise RuntimeError('Missing final-protocol records: '+', '.join(missing))
    cases=[]
    for cid in ('A','B','C'):
        if not any((PUB/f'{cid}_{v}_pulse.npz').exists() for v in ('base','halfstep','quarterstep','eighthstep')):continue
        variants={}
        for variant in ('base','halfstep','quarterstep','eighthstep','eighthseedhalf'):
            names=[f'{cid}_{variant}_seedA',f'{cid}_{variant}_seedB']
            if not all((PUB/(n+'.npz')).exists() for n in names):continue
            records=[read_record(n) for n in names]
            polefits=[fit_poles(records,w) for w in POLE_WINDOWS]
            variants[variant]={'pole_fits':polefits}
        pole_order=[v for v in ('base','halfstep','quarterstep','eighthstep','eighthseedhalf') if v in variants]
        if not pole_order:raise RuntimeError(f'No pair of free-seed records for case {cid}')
        chosen_pole=pole_order[-1];pole=variants[chosen_pole]['pole_fits'][0]
        gamma=pole['Gamma_bar'];omega=pole['Omega_bar']
        pole_final_variants=(['quarterstep','eighthstep','eighthseedhalf'] if 'eighthseedhalf' in variants else pole_order[-2:])
        final_poles=[p for v in pole_final_variants for p in variants[v]['pole_fits']]
        pole_envelope={key:max(abs(p[key]-pole[key]) for p in final_poles) for key in ('Gamma_bar','Omega_bar')}
        pulse_records={}
        for variant in ('base','halfstep','halfamp','quarterstep','quarter_halfamp','eighthstep','eighth_halfamp'):
            name=f'{cid}_{variant}_pulse'
            if not (PUB/(name+'.json')).exists():continue
            meta,raw=read_record(name);pulse_records[variant]=(meta,raw)
            X=convolution_design(meta,raw,gamma,omega)
            fitlist=[coefficients(X,raw,w) for w in FIT_WINDOWS]
            variants.setdefault(variant,{})['coefficient_fits']=fitlist
            variants[variant]['coefficient_fixed_pole_variant']=chosen_pole
            np.savez_compressed(OUT/f'{name}_fit.npz',t_s=raw['t_s'],observed=raw['Mx'],predicted=X@np.array([fitlist[0]['A_tilde_per_T'],fitlist[0]['B_tilde_per_T']]))
        chosen=next(v for v in ('eighthstep','quarterstep','halfstep','base') if v in variants and 'coefficient_fits' in variants[v])
        fit=variants[chosen]['coefficient_fits'][0]
        all_poles=[p for v in variants.values() for p in v.get('pole_fits',[])]
        all_fits=[p for v in variants.values() for p in v.get('coefficient_fits',[])]
        envelope={key:max(abs(p[key]-pole[key]) for p in all_poles) for key in ('Gamma_bar','Omega_bar')}
        envelope.update({key:max(abs(p[key]-fit[key]) for p in all_fits) for key in ('A_tilde_per_T','B_tilde_per_T','rho_s')})
        ordered=[v for v in ('base','halfstep','quarterstep','eighthstep') if v in variants and 'coefficient_fits' in variants[v]]
        finest=ordered[-2:]
        if chosen=='eighthstep' and 'eighth_halfamp' in variants:finest.append('eighth_halfamp')
        if chosen=='quarterstep' and 'quarter_halfamp' in variants:finest.append('quarter_halfamp')
        final_fits=[p for v in finest for p in variants[v].get('coefficient_fits',[])]
        joint_fits=[]
        # Propagate the independent free-pole convergence envelope through all
        # retained pulse/step/amplitude/window fits, with no material truth inputs.
        for pulse_variant in finest:
            meta,raw=pulse_records[pulse_variant]
            for sg in (-1,1):
                for so in (-1,1):
                    gg=gamma+sg*pole_envelope['Gamma_bar'];oo=omega+so*pole_envelope['Omega_bar']
                    X=convolution_design(meta,raw,gg,oo)
                    for window in FIT_WINDOWS:
                        rec=coefficients(X,raw,window)
                        joint_fits.append({'pulse_variant':pulse_variant,'Gamma_bar':gg,'Omega_bar':oo,**rec})
        final_envelope=dict(pole_envelope)
        final_envelope.update({key:max(abs(p[key]-fit[key]) for p in final_fits+joint_fits) for key in ('A_tilde_per_T','B_tilde_per_T','rho_s')})
        cases.append({'case_id':cid,'chosen_variant':chosen,'chosen_pole_variant':chosen_pole,'Gamma_bar':pole['Gamma_bar'],'Omega_bar':pole['Omega_bar'],
          **{key:fit[key] for key in ('A_tilde_per_T','B_tilde_per_T','a_tilde_per_T_s','b_tilde_per_T_s2','rho_s')},
          'numerical_variation_envelope':final_envelope,'envelope_variants':finest,
          'pole_envelope_variants':pole_final_variants,'joint_pole_pulse_fit_variants':joint_fits,
          'all_step_convergence_history_envelope':envelope,'variants':variants})
        print(json.dumps({'case_id':cid,'chosen':chosen,'Gamma_bar':pole['Gamma_bar'],'Omega_bar':pole['Omega_bar'],'rho_s':fit['rho_s'],'normalized_condition':fit['normalized_design_condition'],'fit_residual':fit['relative_residual_L2']}),flush=True)
    result={'scope':'Independent physical free-response fit using only time, spin observations and applied Bx waveform; no coupled fit and no material truth inputs.',
     'unit_rate_rad_s':U,'units':{'Gamma_bar':'Gamma/u','Omega_bar':'Omega/u','A_tilde_per_T':'a_tilde/u [1/T]','B_tilde_per_T':'b_tilde/u^2 [1/T]','rho_s':'A_tilde/B_tilde/u [s]'},
     'cases':cases,'public_read_manifest':READ,'public_paths_relative_to':'supplied public directory','analyzer_sha256':sha(__file__),'final_records_required':not args.allow_partial,
     'formula_used':'Dbar Mx = A_tilde dB/dtau + B_tilde B; tau=u*t. Poles estimated by unconstrained 4-channel DMD of two free seeds.'}
    if {'A','B'}.issubset({c['case_id'] for c in cases}):
        pair={c['case_id']:c for c in cases};effect=abs(pair['B']['rho_s']-pair['A']['rho_s'])
        error=max(pair[c]['numerical_variation_envelope']['rho_s'] for c in ('A','B'))
        result['local_shape_resolution']={'delta_rho_s':effect,'maximum_variation_s':error,'variation_over_effect':error/effect,'predeclared_limit':.1,'pass':error<.1*effect}
    (OUT/'calibration_public.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
