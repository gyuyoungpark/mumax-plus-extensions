"""C4: numerical calibration bounds and explicitly diagnostic nuisance fits.

Run only after freezing C2 calibration and completing C3 validation. Does not mutate calibration,
predictions, validators or manuscripts. Coupled truth is used only here to test
the already frozen scalar/full-shape predictions, never to refit the predictor.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import itertools
import json
import math
import numpy as np
import mpmath as mp
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
U = 2*np.pi*1e9
PREDECLARED_RATIO = .1


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def enc(value):
    if isinstance(value, dict): return {k: enc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [enc(v) for v in value]
    if isinstance(value, np.ndarray): return enc(value.tolist())
    if isinstance(value, np.generic): return value.item()
    return value


def cvec(z):
    return np.concatenate([z.real, z.imag])


def metrics(pred, target):
    diff=pred-target
    peak=float(np.max(abs(target)))
    return {'max_absolute': float(np.max(abs(diff))),
            'peak_full': peak,
            'epsilon_S_max': float(np.max(abs(diff))/peak),
            'complex_L2_normalized': float(np.linalg.norm(diff)/np.linalg.norm(target))}


def scalar_response(params, z, factor, eta_external, derivatives=False):
    """Parameter order: Gamma, Omega, physical btilde, rho_bar, kappa, wc, Bstar.

    All rates are divided by U, btilde by U^2, and rho_bar = U*rho_s.
    The input/output convention agrees with the declared cavity ports.
    """
    gam, om, bb, rho, kap, wc, bstar = params
    D=z*z+2*gam*z+om*om+gam*gam
    N=bb*(1+rho*z)
    chi=N/D
    L=factor*wc*bstar*bstar
    den=z*z+2*kap*z+wc*wc-L*chi
    resp=-2*eta_external*kap*z/den
    if not derivatives: return resp
    dchi_g=-N*(2*z+2*gam)/(D*D)
    dchi_o=-N*(2*om)/(D*D)
    dchi_b=(1+rho*z)/D
    dchi_r=bb*z/D
    dden=np.column_stack([-L*dchi_g,-L*dchi_o,-L*dchi_b,-L*dchi_r,
                         2*z, 2*wc-factor*bstar*bstar*chi,
                         -2*factor*wc*bstar*chi])
    jac=-resp[:,None]/den[:,None]*dden
    jac[:,4]+=resp/kap
    return resp,jac


def full6_response(truth, point, z, settings):
    """Independent original sublattice generator, built from declared energy.

    No response numerator, fitted calibration, or reduced model enters this
    matrix. The uniform Zeeman Hessian element is written directly in Bstar.
    """
    gamma=truth['gamma_rad_s_T']; alpha=truth['alpha']
    hA=gamma*truth['BA_T']/U; hE=gamma*truth['BE_T']/U
    kap=float(point['kappa_GHz']); wc=float(point['wc_GHz'])
    bf=float(point['B_star_T']); eta=1/(1+alpha*alpha)
    K=np.zeros((6,6))
    K[np.arange(4),np.arange(4)]=hA+hE
    K[0,2]=K[2,0]=K[1,3]=K[3,1]=hE
    K[4,4]=K[5,5]=wc
    coupling=bf*np.sqrt(2*gamma*settings['mu_s_J_per_T']/settings['hbar_J_s'])/U
    K[0,4]=K[4,0]=K[2,4]=K[4,2]=-coupling
    J=np.zeros((6,6)); J[0,1]=-eta;J[1,0]=eta
    J[2,3]=eta;J[3,2]=-eta;J[4,5]=1;J[5,4]=-1
    R=np.diag([eta*alpha]*4+[0,2*kap/wc])
    A=(J-R)@K
    rhs=np.zeros(6);rhs[5]=1
    state=np.linalg.solve(z[:,None,None]*np.eye(6)-A,np.broadcast_to(rhs,(len(z),6))[...,None])[...,0]
    response=-2*settings['eta_external']*kap*state[:,5]
    checks={'energy_min_eigenvalue':float(np.linalg.eigvalsh(K).min()),
            'max_generator_real_eigenvalue':float(np.linalg.eigvals(A).real.max()),
            'relative_Lyapunov_error':float(np.linalg.norm(A.T@K+K@A+2*K@R@K)/(np.linalg.norm(A)*np.linalg.norm(K)))}
    assert checks['energy_min_eigenvalue']>0
    assert checks['max_generator_real_eigenvalue']<0
    return response,checks


def bounded_multistart(center, widths, varying, z, target, factor, eta, starts):
    peak=np.max(abs(target)); widths=np.asarray(widths,float)
    def model(x, derivatives=False):
        p=center.copy();p[varying]+=widths*x
        return scalar_response(p,z,factor,eta,derivatives)
    def fun(x): return cvec((model(x)-target)/peak)
    def jac(x):
        _,j=model(x,True);j=j[:,varying]*widths[None,:]/peak
        return np.vstack([j.real,j.imag])
    candidates=[]
    for start in starts:
        sol=least_squares(fun,start,jac=jac,bounds=(-np.ones(len(varying)),np.ones(len(varying))),
                          ftol=1e-13,xtol=1e-13,gtol=1e-15,max_nfev=1500)
        candidates.append((float(sol.fun@sol.fun),sol))
    _,best=min(candidates,key=lambda s:s[0])
    p=center.copy();p[varying]+=widths*best.x
    result={'normalized_box_coordinates':best.x.tolist(),'parameter_values':p.tolist(),
            'parameter_shifts':(widths*best.x).tolist(),
            'metrics':metrics(model(best.x),target),
            'start_count':len(starts),'successful_starts':sum(s.success for _,s in candidates),
            'best_status':int(best.status),'best_message':best.message,
            'best_optimality':float(best.optimality),
            'minimum_LS_cost_found':float(best.cost),
            'start_LS_cost_range':[float(min(s.cost for _,s in candidates)),float(max(s.cost for _,s in candidates))],
            'global_optimality_certified':False,
            'objective':'Complex least squares normalized by full-response peak; epsilon_S_max evaluated separately.'}
    return result,model(best.x)


def derivative_check(center,z,factor,eta):
    """Scaled directional derivative check, not a statistical assertion."""
    _,j=scalar_response(center,z,factor,eta,True)
    steps=np.array([1e-5,1e-4,1e-2,1e-8,1e-5,1e-4,center[6]*1e-4])
    errors=[]
    for k,h in enumerate(steps):
        p=center.copy();m=center.copy();p[k]+=h;m[k]-=h
        fd=(scalar_response(p,z,factor,eta)-scalar_response(m,z,factor,eta))/(2*h)
        errors.append(float(np.max(abs(fd-j[:,k]))/max(np.max(abs(j[:,k])),1e-30)))
    assert max(errors)<2e-6,errors
    return errors


def diagnostic_one_nuisance(center,z,target,factor,eta,gu):
    """Deliberately coupled-data fits: these never alter the frozen prediction.

    Search spans are numerical diagnostic domains, not empirical uncertainties.
    Single-nuisance best fits cannot prove global identifiability or detection.
    """
    ans=[]
    starts=[np.array([x]) for x in [-.8,-.2,0,.2,.8]]
    for name,index,span,unit in [('delta_Omega',1,.05,'GHz'),('delta_Gamma',0,.05,'GHz'),
                                ('delta_wc',5,.05,'GHz'),('delta_kappa',4,.05,'GHz')]:
        result,_=bounded_multistart(center,np.array([span]),[index],z,target,factor,eta,starts)
        result.update({'nuisance':name,'shift':result['parameter_shifts'][0],'unit':unit,
                       'diagnostic_search_halfwidth':span,'search_bound_is_empirical':False})
        if name=='delta_kappa':
            result['port_assumption']='Fixed eta_external: numerator port normalization scales with total kappa. This is not an internal-loss-only perturbation at fixed external port rates.'
        ans.append(result)
    # At fixed r the relative G change equals the relative physical-field change.
    result,_=bounded_multistart(center,np.array([center[6]*.05/gu]),[6],z,target,factor,eta,starts)
    result.update({'nuisance':'delta_G_U','shift':result['parameter_shifts'][0]/center[6]*gu,
                   'unit':'GHz','equivalent_Bstar_shift_T':result['parameter_shifts'][0],
                   'diagnostic_search_halfwidth':.05,'search_bound_is_empirical':False,
                   'normalization':'G computed from truth r solely to report a diagnostic coupling shift; predictor remains physical-Bstar based.'})
    ans.append(result)
    model=scalar_response(center,z,factor,eta)
    gain=np.vdot(model,target)/np.vdot(model,model)
    # Independent real/imaginary box of +/-0.02 around unit complex gain.
    allowed_gain=complex(np.clip(gain.real,.98,1.02),np.clip(gain.imag,-.02,.02))
    ans.append({'nuisance':'constant_complex_gain','gain_real':allowed_gain.real,'gain_imag':allowed_gain.imag,
                'amplitude_shift':abs(allowed_gain)-1,'phase_shift_deg':float(np.angle(allowed_gain,deg=True)),
                'unconstrained_LS_gain':{'real':gain.real,'imag':gain.imag},
                'diagnostic_real_imag_halfwidth':.02,'search_bound_is_empirical':False,
                'metrics':metrics(allowed_gain*model,target),'gain_changes_EP_coordinates':False,
                'objective':'Analytic constant-complex-gain LS solution, clipped to the declared diagnostic rectangle.'})
    return ans


def criterion(error,effect):
    error=abs(float(error));effect=abs(float(effect))
    return {'absolute_error':error,'absolute_shape_effect':effect,
            'error_over_effect':error/effect if effect else None,
            'internal_limit':PREDECLARED_RATIO,
            'pass':bool(error<PREDECLARED_RATIO*effect) if effect else None,
            'scope':'Predeclared numerical-resolution target; not an experimental or statistical criterion.'}


def physical_ep(coefs,kap,settings):
    """Propagate calibrated coefficients; not an independent EP certificate."""
    with mp.workdps(65):
        gam,om,aa,bb=[mp.mpf(str(v)) for v in coefs]
        kap=mp.mpf(str(kap));rho=aa/bb;delta=gam-kap
        x=2*delta*rho*om*om/(mp.sqrt((1-rho*gam)**2+(rho*om)**2)+1-rho*gam)
        wc=mp.sqrt(om*om+kap*kap+x)
        H=om*om*delta*delta+gam*delta*x-x*x/4
        factor=2*mp.mpf(str(settings['mu_s_J_per_T']))/(mp.mpf(str(settings['hbar_J_s']))*2*mp.pi*mp.mpf('1e9'))
        return [mp.nstr(mp.sqrt(H/(wc*factor*bb)),55),mp.nstr(wc,55)]


def propagate_envelope(c,kap,point,z,full,settings,truth,shape_effect=None):
    keys=['Gamma_bar','Omega_bar','A_tilde_per_T','B_tilde_per_T']
    center=[c[k] for k in keys];widths=[c['numerical_variation_envelope'][k] for k in keys]
    center_ep=physical_ep(center,kap,settings)
    factor=2*settings['mu_s_J_per_T']/(settings['hbar_J_s']*U)
    def pars(v):
        gam,om,aa,bb=map(float,v)
        return np.array([gam,om,bb,aa/bb,point['kappa_GHz'],point['wc_GHz'],point['B_star_T']])
    center_response=scalar_response(pars(center),z,factor,settings['eta_external'])
    gfactor=np.sqrt(truth['gamma_rad_s_T']*settings['mu_s_J_per_T']/(2*settings['hbar_J_s']))*np.sqrt(truth['r'])/U
    bounds={'B_T':0.,'fc_MHz':0.,'G_kHz':0.,'response_normalized':0.}
    records=[]
    with mp.workdps(65):
        for signs in itertools.product([-1,1],repeat=4):
            v=[mp.nstr(mp.mpf(str(a))+s*mp.mpf(str(b)),55) for a,b,s in zip(center,widths,signs)]
            assert all(mp.mpf(t)>0 for t in v)
            ep=physical_ep(v,kap,settings)
            bdiff=float(abs(mp.mpf(ep[0])-mp.mpf(center_ep[0])))
            fdiff=float(abs(mp.mpf(ep[1])-mp.mpf(center_ep[1]))*1000)
            gdiff=bdiff*gfactor*1e6
            response=scalar_response(pars(v),z,factor,settings['eta_external'])
            sdiff=float(np.max(abs(response-center_response))/np.max(abs(full)))
            diffs={'B_T':bdiff,'fc_MHz':fdiff,'G_kHz':gdiff,'response_normalized':sdiff}
            for k,vv in diffs.items():bounds[k]=max(bounds[k],vv)
            records.append({'corner_signs':signs,'EP_B_T':ep[0],'EP_wc_GHz':ep[1],'deviations_from_center':diffs})
    ans={'case_id':c['case_id'],'kappa_GHz':float(kap),'parameter_order':keys,
         'center':center,'halfwidths':widths,'corner_count':16,
         'center_EP':{'B_star_T':center_ep[0],'wc_GHz':center_ep[1]},
         'response_operating_point':point,'maximum_corner_deviation_from_center':bounds,
         'interpretation':'Deterministic independent enclosing-box corner sensitivity; covariance discarded, no statistical coverage and no certified global bound over the continuous box.',
         'rho_varied_independently':False,'corner_results':records}
    if shape_effect:
        ans['resolution_gates']={k:criterion(bounds[k],shape_effect[k]) for k in bounds}
    else:
        ans['resolution_gates']='No EP-coordinate effect ratios requested in this call; absolute corner sensitivities are reported. Reference A has zero shape effect.'
    return ans


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final-inputs-confirmed',action='store_true')
    args=parser.parse_args()
    assert args.final_inputs_confirmed,'Use --final-inputs-confirmed only after freezing C2 and validating C3.'
    if not (HERE/'protocol.json').exists():
        (HERE/'protocol.json').write_text(json.dumps({'scope':'Numerical envelope and nuisance diagnostics; not experimental identifiability.', 'error_over_effect_limit':PREDECLARED_RATIO},indent=2)+'\n',encoding='utf-8')
    files={'calibration':BASE/'c3/calibration_public_frozen.json',
           'settings':BASE/'c3/prediction_settings.json','predictions':BASE/'c3/predictions_frozen.json',
           'validation':BASE/'c3/validation_results.json','free_truth':BASE/'c3/free_response_truth.json',
           'material_truth':BASE/'c2/private/truth_by_case.json'}
    data={k:json.loads(p.read_text(encoding='utf-8')) for k,p in files.items()}
    cal=data['calibration'];settings=data['settings'];validation=data['validation']
    predictions=data['predictions']; truths=data['material_truth']['cases']
    assert predictions['read_files'][files['calibration'].name]==sha(files['calibration'])
    assert predictions['read_files'][files['settings'].name]==sha(files['settings'])
    cases={c['case_id']:c for c in cal['cases']}
    ft=data['free_truth'];free_cases=ft.get('cases',ft)
    if isinstance(free_cases,list):free_cases={r['case_id']:r for r in free_cases}
    true_rho_ref=float(free_cases['A']['A_tilde_per_T'])/float(free_cases['A']['B_tilde_per_T'])
    ref=cases['A'];rho_ref=ref['A_tilde_per_T']/ref['B_tilde_per_T']
    freq=np.array(predictions['frequency_GHz'],float);z=-1j*freq
    factor=2*settings['mu_s_J_per_T']/(settings['hbar_J_s']*U)
    eta=settings['eta_external']
    rng=np.random.default_rng(20260920)
    starts=[np.zeros(4)]+[np.array(v)*.95 for v in itertools.product([-1,1],repeat=4)]
    starts += [rng.uniform(-.9,.9,4) for _ in range(16)]
    results=[];arrays={'frequency_GHz':freq}
    for cid in ['A','B']:
        c=cases[cid];point=settings['common_G_operating_points'][cid]
        center=np.array([c['Gamma_bar'],c['Omega_bar'],c['B_tilde_per_T'],rho_ref,
                         point['kappa_GHz'],point['wc_GHz'],point['B_star_T']],float)
        envelope=c['numerical_variation_envelope'];refenv=ref['numerical_variation_envelope']
        widths=np.array([envelope['Gamma_bar'],envelope['Omega_bar'],envelope['B_tilde_per_T'],refenv['rho_s']*U])
        assert np.all(widths>0),'Final independent envelope must supply all four nonzero widths.'
        full,fullchecks=full6_response(truths[cid],point,z,settings)
        scalar=scalar_response(center,z,factor,eta)
        shapepars=center.copy();shapepars[3]=c['A_tilde_per_T']/c['B_tilde_per_T']
        shape=scalar_response(shapepars,z,factor,eta)
        jac_check=derivative_check(center,z,factor,eta)
        bounded,fit=bounded_multistart(center,widths,[0,1,2,3],z,full,factor,eta,starts)
        gu=point['B_star_T']*np.sqrt(truths[cid]['gamma_rad_s_T']*settings['mu_s_J_per_T']/(2*settings['hbar_J_s']))*np.sqrt(truths[cid]['r'])/U
        diagnostics=diagnostic_one_nuisance(center,z,full,factor,eta,gu)
        baseline=metrics(scalar,full);shape_metric=metrics(shape,full)
        truthpars=center.copy()
        truthpars[:4]=[float(free_cases[cid]['Gamma_bar']),float(free_cases[cid]['Omega_bar']),
                       float(free_cases[cid]['B_tilde_per_T']),true_rho_ref]
        truth_scalar_metric=metrics(scalar_response(truthpars,z,factor,eta),full)
        prior=next(v for v in validation['common_G_response'] if v['case_id']==cid)['response']
        assert abs(shape_metric['epsilon_S_max']-prior['shape_normalized'])<1e-10
        assert abs(baseline['epsilon_S_max']-prior['scalar_normalized'])<1e-10
        row={'case_id':cid,'role':'reference calibration control' if cid=='A' else 'representative local pair B versus A reference shape',
             'operating_point':point,'G_U_GHz_reporting_only':gu,
             'parameter_order':['Gamma_bar','Omega_bar','B_tilde_per_T','rho_ref_bar','kappa_GHz','wc_GHz','Bstar_T'],
             'baseline_parameters':center,'conservative_independent_box_halfwidths':widths,
             'box_parameters':['Gamma_bar_target','Omega_bar_target','B_tilde_target','rho_ref_bar_A'],
             'box_provenance':{'target_variant':c['chosen_variant'],'target_envelope_variants':c['envelope_variants'],
                               'reference_variant':ref['chosen_variant'],'reference_envelope_variants':ref['envelope_variants']},
             'box_ignores_parameter_covariance':True,
             'box_empirical_confidence_interval':False,
             'exact_declared_inputs':['kappa','wc','Bstar','mu_s','input/output gain'],
             'independent_full6_checks':fullchecks,'analytic_jacobian_relative_errors':jac_check,
             'strength_scalar_baseline':baseline,'full_shape_frozen':shape_metric,
             'exact_strength_shape_effect_at_common_point':truth_scalar_metric,
             'C3_common_response_summary_agreement':True,
             'bounded_scalar_fit':bounded,'one_nuisance_coupled_diagnostics':diagnostics,
             'bounded_fit_residual_over_baseline':bounded['metrics']['epsilon_S_max']/baseline['epsilon_S_max'],
             'bounded_fit_residual_over_shape_prediction_error':bounded['metrics']['epsilon_S_max']/shape_metric['epsilon_S_max'],
             'experimental_identifiability_decidable':False}
        results.append(row)
        for name,value in [('full',full),('scalar',scalar),('shape',shape),('bounded_scalar',fit)]:arrays[f'{cid}_{name}']=value

    # Summaries consume C3's independent coupled validation, never feed it back.
    gates=[];ep_propagation=[]
    for row in validation['rows']:
        cid=row['case_id'];response=row['response']
        gate={'case_id':cid,'kappa_GHz':row['kappa_GHz']}
        if cid=='A':
            gate['reference_shape_effect_zero']='No error/effect ratio at reference A; direct prediction errors are still reported.'
            gate['prediction_errors']=row['shape_prediction_errors']
        else:
            gate['fc_MHz']=criterion(row['shape_prediction_errors']['fc_MHz'],row['strength_shape_effect']['fc_MHz'])
            gate['G_kHz']=criterion(row['shape_prediction_errors']['G_kHz'],row['strength_shape_effect']['G_kHz'])
            gate['B_T']=criterion(row['shape_prediction_errors']['B_T'],row['strength_shape_effect']['B_T'])
        pr=next(v for v in predictions['rows'] if v['case_id']==cid and abs(float(v['kappa_GHz'])-float(row['kappa_GHz']))<1e-12)
        point={'B_star_T':float(pr['shape_prediction']['B_star_T']),
               'wc_GHz':float(pr['shape_prediction']['wc_GHz']),'kappa_GHz':float(row['kappa_GHz'])}
        full,_=full6_response(truths[cid],point,z,settings)
        t=free_cases[cid]
        truthpars=np.array([float(t['Gamma_bar']),float(t['Omega_bar']),float(t['B_tilde_per_T']),
                            true_rho_ref,point['kappa_GHz'],point['wc_GHz'],point['B_star_T']])
        pure_effect=metrics(scalar_response(truthpars,z,factor,eta),full)
        if cid!='A':
            gate['response']=criterion(response['shape_normalized'],pure_effect['epsilon_S_max'])
            gate['C3_fitted_scalar_response_error']=response['scalar_normalized']
            gate['pure_response_shape_effect']=pure_effect
        effect={**row['strength_shape_effect'],'response_normalized':pure_effect['epsilon_S_max']} if cid!='A' else None
        ep_propagation.append(propagate_envelope(cases[cid],row['kappa_GHz'],point,z,full,settings,truths[cid],effect))
        gates.append(gate)
    common_propagation=[]
    for cid,c in cases.items():
        point=settings['common_G_operating_points'][cid]
        full,_=full6_response(truths[cid],point,z,settings)
        t=free_cases[cid]
        truthpars=np.array([float(t['Gamma_bar']),float(t['Omega_bar']),float(t['B_tilde_per_T']),
                            true_rho_ref,point['kappa_GHz'],point['wc_GHz'],point['B_star_T']])
        pure_effect=metrics(scalar_response(truthpars,z,factor,eta),full)
        entry=propagate_envelope(c,point['kappa_GHz'],point,z,full,settings,truths[cid])
        if cid!='A':
            entry['common_response_resolution_gate']=criterion(entry['maximum_corner_deviation_from_center']['response_normalized'],pure_effect['epsilon_S_max'])
            actual=next(v for v in validation['common_G_response'] if v['case_id']==cid)['response']['shape_normalized']
            entry['actual_prediction_response_resolution_gate']=criterion(actual,pure_effect['epsilon_S_max'])
        common_propagation.append(entry)
    calibration_checks=[]
    for cid,c in cases.items():
        t=free_cases[cid]
        vals={}
        for key in ['Gamma_bar','Omega_bar','A_tilde_per_T','B_tilde_per_T']:
            error=float(c[key])-float(t[key]);width=c['numerical_variation_envelope'][key]
            vals[key]={'fit_minus_truth':error,'numerical_envelope':width,'truth_inside_numerical_envelope':abs(error)<=width}
            if key in ['Gamma_bar','Omega_bar']:vals[key]['absolute_error_Hz']=abs(error)*1e9
        rho_truth=float(t['A_tilde_per_T'])/float(t['B_tilde_per_T'])/U
        vals['rho_s']={'fit_minus_truth':c['rho_s']-rho_truth,'numerical_envelope':c['numerical_variation_envelope']['rho_s']}
        calibration_checks.append({'case_id':cid,'checks':vals})
    delta_rho=abs(cases['B']['rho_s']-cases['A']['rho_s'])
    max_rho_env=max(cases[k]['numerical_variation_envelope']['rho_s'] for k in ['A','B'])
    max_rho_error=max(abs(x['checks']['rho_s']['fit_minus_truth']) for x in calibration_checks if x['case_id'] in ['A','B'])
    out={'scope':'C4 numerical calibration bounds and coupled-data diagnostics, not experimental identifiability.',
         'created_utc':datetime.now(timezone.utc).isoformat(),'input_sha256':{k:sha(p) for k,p in files.items()},
         'script_sha256':sha(Path(__file__)),'protocol_sha256':sha(HERE/'protocol.json'),
         'frozen_predictor_or_calibration_modified':False,
         'coupled_truth_used_only_for_validation_and_diagnostic_nuisance_fits':True,
         'calibration_checks':calibration_checks,
         'local_rho_variation_gate':criterion(max_rho_env,delta_rho),
         'local_rho_truth_error_gate':criterion(max_rho_error,delta_rho),
         'matched_free_pole_gate':'Gamma/Omega have zero cross-family shape effect by construction; report absolute calibration errors and their propagated EP/response errors, not a ratio to zero.',
         'representative_pair_results':results,'C3_coordinate_resolution_gates':gates,
         'propagated_EP_and_response_envelopes':ep_propagation,
         'propagated_common_G_response_envelopes':common_propagation,
         'common_G_response_resolution_gates':[
             {'case_id':r['case_id'],'response':r['actual_prediction_response_resolution_gate']}
             for r in common_propagation if r['case_id']!='A'],
         'limits':['Envelope derives from deterministic step/amplitude/window variations; it is not a statistical confidence interval.',
                   'The independent box discards covariance and is conservative relative to observed parameter tuples.',
                   'Multistart LS is a bounded numerical test, not a certified global impossibility theorem.',
                   'One-nuisance search spans are hypothetical diagnostic domains, not empirical instrument calibration bounds.',
                   'Diagnostic parameter variations are transfer-response nuisance models; they do not certify a new material path or a physical reference device at every search point.',
                   'Constant complex gain can change measured response but cannot move EP coordinates.',
                   'Hypothetical delta_kappa keeps eta_external fixed, so the response numerator scales with kappa; fixed external ports would be a different nuisance convention.',
                   'Actual instrument/noise/phase drift data are unavailable; experimental distinguishability is not established.']}
    nonref=[g for g in gates if g['case_id']!='A']
    prop_nonref=[g for g in ep_propagation if g['case_id']!='A']
    out['summary']={
        'actual_prediction':{key:{'maximum_absolute_error':max(abs(v['shape_prediction_errors'][key]) for v in validation['rows']),
                                 'maximum_error_over_effect':max(g[key]['error_over_effect'] for g in nonref),
                                 'all_nonreference_pass':all(g[key]['pass'] for g in nonref)} for key in ['fc_MHz','G_kHz','B_T']},
        'actual_EP_response_max_error_over_effect':max(g['response']['error_over_effect'] for g in nonref),
        'propagated_EP':{key:{'maximum_corner_deviation':max(g['maximum_corner_deviation_from_center'][key] for g in ep_propagation),
                             'maximum_deviation_over_effect':max(g['resolution_gates'][key]['error_over_effect'] for g in prop_nonref),
                             'all_nonreference_pass':all(g['resolution_gates'][key]['pass'] for g in prop_nonref)} for key in ['fc_MHz','G_kHz','B_T','response_normalized']},
        'actual_common_response_max_error_over_effect':max(g['actual_prediction_response_resolution_gate']['error_over_effect'] for g in common_propagation if g['case_id']!='A'),
        'propagated_common_response_max_deviation_over_effect':max(g['common_response_resolution_gate']['error_over_effect'] for g in common_propagation if g['case_id']!='A'),
        'any_numerical_resolution_failure':any(not g[key]['pass'] for g in nonref for key in ['fc_MHz','G_kHz','B_T','response']) or
            any(not g['resolution_gates'][key]['pass'] for g in prop_nonref for key in ['fc_MHz','G_kHz','B_T','response_normalized']) or
            any(not g['common_response_resolution_gate']['pass'] or not g['actual_prediction_response_resolution_gate']['pass'] for g in common_propagation if g['case_id']!='A')}
    (HERE/'results.json').write_text(json.dumps(enc(out),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    np.savez_compressed(HERE/'response_curves.npz',**arrays)
    with (HERE/'representative_pair_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['case_id','scalar_epsilon_S','shape_epsilon_S','bounded_scalar_epsilon_S','bounded_over_shape_error'])
        for r in results:w.writerow([r['case_id'],r['strength_scalar_baseline']['epsilon_S_max'],r['full_shape_frozen']['epsilon_S_max'],r['bounded_scalar_fit']['metrics']['epsilon_S_max'],r['bounded_fit_residual_over_shape_prediction_error']])
    report(out)
    manuscript_values(out)
    # Verify the prediction inputs remained bit-identical across this diagnostic run.
    assert all(sha(p)==out['input_sha256'][k] for k,p in files.items())
    print(json.dumps({'pair':[{r['case_id']:r['bounded_scalar_fit']['metrics']['epsilon_S_max']} for r in results],
                      'rho_variation_gate':out['local_rho_variation_gate'],'output':str(HERE/'results.json')}))


def report(out):
    rows=out['representative_pair_results']
    lines=['# C4: numerical error and calibration-bound test','',
           'This report uses final frozen C2/C3 inputs. Coupled truth is used only for validation and explicitly labeled diagnostic fits; the predictor and its coefficients remain unchanged.','',
           'The scalar model has the target fitted strength and A-reference fitted shape. Its four nuisance axes are target Gamma, Omega and physical Btilde, plus reference rho. Their halfwidths come only from the final C2 numerical-variation envelope. The independent box discards fit covariance and is therefore conservative. Cavity/field/port inputs are exact declared simulation settings, with no empirical uncertainty assigned.','',
           '| Case | Scalar epsilon_S | Full-shape epsilon_S | Bounded scalar epsilon_S | Bounded residual / shape prediction error |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['case_id']} | {r['strength_scalar_baseline']['epsilon_S_max']:.8g} | {r['full_shape_frozen']['epsilon_S_max']:.8g} | {r['bounded_scalar_fit']['metrics']['epsilon_S_max']:.8g} | {r['bounded_fit_residual_over_shape_prediction_error']:.6g} |")
    lines+=['','A is the reference control: its scalar and shape coefficients coincide, so no nonzero shape effect is expected. B is the requested local pair relative to A. The objective is complex least squares; the table reports the separately evaluated max-norm response metric. The bounded search uses 33 starts and analytic response derivatives; it does not certify a global lower bound.','',
            '## One-nuisance diagnostics','',
            'The following fits use coupled data and are diagnostics only. Rates and shifts quoted in GHz mean the angular rate divided by 2pi. Rate/coupling search spans are +/-0.05 GHz; the constant-gain real/imaginary search rectangle is +/-0.02 about unit gain. These are declared numerical search domains, not instrument specifications or independently measured allowed errors. Each other parameter stays at the frozen scalar baseline.','',
            '| Case | Nuisance | Best shift found | Unit | Remaining epsilon_S |','|---|---|---:|---|---:|']
    for r in rows:
        for d in r['one_nuisance_coupled_diagnostics']:
            if d['nuisance']=='constant_complex_gain':
                value=f"amplitude {d['amplitude_shift']:.6g}; phase {d['phase_shift_deg']:.6g} deg";unit='dimensionless / degree'
            else:value=f"{d['shift']:.9g}";unit=d['unit']
            lines.append(f"| {r['case_id']} | {d['nuisance']} | {value} | {unit} | {d['metrics']['epsilon_S_max']:.8g} |")
    lines+=['','A best shift is not a proof that a single nuisance fully absorbs the difference: the remaining full-window residual is retained in the table. Constant gain does not change EP coordinates. Multi-parameter tradeoffs and actual experimental drift/noise require independent empirical bounds before an experimental discrimination claim can be assessed.','',
            '## Resolution relative to the residual shape effect','',
            f"Local rho envelope/effect: {out['local_rho_variation_gate']['error_over_effect']:.8g}; pass below 0.1: {out['local_rho_variation_gate']['pass']}.",
            f"Local rho truth-error/effect: {out['local_rho_truth_error_gate']['error_over_effect']:.8g}; pass below 0.1: {out['local_rho_truth_error_gate']['pass']}.",'',
            '| Case | kappa (GHz) | fc error/effect | G error/effect | response error/effect |','|---|---:|---:|---:|---:|']
    for g in out['C3_coordinate_resolution_gates']:
        if g['case_id']=='A':continue
        lines.append(f"| {g['case_id']} | {float(g['kappa_GHz']):.9g} | {g['fc_MHz']['error_over_effect']:.8g} | {g['G_kHz']['error_over_effect']:.8g} | {g['response']['error_over_effect']:.8g} |")
    lines+=['','The 0.1 condition is a predeclared internal numerical-resolution target, not a journal criterion or statistical significance level. Gamma and Omega are matched across the mathematical family, so their cross-family effect is zero; absolute fit errors and propagation into EP/response are reported instead. Exact reference A also has zero shape effect and is excluded from ratios to effect. Per-coordinate pass/fail, absolute errors, bounds, hashes and all solver diagnostics are in results.json.','',
            '## Propagation of the numerical calibration envelope','',
            'Separately from the actual center-prediction error, the 16 corners of the independent Gamma/Omega/Atilde/Btilde enclosing box are propagated. Rho is derived as Atilde/Btilde at each corner and is not an extra independent coordinate. The table gives the largest evaluated corner deviation from the frozen center, divided by the ideal scalar shape effect. This is a deterministic corner sensitivity test, not a certified maximum over the continuous box or a confidence interval.','',
            '| Case | kappa (GHz) | fc envelope/effect | G envelope/effect | response envelope/effect |','|---|---:|---:|---:|---:|']
    for e in out['propagated_EP_and_response_envelopes']:
        if e['case_id']=='A':continue
        g=e['resolution_gates']
        lines.append(f"| {e['case_id']} | {e['kappa_GHz']:.9g} | {g['fc_MHz']['error_over_effect']:.8g} | {g['G_kHz']['error_over_effect']:.8g} | {g['response_normalized']['error_over_effect']:.8g} |")
    lines+=['','| Common-G case | Actual response error/effect | Propagated response envelope/effect |','|---|---:|---:|']
    for e in out['propagated_common_G_response_envelopes']:
        if e['case_id']=='A':continue
        lines.append(f"| {e['case_id']} | {e['actual_prediction_response_resolution_gate']['error_over_effect']:.8g} | {e['common_response_resolution_gate']['error_over_effect']:.8g} |")
    lines+=['',f"Any actual/predicted-envelope numerical-resolution failure: {out['summary']['any_numerical_resolution_failure']}.",'',
            '## Limits','']+[f'- {v}' for v in out['limits']]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def latexnum(x,digits=5):
    x=float(x)
    if not x:return '0'
    if .001<=abs(x)<10000:return f'{x:.{digits}g}'
    mant,exp=f'{x:.{digits-1}e}'.split('e')
    return mant.rstrip('0').rstrip('.')+r'\times10^{'+str(int(exp))+'}'


def manuscript_values(out):
    s=out['summary'];a=s['actual_prediction'];e=s['propagated_EP']
    resolution=(
        r'The largest absolute frozen-prediction errors among the nine EPs are $'
        +latexnum(a['fc_MHz']['maximum_absolute_error'])+r'$ MHz in $\omega_c/(2\pi)$ and $'
        +latexnum(a['G_kHz']['maximum_absolute_error'])+r'$ kHz in $G_U/(2\pi)$. '
        r'For the six nonreference EP conditions, the largest error/effect ratios are $'
        +latexnum(a['fc_MHz']['maximum_error_over_effect'])+r'$, $'
        +latexnum(a['G_kHz']['maximum_error_over_effect'])+r'$ and $'
        +latexnum(s['actual_EP_response_max_error_over_effect'])
        +r'$ for these two coordinates and the complex response, respectively. '
        r'Separately, we propagate the 16 corners of the independent '
        r'$(\Gamma,\Omega,\widetilde a,\widetilde b)$ numerical envelope, deriving '
        r'$\rho=\widetilde a/\widetilde b$ at each corner. This discards covariance and '
        r'tests a conservative enclosing box, but is not a certified maximum over its continuous interior. '
        r'The largest evaluated corner-deviation/effect ratios are $'
        +latexnum(e['fc_MHz']['maximum_deviation_over_effect'])+r'$, $'
        +latexnum(e['G_kHz']['maximum_deviation_over_effect'])+r'$ and $'
        +latexnum(e['response_normalized']['maximum_deviation_over_effect'])
        +r'$ for the frequency coordinate, coupling coordinate and same-point response. '
        r'At common $G_U$, the largest actual response-error/effect ratio is $'
        +latexnum(s['actual_common_response_max_error_over_effect'])
        +r'$, while the corresponding largest propagated corner ratio is $'
        +latexnum(s['propagated_common_response_max_deviation_over_effect'])+r'$. ')
    if not s['any_numerical_resolution_failure']:
        resolution+=r'All these evaluated ratios remain below the predeclared 0.1 target.'
    else:
        failed=[]
        for key,label in [('fc_MHz','frequency'),('G_kHz','coupling'),('response_normalized','response')]:
            if not e[key]['all_nonreference_pass']:failed.append(label+' envelope')
        for key,label in [('fc_MHz','frequency'),('G_kHz','coupling')]:
            if not a[key]['all_nonreference_pass']:failed.append(label+' prediction')
        if s['actual_EP_response_max_error_over_effect']>=.1:failed.append('EP response prediction')
        if s['actual_common_response_max_error_over_effect']>=.1:failed.append('common-point response prediction')
        if s['propagated_common_response_max_deviation_over_effect']>=.1:failed.append('common-point response envelope')
        resolution+=r'The 0.1 target is not met for '+', '.join(failed)+r'; these failures are retained rather than relaxing the target.'
    r=next(r for r in out['representative_pair_results'] if r['case_id']=='B')
    bounded=(r'$\epsilon_S='+latexnum(r['bounded_scalar_fit']['metrics']['epsilon_S_max'])
             +r'$ for $r=0.126$, compared with $'
             +latexnum(r['strength_scalar_baseline']['epsilon_S_max'])
             +r'$ before allowing the numerical box and $'
             +latexnum(r['full_shape_frozen']['epsilon_S_max'])
             +r'$ for the frozen full-shape prediction; the bounded scalar residual is $'
             +latexnum(r['bounded_fit_residual_over_shape_prediction_error'])
             +r'$ times the full-shape prediction error')
    ds={d['nuisance']:d for d in r['one_nuisance_coupled_diagnostics']}
    seq=[('delta_Omega',r'\Omega'),('delta_Gamma',r'\Gamma'),('delta_wc',r'\omega_c'),('delta_kappa',r'\kappa')]
    clauses=[]
    for key,symbol in seq:
        d=ds[key]
        clauses.append(r'The $'+symbol+r'$-only fit gives $\delta '+symbol+r'/(2\pi)='
                       +latexnum(d['shift']*1000)+r'$ MHz and leaves $\epsilon_S='
                       +latexnum(d['metrics']['epsilon_S_max'])+r'$.')
    dG=ds['delta_G_U'];gain=ds['constant_complex_gain']
    nuisance=(
        r'For the same close-pair response we also perform one-nuisance-at-a-time '
        r'diagnostic fits over declared search intervals of $\pm0.05$ GHz for each '
        r'frequency, decay and coupling rate, and $\pm0.02$ in each component of '
        r'a constant complex gain about unit gain. These intervals are numerical '
        r'search domains, not measured calibration tolerances. '+' '.join(clauses)+r' The coupling-only fit gives '
        r'$\delta G_U/(2\pi)='+latexnum(dG['shift']*1e6)
        +r'$ kHz with $\epsilon_S='+latexnum(dG['metrics']['epsilon_S_max'])
        +r'$. The gain-only fit changes the amplitude by $'
        +latexnum(gain['amplitude_shift'])+r'$ and phase by $'
        +latexnum(gain['phase_shift_deg'])+r'^\circ$, leaving $\epsilon_S='
        +latexnum(gain['metrics']['epsilon_S_max'])
        +r'$. A constant output gain cannot move the EP coordinates.')
    values={'RESOLUTION_RESULTS':resolution,'BOUNDED_RESULT':bounded,'NUISANCE_RESULTS':nuisance}
    (HERE/'manuscript_values.json').write_text(json.dumps(values,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if not s['any_numerical_resolution_failure']:
        main_gate=(r'Both the direct prediction errors and the sampled propagation of the deterministic calibration envelopes remain below one tenth of the ideal residual shape effect in the frequency, coupling and response coordinates for all six nonreference EP tests, and in both nonreference common-$G_U$ response tests. This verifies numerical resolution within the declared model and input calibration, without establishing experimental distinguishability.')
    else:
        main_gate=(r'The independent numerical prediction and the propagation of its calibration envelope are assessed separately against one tenth of the ideal residual shape effect. The sampled numerical-envelope or direct-error tests do not meet this target in all coordinates; the unresolved resolution limits are reported in the Supplemental Material rather than interpreted as experimental distinguishability.')
    (HERE/'main_values.json').write_text(json.dumps({'MAIN_GATE':main_gate},indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
