"""C1: separate numerator strength from shape in the existing Gilbert family.

Reuse the 37 stored full-EP coordinates and the old response window; add the
strength-matched reference and one representative r=.126. No GPU or new bath.
All outputs stay next to this script.
"""
from pathlib import Path
import hashlib
import json
import mpmath as mp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parents[1]/'single_message_ep'/'primary'
OLD=json.loads((PRIOR/'results.json').read_text(encoding='utf-8'))
OLD_CURVES=np.load(PRIOR/'same_operating_point_response.npz')
mp.mp.dps=80
M=mp.mpf
GAM=M(str(OLD['target_Gamma_GHz'])); OMG=M(str(OLD['target_Omega_GHz']))
KAP=M(str(OLD['fixed_kappa_GHz'])); S=OMG**2+GAM**2
RREF=M('.084287616'); REPS=[RREF,M('.126'),M('.99')]
CFG=OLD['same_operating_point_response']
GCOMMON=M(str(CFG['fixed_G_U_GHz'])); WCOMMON=M(str(CFG['fixed_cavity_frequency_GHz']))
EXT=M(str(CFG['eta_external']))
FREQ=OLD_CURVES['frequency_GHz']
Z=-1j*FREQ

def encoded(x):
    if isinstance(x,mp.mpf): return mp.nstr(x,65)
    if isinstance(x,mp.mpc): return {'real':mp.nstr(x.real,65),'imag':mp.nstr(x.imag,65)}
    if isinstance(x,dict): return {k:encoded(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [encoded(v) for v in x]
    if isinstance(x,np.floating): return float(x)
    return x

def material(r):
    C=(r+1/r)/2
    b=mp.sqrt(OMG**2+GAM**2*((1-r*r)/(1+r*r))**2)
    alpha=GAM/(b*C); eta=1/(1+alpha*alpha); wm=b/eta
    a=eta*alpha/r
    return dict(r=r,a=a,b=b,rho=a/b,alpha=alpha,eta=eta,wm=wm)

REF=material(RREF)

def ep(rho,b):
    delta=GAM-KAP
    x=2*delta*rho*OMG**2/(mp.sqrt((1-rho*GAM)**2+(rho*OMG)**2)+1-rho*GAM)
    wc=mp.sqrt(OMG**2+KAP**2+x)
    H=OMG**2*delta**2+GAM*delta*x-x*x/4
    G=mp.sqrt(H/(8*wc*b))
    pole=-(GAM+KAP)/2-mp.j*mp.sqrt(OMG**2-delta**2/4+x/2)
    return dict(G=G,wc=wc,pole=pole)

REF_EP=ep(REF['rho'],REF['b'])

def raw6(m,G,wc):
    """Original physical sublattice coordinates and canonical photon Q/P."""
    r=m['r']; wm=m['wm']; alpha=m['alpha']; eta=m['eta']
    hA=wm*r; hE=wm*(1/r-r)/2
    K=mp.zeros(6)
    for j in range(4):K[j,j]=hA+hE
    K[0,2]=K[2,0]=hE; K[1,3]=K[3,1]=hE
    K[4,4]=K[5,5]=wc
    h=2*G/mp.sqrt(r)
    K[0,4]=K[4,0]=K[2,4]=K[4,2]=-h
    J=mp.zeros(6)
    J[0,1]=-eta;J[1,0]=eta;J[2,3]=eta;J[3,2]=-eta
    J[4,5]=1;J[5,4]=-1
    R=mp.diag([eta*alpha]*4+[0,2*KAP/wc])
    return (J-R)*K,K,R,J

def exact2(m,G,wc):
    """Two second-order coordinates, state (X,dX,Q,dQ); dark pair decoupled."""
    c=2*mp.sqrt(2)*G
    return mp.matrix([[0,1,0,0],[-S,-2*GAM,c*m['b'],c*m['a']],
                      [0,0,0,1],[c*wc,0,-wc*wc,-2*KAP]])

def norm(A): return mp.sqrt(sum(abs(v)**2 for v in A))

def passivity(m,G,wc,high=False):
    A,K,R,J=raw6(m,G,wc)
    kn=np.array(K.tolist(),float);rn=np.array(R.tolist(),float)
    an=np.array(A.tolist(),float)
    kmin=float(np.linalg.eigvalsh(kn)[0]); rmin=float(np.linalg.eigvalsh(rn)[0])
    maxre=float(np.linalg.eigvals(an).real.max())
    out=dict(energy_min_eigenvalue=kmin,dissipation_min_eigenvalue=rmin,
             max_generator_eigenvalue_real_part=maxre,
             dimensionless_energy_ratio=8*G*G/(m['wm']*wc))
    assert kmin>0 and rmin>=0 and maxre<0
    if high:
        residual=norm(A.T*K+K*A+2*K*R*K)/(norm(A)*norm(K))
        out['high_precision_Lyapunov_relative_error']=residual
        assert residual<M('1e-70')
    return out

def response_formula(a,b,G,wc,z=Z):
    z=np.asarray(z); D=z*z+2*float(GAM)*z+float(S)
    P=D*(z*z+2*float(KAP)*z+float(wc*wc))-8*float(wc*G*G)*(float(a)*z+float(b))
    return -2*float(EXT*KAP)*z*D/P

def resolvent_numpy(A,rhs,readout):
    an=np.array(A.tolist(),float)
    bn=np.tile(np.array(rhs,dtype=complex),(len(Z),1))[:,:,None]
    states=np.linalg.solve(Z[:,None,None]*np.eye(len(rhs))-an,bn)[:,:,0]
    return states@np.array(readout,dtype=float)

def ep_errors(ep_pred,ep_full):
    return dict(cavity_frequency_error_MHz=(ep_pred['wc']-ep_full['wc'])*1000,
                target_equivalent_coupling_error_kHz=(ep_pred['G']-ep_full['G'])*10**6)

# Existing full EPs are READ, not searched again. Strength-model predictions
# are added to exactly those stored values; r=.126 is the sole added full point.
grid=[]
for old in OLD['rows']:
    r=M(str(old['r']));m=material(r)
    oldep=dict(G=M(old['G_decimal']),wc=M(old['wc_decimal']))
    scalar=ep(REF['rho'],m['b'])
    physical_gref=scalar['G']*mp.sqrt(m['b']/REF['b'])
    row=dict(r=r,full_ep=oldep,full_ep_source='reused original high-precision JSON',
             strength_ep=scalar,physical_reference_G_at_strength_EP=physical_gref,
             b=m['b'],rho=m['rho'],scalar_errors=ep_errors(scalar,oldep),
             scalar_passivity_at_own_EP=passivity(REF,physical_gref,scalar['wc']),
             scalar_passivity_at_common_target_G=passivity(REF,GCOMMON*mp.sqrt(m['b']/REF['b']),WCOMMON))
    assert abs(physical_gref-REF_EP['G'])<M('1e-70')
    assert abs(scalar['wc']-REF_EP['wc'])<M('1e-70')
    grid.append(row)

arrays={'frequency_GHz':FREQ}
records=[]
for r in REPS:
    m=material(r); rs=str(float(r))
    oldrow=next((row for row in grid if row['r']==r),None)
    full_ep=oldep=oldrow['full_ep'] if oldrow else ep(m['rho'],m['b'])
    scalar_ep=ep(REF['rho'],m['b'])
    exact_ep=ep(m['rho'],m['b'])
    # This formula agreement is explicitly a check of the reused EP record,
    # not another numerical search or an independent empirical estimate.
    exact_ep_error=ep_errors(exact_ep,full_ep)
    assert abs(exact_ep_error['cavity_frequency_error_MHz'])<M('1e-55')
    assert abs(exact_ep_error['target_equivalent_coupling_error_kHz'])<M('1e-55')
    Gref=GCOMMON*mp.sqrt(m['b']/REF['b'])
    A,K,R,J=raw6(m,GCOMMON,WCOMMON)
    Aref,_,_,_=raw6(REF,Gref,WCOMMON)
    A2=exact2(m,GCOMMON,WCOMMON)
    full=resolvent_numpy(A,[0,0,0,0,0,1],[0,0,0,0,0,-2*float(EXT*KAP)])
    scalar_raw=resolvent_numpy(Aref,[0,0,0,0,0,1],[0,0,0,0,0,-2*float(EXT*KAP)])
    reduced=resolvent_numpy(A2,[0,0,0,float(WCOMMON)],[0,0,0,-2*float(EXT*KAP/WCOMMON)])
    scalar_formula=response_formula(REF['rho']*m['b'],m['b'],GCOMMON,WCOMMON)
    full_formula=response_formula(m['a'],m['b'],GCOMMON,WCOMMON)
    peak=float(np.max(abs(full)))
    oldkey=f'S21_r_{float(r)}'
    reuse_error=float(np.max(abs(full-OLD_CURVES[oldkey]))) if oldkey in OLD_CURVES else None
    # Physical free response Mx/h, h=gamma_g Bx/u. Static normalization removes
    # both 2r and b, without passing a pre-normalized canonical measurement.
    freeA=A[:4,:4]
    freeJ=J[:4,:4];freeR=R[:4,:4]
    u=mp.matrix([1,0,1,0]);drive=-(freeJ-freeR)*u
    chi=resolvent_numpy(freeA,list(drive),list(u))
    chi0=(u.T*mp.lu_solve(-freeA,drive))[0]
    chi_hat=chi/float(chi0)
    zz=Z; DD=zz*zz+2*float(GAM)*zz+float(S)
    hat_formula=float(S)*(1+float(m['rho'])*zz)/DD
    hat_ref=float(S)*(1+float(REF['rho'])*zz)/DD
    hat_delta=float(S)*float(m['rho']-REF['rho'])*zz/DD
    phase_difference=np.angle(chi_hat/hat_ref,deg=True)
    arrays[f'chi_hat_r_{rs}']=chi_hat
    arrays[f'chi_hat_delta_r_{rs}']=chi_hat-hat_ref
    arrays[f'phase_difference_deg_r_{rs}']=phase_difference
    arrays[f'S21_full_r_{rs}']=full
    arrays[f'S21_strength_r_{rs}']=scalar_raw
    arrays[f'S21_exact_two_coordinate_r_{rs}']=reduced
    checks=[]
    for index in (0,1600,4000,7200,8000,8800,12000,14400,16000):
        z=-mp.j*M(str(FREQ[index]));D=z*z+2*GAM*z+S
        fullmp=-2*EXT*KAP*mp.lu_solve(z*mp.eye(6)-A,mp.matrix([0,0,0,0,0,1]))[5]
        twomp=-2*EXT*KAP/WCOMMON*mp.lu_solve(z*mp.eye(4)-A2,mp.matrix([0,0,0,WCOMMON]))[3]
        scalarmp=-2*EXT*KAP*mp.lu_solve(z*mp.eye(6)-Aref,mp.matrix([0,0,0,0,0,1]))[5]
        scalarformula=-2*EXT*KAP*z*D/(D*(z*z+2*KAP*z+WCOMMON**2)-8*WCOMMON*GCOMMON**2*m['b']*(1+REF['rho']*z))
        selfenergy_diff=8*WCOMMON*(Gref**2*REF['b']-GCOMMON**2*m['b'])*(1+REF['rho']*z)/D
        checks.append(dict(full_vs_exact2=abs(fullmp-twomp),strength_raw_vs_formula=abs(scalarmp-scalarformula),
                           selfenergy_difference=abs(selfenergy_diff)))
    # Certify exact two-coordinate double root at the full EP; no branch fit.
    ep_pole=exact_ep['pole'];epA2=exact2(m,exact_ep['G'],exact_ep['wc'])
    p2=lambda z:mp.det(z*mp.eye(4)-epA2)
    two_certificate=dict(P_abs_over_Omega4=abs(p2(ep_pole))/OMG**4,
                         dP_abs_over_Omega3=abs(mp.diff(p2,ep_pole))/OMG**3,
                         ddP_abs_over_Omega2=abs(mp.diff(p2,ep_pole,2))/OMG**2)
    full_ep_A,_,_,_=raw6(m,exact_ep['G'],exact_ep['wc'])
    p6=lambda z:mp.det(z*mp.eye(6)-full_ep_A)
    sv=mp.svd((full_ep_A-ep_pole*mp.eye(6))/OMG,compute_uv=False)
    full_certificate=dict(P_abs_over_Omega6=abs(p6(ep_pole))/OMG**6,
                          dP_abs_over_Omega5=abs(mp.diff(p6,ep_pole))/OMG**5,
                          ddP_abs_over_Omega4=abs(mp.diff(p6,ep_pole,2))/OMG**4,
                          smallest_singular_value=sv[5],second_smallest_singular_value=sv[4])
    assert full_certificate['P_abs_over_Omega6']<M('1e-65')
    assert full_certificate['dP_abs_over_Omega5']<M('1e-65')
    assert full_certificate['ddP_abs_over_Omega4']>M('1e-8')
    assert sv[5]<M('1e-65') and sv[4]>M('1e-5')
    rec=dict(r=r,a=m['a'],b=m['b'],rho=m['rho'],
      strength_numerator_a=REF['rho']*m['b'],full_ep=full_ep,
      full_ep_source='reused 37-point JSON' if oldrow else 'one added representative point, analytic quartic',
      strength_ep=scalar_ep,exact_two_coordinate_ep=exact_ep,
      strength_ep_errors=ep_errors(scalar_ep,full_ep),exact_two_coordinate_ep_errors=exact_ep_error,
      common_operating_point_physical_reference_G=Gref,
      common_operating_point_selfenergy_multiplier=m['b']/REF['b'],
      scalar_physical_reference_G_at_own_EP=scalar_ep['G']*mp.sqrt(m['b']/REF['b']),
      full_peak_abs_S21=peak,
      strength_response_max_abs_error=float(np.max(abs(scalar_raw-full))),
      strength_response_relative_peak_error=float(np.max(abs(scalar_raw-full)))/peak,
      exact2_response_max_abs_error=float(np.max(abs(reduced-full))),
      full_raw_vs_analytic_response_error=float(np.max(abs(full-full_formula))),
      strength_raw_vs_analytic_response_error=float(np.max(abs(scalar_raw-scalar_formula))),
      prior_saved_full_response_max_abs_difference=reuse_error,
      normalized_free_response_max_abs_difference_from_ref=float(np.max(abs(chi_hat-hat_ref))),
      normalized_free_difference_formula_error=float(np.max(abs(chi_hat-hat_ref-hat_delta))),
      normalized_free_response_formula_error=float(np.max(abs(chi_hat-hat_formula))),
      normalized_free_phase_difference_at_Omega_deg=float(phase_difference[8000]),
      high_precision_comparisons={key:max(q[key] for q in checks) for key in checks[0]},
      exact2_EP_certificate=two_certificate,
      full6D_EP_certificate=full_certificate,
      full_passivity_at_own_EP=passivity(m,exact_ep['G'],exact_ep['wc'],True),
      full_passivity_at_common_point=passivity(m,GCOMMON,WCOMMON,True),
      strength_passivity_at_common_point=passivity(REF,Gref,WCOMMON,True),
      strength_passivity_at_own_EP=passivity(REF,scalar_ep['G']*mp.sqrt(m['b']/REF['b']),scalar_ep['wc'],True))
    assert rec['exact2_response_max_abs_error']<1e-11
    assert rec['strength_raw_vs_analytic_response_error']<1e-11
    assert max(rec['high_precision_comparisons'].values())<M('1e-65')
    assert two_certificate['P_abs_over_Omega4']<M('1e-65')
    assert two_certificate['dP_abs_over_Omega3']<M('1e-65')
    if r!=RREF: assert rec['strength_response_max_abs_error']>1e-7
    records.append(rec)

# Record the tiny difference between the chosen 9-digit reference and the
# reference implied by the original supplied MnF2-labeled field tuple.
be=M('54.96244102333783');ba=M('0.7865384615384615')
r_fields=mp.sqrt(ba/(2*be+ba)); mf=material(r_fields); ef=ep(mf['rho'],mf['b'])
rounding=dict(chosen_r_reference=RREF,prior_37point_r_reference=M(str(OLD['reference_r'])),
    chosen_minus_prior_37point_reference=RREF-M(str(OLD['reference_r'])),
    reference_from_original_declared_fields=r_fields,
    chosen_minus_original_field_reference=RREF-r_fields,
    matched_family_EP_shift_from_original_field_r_Hz=dict(
        wc=(REF_EP['wc']-ef['wc'])*10**9,G=(REF_EP['G']-ef['G'])*10**9),
    note='This compares r rounding along the fixed-pole family; it is not a new fit or a change to the original physical input tuple.')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors=['#235789','#D28523','#248761']
def savefig(fig,name):
    fig.savefig(HERE/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(HERE/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)

offset=FREQ-float(OMG)
fig,axs=plt.subplots(1,3,figsize=(13,3.3),constrained_layout=True)
for r,col in zip(REPS,colors):
    rs=str(float(r));h=arrays[f'chi_hat_r_{rs}'];ph=arrays[f'phase_difference_deg_r_{rs}']
    for ax,y in zip(axs,[h.real,h.imag,ph]):ax.plot(offset,y,color=col,label=f'$r={float(r):g}$')
axs[0].set(ylabel=r'Re $[\chi_r(\omega)/\chi_r(0)]$',title='(a) Static strength removed')
axs[1].set(ylabel=r'Im $[\chi_r(\omega)/\chi_r(0)]$',title='(b) Identical free poles')
axs[2].set(ylabel='Phase difference from reference (deg)',title='(c) Shape difference remains')
for ax in axs:ax.set_xlabel(r'$(\omega-\Omega_0)/(2\pi)$ (GHz)');ax.grid(alpha=.15)
axs[0].legend(frameon=False,fontsize=9)
savefig(fig,'c1_normalized_free_response')

fig,axs=plt.subplots(1,2,figsize=(10.8,3.6),constrained_layout=True)
rg=np.array([float(v['r']) for v in grid]);gf=np.array([float(v['full_ep']['G']) for v in grid]);wf=np.array([float(v['full_ep']['wc']) for v in grid])
gs=np.array([float(v['strength_ep']['G']) for v in grid]);ws=np.array([float(v['strength_ep']['wc']) for v in grid])
axs[0].plot(rg,(wf-float(REF_EP['wc']))*1000,'o-',ms=3,color=colors[0],label='Full energy model (37 points)')
axs[0].plot(rg,(ws-float(REF_EP['wc']))*1000,'--',color=colors[1],label='Strength-matched scalar')
axs[1].plot(rg,(gf-float(REF_EP['G']))*1e6,'o-',ms=3,color=colors[0])
axs[1].plot(rg,(gs-float(REF_EP['G']))*1e6,'--',color=colors[1])
for rec in records:
    axs[0].plot(float(rec['r']),float(rec['exact_two_coordinate_ep']['wc']-REF_EP['wc'])*1000,'x',color=colors[2],ms=7)
    axs[1].plot(float(rec['r']),float(rec['exact_two_coordinate_ep']['G']-REF_EP['G'])*1e6,'x',color=colors[2],ms=7)
axs[0].plot([],[],'x',color=colors[2],label='Exact two-coordinate, 3 representatives')
axs[0].set(ylabel=r'$\Delta f_{c,\mathrm{EP}}$ from reference (MHz)',title='(a) Strength cannot fix the frequency coordinate')
axs[1].set(ylabel=r'$\Delta G_{U,\mathrm{EP}}/(2\pi)$ (kHz)',title='(b) Target-equivalent coupling coordinate')
for ax in axs:ax.set_xlabel(r'Squeezing ratio $r$');ax.grid(alpha=.15)
axs[0].legend(frameon=False,fontsize=8,loc='lower left')
savefig(fig,'c1_strength_matched_ep')

fig,axs=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
for r,col in zip(REPS,colors):
    rs=str(float(r));full=arrays[f'S21_full_r_{rs}'];scalar=arrays[f'S21_strength_r_{rs}'];reduced=arrays[f'S21_exact_two_coordinate_r_{rs}']
    for ax,func in zip(axs[:2],[np.real,np.imag]):
        ax.plot(offset,func(full),color=col,lw=1.3,label=f'$r={float(r):g}$ full' if ax is axs[0] else '_nolegend_')
        ax.plot(offset,func(scalar),'--',color=col,lw=.85)
        idx=np.arange(0,len(offset),1600)
        ax.plot(offset[idx],func(reduced[idx]),'x',color=col,ms=3)
    if r!=RREF:
        axs[2].semilogy(offset,np.maximum(abs(scalar-full),1e-18),color=col,label=f'$r={float(r):g}$ scalar')
        axs[2].semilogy(offset[::160],np.maximum(abs(reduced-full)[::160],1e-18),'.',color=col,ms=2)
axs[0].set(ylabel=r'Re $S_{21}$',title='(a) Common absolute port normalization')
axs[1].set(ylabel=r'Im $S_{21}$',title='(b) Same actual target operating point')
axs[2].set(ylabel=r'$|S_{21}^{\mathrm{prediction}}-S_{21}^{\mathrm{full}}|$',title='(c) Scalar residual; exact closure at roundoff',ylim=(1e-16,3e-3))
for ax in axs:ax.set_xlabel(r'$(\omega-\Omega_0)/(2\pi)$ (GHz)');ax.grid(alpha=.15)
axs[0].legend(frameon=False,fontsize=8,loc='lower left')
axs[1].plot([],[],'--',color='.3',label='Scalar');axs[1].plot([],[],'x',color='.3',label='Exact two-coordinate');axs[1].legend(frameon=False,fontsize=8)
axs[2].plot([],[],'.',color='.3',label='Exact two-coordinate');axs[2].legend(frameon=False,fontsize=8,loc='center right')
savefig(fig,'c1_absolute_cavity_response')

np.savez_compressed(HERE/'c1_curves.npz',**arrays)
result=dict(scope='C1 only: separate numerator scale b from shape rho in the specified native-Gilbert zero-bias uniform-Zeeman family. No new GPU, bath or branch campaign.',
    units='All internal rates divided by u=2*pi*1e9 s^-1; reported frequency errors in MHz and coupling errors in kHz are separate.',
    inputs=dict(Gamma=GAM,Omega=OMG,kappa=KAP,r_reference=RREF,representatives=REPS,
                common_G_U=GCOMMON,common_wc=WCOMMON,eta_external=EXT,frequency_count=len(FREQ)),
    reference_rounding=rounding,
    definitions=dict(full_numerator='b_r(1+rho_r lambda)',
      strength_numerator='b_r(1+rho_ref lambda)',
      physical_strength_realization='reference magnet at G_ref=G_U*sqrt(b_r/b_ref)',
      exact_two_coordinate='Actual numerator; derivative coupling retained. Reproduces bright quartic and driven transfer, dark pair remains decoupled.',
      free_normalization='chi_r(lambda)/chi_r(0); static strength removed only in the free response figure',
      cavity_normalization='Same physical ports and absolute S21; no separate normalization of displayed cavity curves',
      coupling_axis='Scalar EP G_U is target-equivalent; the real reference device uses separately reported G_ref.',
      theorem_limits='Gamma>0, fixed Gamma/Omega/kappa, admissible nonzero real-coupling complex EP branch. Excludes Gamma=kappa,G=0 ordinary coincidence and arbitrary complex effective couplings.'),
    reused_37point_grid=grid,representatives=records,
    completion=dict(C1a=True,C1b=True,C1c=True,criterion_satisfied=True,
       statement='Full energy and exact two-coordinate predictions agree; a passive strength-matched reference retains nonzero shape residuals in both EP coordinates and common-point response for nonreference r.'),
    provenance=dict(input_files=[dict(path=p.relative_to(HERE.parents[1]).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for p in [PRIOR/'results.json',PRIOR/'same_operating_point_response.npz',PRIOR/'inputs'/'energy_model.py']],
         script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         prior_physics_modules_imported=False,new_full_EP_points=1,reused_full_EP_points=37,
         figures=[p.name for p in sorted(HERE.glob('c1_*.pdf'))]))
(HERE/'c1_results.json').write_text(json.dumps(encoded(result),indent=2)+'\n',encoding='utf-8')
print(json.dumps(encoded(dict(reference_rounding=rounding,representatives=records)),indent=2))
