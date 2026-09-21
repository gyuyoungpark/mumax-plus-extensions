"""Independent B=0 AFM/cavity EP2 audit using a real-quartic perfect square.

No primary EP solver or frozen energy-model module is imported. Rates and
eigenvalues are angular quantities divided by 2*pi*GHz; physical gamma is SI.
Only files adjacent to this script are written.
"""
from pathlib import Path
import json
import hashlib
import mpmath as mp
import numpy as np

mp.mp.dps = 80
HERE = Path(__file__).resolve().parent
GAMMA = mp.mpf('1.561154518256')
OMEGA = mp.mpf('261.310962155227')
KAPPA = mp.mpf('261.365756187') / 1000
R0 = mp.mpf('.084287616')
GRID = [mp.mpf(x) for x in ('.014', '.084287616', '.2', '.5', '.99')]
GYRO = mp.mpf('1.7595e11')/(2*mp.pi*10**9)


def as_json(obj):
    if isinstance(obj, mp.mpc):
        return {'real': mp.nstr(obj.real, 45), 'imag': mp.nstr(obj.imag, 45)}
    if isinstance(obj, mp.mpf):
        return mp.nstr(obj, 45)
    if isinstance(obj, dict):
        return {k: as_json(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [as_json(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def submatrix(A, row, col):
    return mp.matrix([[A[i,j] for j in range(A.cols) if j != col]
                      for i in range(A.rows) if i != row])


def maxentry(A):
    return max(abs(x) for x in A)


def determinant(A):
    # mpmath's LU determinant can fail on structurally singular minors with
    # a None pivot. Explicit partial pivoting handles those exact zeros.
    B=A.copy()
    value=mp.mpf(1)
    for j in range(B.rows):
        pivot=max(range(j,B.rows),key=lambda i:abs(B[i,j]))
        if B[pivot,j] == 0:
            return mp.mpf(0)
        if pivot != j:
            for k in range(B.cols):
                B[j,k],B[pivot,k]=B[pivot,k],B[j,k]
            value=-value
        value*=B[j,j]
        for i in range(j+1,B.rows):
            factor=B[i,j]/B[j,j]
            for k in range(j+1,B.cols):
                B[i,k]-=factor*B[j,k]
            B[i,j]=0
    return value


def controls(r, gamma, omega):
    C=(r+1/r)/2
    T=(1/r-r)/2
    # b is the constant coefficient of the Gilbert susceptibility numerator.
    b=mp.sqrt(omega**2+gamma**2*(T/C)**2)
    alpha=gamma/(b*C)
    eta=1/(1+alpha**2)
    wm=b/eta
    a=eta*alpha/r
    h=wm/GYRO
    return dict(r=r, C=C, T=T, alpha=alpha, eta=eta, wm=wm,
                BE=h*T, BA=h*r, a=a, b=b,
                free_Gamma=eta*alpha*wm*C,
                free_Omega=eta*wm*mp.sqrt(1-alpha**2*T**2))


def generator(c, wc, coupling, kappa):
    # Canonical coordinates X,Y,U,V,Q,P. This constructs the full real
    # generator independently from its Hessian, symplectic and damping parts.
    K=mp.zeros(6)
    for i in range(4):
        K[i,i]=c['wm']
    K[4,4]=K[5,5]=wc
    K[0,4]=K[4,0]=-2*mp.sqrt(2)*coupling
    J=mp.zeros(6)
    J[0,3]=-c['eta']; J[3,0]=c['eta']
    J[1,2]=c['eta']; J[2,1]=-c['eta']
    J[4,5]=1; J[5,4]=-1
    R=mp.zeros(6)
    R[0,0]=R[1,1]=c['eta']*c['alpha']/c['r']
    R[2,2]=R[3,3]=c['eta']*c['alpha']*c['r']
    R[5,5]=2*kappa/wc
    return K,J,R,(J-R)*K


def solve_one(r, gamma, omega, kappa, certify=True):
    c=controls(r,gamma,omega)
    S=gamma**2+omega**2
    delta=gamma-kappa
    rho=c['a']/c['b']
    A1=1-rho*gamma
    # Physical-sign root, rationalized to avoid cancellation at small gamma.
    x=2*delta*rho*omega**2/(mp.sqrt(A1*A1+(rho*omega)**2)+A1)
    wc2=omega**2+kappa**2+x
    if wc2 <= 0:
        return dict(status='no_positive_cavity_on_this_candidate',controls=c)
    wc=mp.sqrt(wc2)
    coupling2=(omega**2*delta**2+gamma*delta*x-x*x/4)/(8*wc*c['b'])
    nu2=omega**2-delta**2/4+x/2
    if coupling2 < 0 or nu2 <= 0:
        return dict(status='candidate_outside_complex_EP_domain', controls=c,
                    coupling2=coupling2,nu2=nu2,wc=wc)
    G=mp.sqrt(coupling2)
    d=(gamma+kappa)/2
    e=omega**2+gamma*kappa+x/2
    pole=-d-1j*mp.sqrt(nu2)
    K,J,R,M=generator(c,wc,G,kappa)
    bright=[0,3,4,5]
    Mbright=mp.matrix([[M[i,j] for j in bright] for i in bright])
    coupling_c=2*mp.sqrt(2)*G
    # Invertible conversion [X,V,Q,P] -> [X,Q,dX/dt,dQ/dt].
    change=mp.matrix([[1,0,0,0],[0,0,1,0],
                     [-c['a']*c['wm'],-c['b'],c['a']*coupling_c,0],
                     [0,0,0,wc]])
    two_coordinate=mp.matrix([[0,0,1,0],[0,0,0,1],
        [-S,coupling_c*c['b'],-2*gamma,coupling_c*c['a']],
        [wc*coupling_c,-wc*wc,0,-2*kappa]])
    out=dict(controls=c,Gamma=gamma,Omega=omega,kappa=kappa,wc=wc,
             loaded_Q=(wc/(2*kappa) if kappa else None),G=G,
             lambda_EP=pole,x=x,e=e,nu2=nu2,
             fixed_pole_error=max(abs(c['free_Gamma']-gamma),
                                  abs(c['free_Omega']-omega)),
             numerator_ratio=rho,
             ratio_identity_error=abs(rho-2*gamma/((1+r*r)*S)),
             exact_two_coordinate_similarity_error=maxentry(change*Mbright-two_coordinate*change),
             exact_two_coordinate_change_determinant=determinant(change))
    if delta == 0:
        # Three uncoupled modes share the conjugate pole pair: not an EP.
        out.update(status='excluded_semisimple_zero_coupling',
                   geometric_multiplicity=3,algebraic_multiplicity=3,
                   diagonalizable_minimal_polynomial_residual=maxentry(M*M+2*gamma*M+S*mp.eye(6)))
        return out
    out['status']='certified_complex_EP2'
    P=[mp.mpf(1), 2*(gamma+kappa),
       S+wc*wc+4*gamma*kappa,
       2*gamma*wc*wc+2*kappa*S-8*wc*G*G*c['a'],
       S*wc*wc-8*wc*G*G*c['b']]
    square=[mp.mpf(1),4*d,2*e+4*d*d,4*d*e,e*e]
    scale=max(omega,wc,mp.mpf(1))
    p=lambda lam: mp.polyval(P,lam)
    dp=lambda lam: mp.polyval([4*P[0],3*P[1],2*P[2],P[3]],lam)
    ddp=lambda lam: mp.polyval([12*P[0],6*P[1],2*P[2]],lam)
    D=lambda lam: lam*lam+2*gamma*lam+S
    dark_poles=[-gamma-1j*omega,-gamma+1j*omega]
    out.update(
        normalized_square_coefficient_error=max(abs(P[i]-square[i])/scale**i for i in range(5)),
        normalized_P=abs(p(pole))/scale**4,
        normalized_dP=abs(dp(pole))/scale**3,
        normalized_ddP=abs(ddp(pole))/scale**2,
        ddP_identity_error=abs(ddp(pole)+8*nu2),
        dark_denominator_at_EP=D(pole),
        min_dark_pole_distance=min(abs(pole-z) for z in dark_poles),
        bright_algebraic_multiplicity=2,
        all_full_poles=[pole,pole,mp.conj(pole),mp.conj(pole)]+dark_poles,
        max_full_pole_real=max(-d,-gamma),
        energy_bound_ratio=8*G*G/(c['wm']*wc),
        K_min_eigen=(c['wm']+wc-mp.sqrt((c['wm']-wc)**2+32*G*G))/2,
        R_min_diagonal=min(R[i,i] for i in range(6)),
        passivity_identity_error=maxentry(M.T*K+K*M+2*K*R*K))
    # Characteristic factorization at independent complex probes, including
    # both real and imaginary components, without relying on eigenvalue fits.
    probes=[mp.mpc('.37','-.91')*scale,
            mp.mpc('-1.3','-1.01')*scale,
            pole+mp.mpc('.023','.014')]
    fact=[]
    for z in probes:
        direct=determinant(z*mp.eye(6)-M)
        target=D(z)*p(z)
        fact.append(abs(direct-target)/max(abs(target),mp.mpf(1)))
    out['max_independent_matrix_factorization_error']=max(fact)
    if certify:
        B=M-pole*mp.eye(6)
        minors=[(abs(determinant(submatrix(B,i,j))),i,j)
                for i in range(6) for j in range(6)]
        minor_size,drop_i,drop_j=max(minors,key=lambda v:v[0])
        v=mp.matrix([(-1)**(drop_i+j)*determinant(submatrix(B,drop_i,j))
                     for j in range(6)])
        v/=maxentry(v)
        rhs=mp.matrix([v[i] for i in range(6) if i!=drop_i])
        reduced=mp.lu_solve(submatrix(B,drop_i,drop_j),rhs)
        w=mp.zeros(6,1)
        for j,value in zip([j for j in range(6) if j!=drop_j],reduced):
            w[j]=value
        sv=np.linalg.svd(np.array(B.tolist(),dtype=complex),compute_uv=False)
        out.update(geometric_multiplicity=1,
                   normalized_nonzero_5x5_minor=minor_size/scale**5,
                   selected_minor_removed_row_col=[drop_i,drop_j],
                   eigenvector_residual=maxentry(B*v)/scale,
                   generalized_eigenvector_residual=maxentry(B*w-v),
                   normalized_direct_6x6_determinant=abs(determinant(B))/scale**6,
                   double_precision_smallest_singular_values=sv[-2:].tolist())
    # The other algebraic x root necessarily gives G^2 < 0 for gamma>0,
    # when it has a real positive cavity frequency.
    if rho > 0:
        other=2*delta/rho*(-A1-mp.sqrt(A1*A1+(rho*omega)**2))
        out['alternate_x']=other
        out['alternate_has_positive_wc2']=(omega**2+kappa**2+other)>0
        out['alternate_delta_x']=delta*other
        if out['alternate_has_positive_wc2']:
            alternate_wc=mp.sqrt(omega**2+kappa**2+other)
            out['alternate_implied_G_squared']=delta*other/(8*alternate_wc*c['a'])
    return out


def summarize(rows):
    yes=[z for z in rows if z['status']=='certified_complex_EP2']
    if not yes:
        return {'certified_EP_count':0}
    g=[z['G'] for z in yes]; w=[z['wc'] for z in yes]
    return dict(certified_EP_count=len(yes),G_min=min(g),G_max=max(g),
                G_span_Hz=(max(g)-min(g))*10**9,
                G_relative_span_percent=100*(max(g)/min(g)-1),
                wc_min=min(w),wc_max=max(w),wc_span_Hz=(max(w)-min(w))*10**9)


primary=[solve_one(r,GAMMA,OMEGA,KAPPA) for r in GRID]
for row in primary:
    assert row['status']=='certified_complex_EP2'
    assert row['energy_bound_ratio'] < 1 and row['K_min_eigen'] > 0
    assert row['normalized_nonzero_5x5_minor'] > mp.mpf('1e-8')
    assert row['normalized_ddP'] > 1 and row['min_dark_pole_distance'] > 0
    assert row['max_full_pole_real'] < 0
    assert row['eigenvector_residual'] < mp.mpf('1e-65')
    assert row['generalized_eigenvector_residual'] < mp.mpf('1e-65')
regular=[]
for ss in ('1','.3','.1','.03','.01','.003','.001','0'):
    scale=mp.mpf(ss)
    rows=[solve_one(r,scale*GAMMA,OMEGA,KAPPA,certify=False) for r in GRID]
    regular.append(dict(Gamma_scale=scale,protocol='fixed Omega and kappa, Gamma scaled; alpha is solved exactly',
                        rows=rows,summary=summarize(rows)))
cavity=[]
for kk in (mp.mpf(0),KAPPA/4,KAPPA,2*KAPPA,4*KAPPA,GAMMA,8*KAPPA):
    rows=[solve_one(r,GAMMA,OMEGA,kk,certify=False) for r in GRID]
    cavity.append(dict(kappa=kk,rows=rows,summary=summarize(rows)))
joint=[]
for ss in ('1','.1','.01','0'):
    scale=mp.mpf(ss)
    rows=[solve_one(r,scale*GAMMA,OMEGA,scale*KAPPA,certify=False) for r in GRID]
    joint.append(dict(loss_scale=scale,rows=rows,summary=summarize(rows)))
matched=solve_one(R0,KAPPA,OMEGA,KAPPA)
all_rows=primary+[row for family in (regular,cavity,joint)
                  for item in family for row in item['rows']]+[matched]
for row in all_rows:
    assert row['fixed_pole_error'] < mp.mpf('1e-65')
    assert row['exact_two_coordinate_similarity_error'] < mp.mpf('1e-65')
    if row['status']=='excluded_semisimple_zero_coupling':
        assert row['G']==0
        assert row['diagonalizable_minimal_polynomial_residual'] < mp.mpf('1e-65')
        continue
    assert row['status']=='certified_complex_EP2'
    assert row['G'] > 0 and row['nu2'] > 0
    assert row['K_min_eigen'] > 0 and row['R_min_diagonal'] >= 0
    assert row['min_dark_pole_distance'] > 0
    assert row['normalized_P'] < mp.mpf('1e-65')
    assert row['normalized_dP'] < mp.mpf('1e-65')
    assert row['max_independent_matrix_factorization_error'] < mp.mpf('1e-65')
    assert row['passivity_identity_error'] < mp.mpf('1e-65')
result=dict(
    scope='Independent B=0 exact bright-quartic calculation; no active manuscript changes, no GPU or device validation.',
    formulation='80-digit perfect-square coefficient matching, independently constructed canonical 6D matrix, minor/rank/Jordan certificates',
    units='All displayed rates/poles are angular values divided by 2*pi GHz. G and wc differences multiplied by 1e9 are Hz of the corresponding /2pi quantity.',
    primary_targets=dict(Gamma=GAMMA,Omega=OMEGA,kappa=KAPPA,gamma_SI=mp.mpf('1.7595e11'),r0=R0,grid=GRID),
    primary_rows=primary,primary_summary=summarize(primary),
    regular_spin_loss_controls=regular,cavity_loss_controls=cavity,
    joint_loss_controls=joint,loss_matched_exclusion=matched,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'zero_field_ep_results.json').write_text(json.dumps(as_json(result),indent=2)+'\n',encoding='utf-8')

def n(x,d=12):
    return mp.nstr(x,d)

lines=['# Independent zero-field EP2 audit','',
'This bounded calculation is independent of the primary root solver: it matches a monic real quartic to the square of a quadratic, then checks the separately constructed full canonical matrix. No primary model module is imported. All rates below mean angular quantities divided by 2π GHz.','',
'## Controls and derivation','',
'The free magnetic polynomial is D(λ)=λ²+2Γλ+S, S=Ω²+Γ². Γ=1.561154518256, Ω=261.310962155227, and κ=0.261365756187 are fixed in the primary test. The bias is exactly zero. The cavity frequency ωc and physical per-branch GU are the two EP coordinates; Q=ωc/(2κ) is consequently not fixed. The native γ=1.7595e11 rad/s/T is fixed.','',
'For C=(r+1/r)/2 and T=(1/r−r)/2, define b=sqrt[Ω²+Γ²(T/C)²], α=Γ/(bC), η=1/(1+α²), and ωm=b/η. The fields are BA=(ωm/γ)r and BE=(ωm/γ)T, with the appropriate angular/GHz conversion. These give exactly the same two conjugate free poles, each doubly degenerate, for every r. The susceptibility numerator is N=aλ+b, a=ηα/r.','',
'Write δ=Γ−κ, ρ=a/b=2Γ/[(1+r²)S], x=ωc²−Ω²−κ² and e=Ω²+Γκ+x/2. A complex EP and its conjugate require the real quartic P=D(λ)(λ²+2κλ+ωc²)−8ωc GU²N to equal [λ²+(Γ+κ)λ+e]². Coefficient matching gives','',
'    (ρ/4)x² + δ(1−ρΓ)x − ρΩ²δ² = 0',
'    x = 2δρΩ² / [sqrt((1−ρΓ)²+ρ²Ω²) + (1−ρΓ)]',
'    GU² = (Ω²δ²+Γδx−x²/4)/(8ωc b)',
'    λEP = −(Γ+κ)/2 − i sqrt[Ω²−δ²/4+x/2].','',
'For Γ>0 and δ≠0, the chosen root has δx>0 and coefficient matching gives 8ωc GU²a=δx, hence positive GU². The other root has δx<0, so it cannot give positive GU² on a real positive-cavity branch. At Γ=0 and κ>0, coefficient matching instead gives x=0 and the finite EP below. This classifies the complex-conjugate EP pair of this bright quartic, not arbitrary overdamped real-root degeneracies or finite-bias systems.','',
'## Primary results','',
'| r | α | ωm/(2π) GHz | ωc/(2π) GHz | GU/(2π) GHz | Q | Im λEP/(2π) GHz |',
'|---:|---:|---:|---:|---:|---:|---:|']
for row in primary:
    lines.append('| '+' | '.join(n(v,15) for v in (row['controls']['r'],row['controls']['alpha'],row['controls']['wm'],row['wc'],row['G'],row['loaded_Q'],row['lambda_EP'].imag))+' |')
su=result['primary_summary']
lines += ['',f"All five rows have Re λEP/(2π)={n(primary[0]['lambda_EP'].real,16)} GHz. The sampled coupling span is {n(su['G_span_Hz'])} Hz ({n(su['G_relative_span_percent'])}%), while the cavity-frequency coordinate spans {n(su['wc_span_Hz'])} Hz. These are different observables; the multi-MHz cavity displacement must not be called a multi-MHz coupling change.",'',
'## Independent certificates','',
f"Maximum normalized P and P′ residuals: {n(max(z['normalized_P'] for z in primary))}, {n(max(z['normalized_dP'] for z in primary))}. Maximum perfect-square coefficient discrepancy: {n(max(z['normalized_square_coefficient_error'] for z in primary))}.",
f"Maximum independent 6D matrix/characteristic factorization discrepancy at complex test points: {n(max(z['max_independent_matrix_factorization_error'] for z in primary))}.",
f"Maximum right-eigenvector and generalized-eigenvector residuals: {n(max(z['eigenvector_residual'] for z in primary))}, {n(max(z['generalized_eigenvector_residual'] for z in primary))}.",
f"The normalized nonzero 5x5 minors are at least {n(min(z['normalized_nonzero_5x5_minor'] for z in primary))}; the determinant vanishes numerically at 80-digit precision. Thus each repeated complex pole has geometric multiplicity one. P″=−8ν²≠0; its minimum normalized magnitude is {n(min(z['normalized_ddP'] for z in primary))}. The nearest dark pole remains at least {n(min(z['min_dark_pole_distance'] for z in primary))} GHz away.",
f"Every primary energy Hessian is positive: its minimum eigenvalue is at least {n(min(z['K_min_eigen'] for z in primary))}, and max[8GU²/(ωmωc)]={n(max(z['energy_bound_ratio'] for z in primary))}<1. The damping metric is nonnegative and the passivity identity AᵀK+KA=−2KRK has maximum residual {n(max(z['passivity_identity_error'] for z in primary))}. All full-system poles decay.",'',
'## Regular spin-loss and cavity-loss controls','',
'The regular spin-loss control scales Γ while keeping Ω and κ fixed, solving α and ωm exactly at each r. It is not an assertion that the microscopic α scales by exactly the same factor. This path crosses an ordinary zero-coupling coincidence when Γ=κ; that one point is excluded rather than called an EP.','',
'| Γ/Γref | sampled GU span (Hz) | sampled relative span (%) | sampled ωc span (Hz) |',
'|---:|---:|---:|---:|']
for z in regular:
    su=z['summary']
    lines.append('| '+' | '.join(n(v) for v in (z['Gamma_scale'],su['G_span_Hz'],su['G_relative_span_percent'],su['wc_span_Hz']))+' |')
lines += ['',
'At Γ=0 and κ>0 the canonical matrices and EP coordinates are r independent, GU²=Ωκ²/[8 sqrt(Ω²+κ²)], ωc=sqrt(Ω²+κ²). The two dark poles are then undamped, while the bright EP remains decaying; passivity does not imply strict decay of every pole in this limit.','',
'The cavity-loss controls cover κ=0, κref/4, κref, 2κref, 4κref, Γref, and 8κref. Finite complex EP2s are found for all five r values except the exact Γ=κ point. There GU=0 and the full uncoupled system has geometric multiplicity three at each conjugate eigenvalue, so it is an ordinary semisimple coincidence. Joint spin/cavity loss reduction gives r-independent zero-coupling conservative coincidence at the all-zero-loss endpoint, also not an EP. Detailed parameters and certificates are in JSON.','',
'The joint small-loss controls also show the expected orders: scaling Γ and κ together by t gives an r-dependent cavity-coordinate span of order t² and an r-dependent coupling-coordinate span of order t³. At t=0.1 the spans are 38,421.3192 Hz and 5.4660710 Hz, respectively; at t=0.01 they are 384.213245 Hz and 0.005466073 Hz. These agree with x≈2Γ(Γ−κ)/(1+r²), ωc≈Ω+[κ²+x]/(2Ω), and GU≈|Γ−κ|/(2√2) at small loss.','',
'## What this establishes, and the scalar-model boundary','',
'Unlike the prior finite-bias EP sweeps, this test fixes the entire free spin spectrum and the cavity amplitude loss. No finite-bias branch-frequency or branch-linewidth change explains the EP-coordinate motion. The remaining mechanism is the numerator N: its slope-to-constant ratio depends on r at finite Γ. This is not a loss-independent effect and does not imply independent experimental tunability of exchange, anisotropy and damping.','',
'No frequency-independent scalar rescaling of a fixed reference numerator can equate both GU²a and GU²b at two distinct r values when Γ>0. Matching the constant term fixes the scale but leaves the λ term different, because a/b=2Γ/[(1+r²)(Ω²+Γ²)]. This is a rigorous limit on a pole-calibrated scalar susceptibility with a fixed numerator and one coupling scale. It is not a prohibition on a general two-oscillator realization: at B=0 the dark magnetic sector factors out exactly, and the bright two-oscillator problem retains derivative/source-residue terms. A frequency-dependent effective model or a more general damping/input operator can encode this residue.','',
'An explicit exact real two-coordinate realization is available, not merely an approximate frequency-dependent fit. With c=2√2 GU, the bright sector obeys','',
'    Ẍ + 2Γ Ẋ + S X = c(a Q̇ + b Q)',
'    Q̈ + 2κ Q̇ + ωc² Q = ωc c X.','',
'The inverse state map is V=(−Ẋ−aωm X+acQ)/b and P=Q̇/ωc. It is nonsingular because bωc>0. Thus the bright sector has exactly two second-order coordinates (four real first-order states). The derivative cross term carries the finite-damping numerator. Calling this intrinsically beyond all two-mode models would be incorrect; an ordinary positive-frequency 2×2 scalar-linewidth approximation is a narrower model class. For GU>0 the 2×2 polynomial matrix has a nonzero off-diagonal entry −ωc c, so its kernel at any root has dimension one. Together with P″≠0 and separation from D, this also establishes EP2 defectiveness analytically. At Γ=κ and GU=0 the full matrix instead satisfies D(A)=0 with distinct roots, proving semisimplicity.','',
f"The independently checked exact two-coordinate similarity has maximum entrywise residual {n(max(z['exact_two_coordinate_similarity_error'] for z in primary))}; its transformation determinant never vanishes. The matched-loss exclusion has max|D(A)|={n(matched['diagonalizable_minimal_polynomial_residual'])}.",'',
'A measured discriminator, robustness to calibration uncertainty, experimental parameter provenance and literature novelty have not been established by this calculation. The coupling-coordinate shift is small even though the cavity-frequency coordinate moves by MHz. No finite-bias, auxiliary-mediated, GPU or device-feasibility claim is added.','',
'## Reproduce','',
'Run `python -B -X utf8 audit_zero_field_ep.py` with mpmath and NumPy. The script writes only this report and the adjacent JSON; it imports no project physics code.']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(as_json({'primary_summary':result['primary_summary'],
                         'primary_G': [x['G'] for x in primary],
                         'max_Jordan_residual':max(x['generalized_eigenvector_residual'] for x in primary)}),indent=2))
