"""Matched entire free-spin spectrum: zero-field passive cavity EP test.

Rates below are angular frequencies divided by 2*pi GHz; lambda uses
the corresponding inverse-time unit. Full energy and viscous photon bath
follow the included energy model. This does not test arbitrary baths.
"""
from pathlib import Path
import sys,json,hashlib,shutil
sys.dont_write_bytecode=True
import numpy as np
import mpmath as mp
from dataclasses import replace
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy,platform

HERE=Path(__file__).resolve().parent
INPUT=HERE/'inputs'; INPUT.mkdir(exist_ok=True)
FROZEN=INPUT/'energy_model.py'
EXPECTED_SOURCE_SHA='a69d062d8cc6d2fb0d5f2f165ebcea0166f53ffbac2aebb6819573f7f4b7b708'
if hashlib.sha256(FROZEN.read_bytes()).hexdigest()!=EXPECTED_SOURCE_SHA:
    raise RuntimeError('Frozen audited input hash mismatch; do not silently replace it.')
sys.path.insert(0,str(INPUT))
import energy_model as em
mp.mp.dps=75
M=mp.mpf
GAM=M('1.561154518256'); OMG=M('261.310962155227')
KAP=M('261.365756187')/1000
R0=M('0.084287616')
GAMMA=M('1.7595e11')/(2*mp.pi*M('1e9'))
def out(v):return float(v)
def cm(z):return [float(mp.re(z)),float(mp.im(z))]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def magnetic(r,gamma,omega=OMG):
    C=(r+1/r)/2;T=(1/r-r)/2
    if gamma==0:alpha=M(0);eta=M(1);wm=omega
    else:
        rho=gamma/omega
        alpha=rho/mp.sqrt(C*C+rho*rho*T*T)
        eta=1/(1+alpha*alpha)
        wm=gamma/(eta*alpha*C)
    a=eta*alpha/r;b=eta*eta*wm*(1+alpha*alpha)
    return dict(r=r,alpha=alpha,eta=eta,wm=wm,a=a,b=b,BE=wm/GAMMA*T,BA=wm/GAMMA*r)

def matrices(m,g,w,k=KAP):
    K=mp.diag([m['wm']]*4+[w]*2)
    K[0,4]=K[4,0]=-2*mp.sqrt(2)*g
    J=mp.zeros(6);J[0,3]=-m['eta'];J[3,0]=m['eta']
    J[1,2]=m['eta'];J[2,1]=-m['eta'];J[4,5]=1;J[5,4]=-1
    R=mp.diag([m['eta']*m['alpha']/m['r']]*2+[m['eta']*m['alpha']*m['r']]*2+[0,2*k/w])
    return K,J,R,(J-R)*K

def ep(r,gamma=GAM,kappa=KAP):
    m=magnetic(r,gamma);S=OMG*OMG+gamma*gamma
    if gamma==kappa:raise ValueError('Zero-coupling ordinary coincidence, not an EP.')
    def residual(g,w):
        c=(S+w*w+4*gamma*kappa-(gamma+kappa)**2)/2
        linear=2*gamma*w*w+2*kappa*S-8*w*g*g*m['a']-2*(gamma+kappa)*c
        constant=S*w*w-8*w*g*g*m['b']-c*c
        return linear/OMG,constant/(OMG*OMG)
    seedg=abs(gamma-kappa)/(2*mp.sqrt(2))
    g,w=mp.findroot(residual,(seedg,OMG),tol=M('1e-65'),maxsteps=50)
    c=(S+w*w+4*gamma*kappa-(gamma+kappa)**2)/2
    eplambda=-(gamma+kappa)/2-mp.j*mp.sqrt(c-(gamma+kappa)**2/4)
    def D(z):return z*z+2*gamma*z+S
    def P(z):return D(z)*(z*z+2*kappa*z+w*w)-8*w*g*g*(m['a']*z+m['b'])
    def fullchar(z):return D(z)*P(z)
    # Two independent representations: coefficients and 6D matrix determinants.
    K,J,R,A=matrices(m,g,w,kappa)
    Af=np.array(A.tolist(),dtype=float);Kf=np.array(K.tolist(),dtype=float);Rf=np.array(R.tolist(),dtype=float)
    vals=np.linalg.eigvals(Af)
    Z=Af-complex(eplambda)*np.eye(6)
    _,sv,vh=np.linalg.svd(Z);v=vh.conj().T[:,-1]
    chain=np.linalg.lstsq(np.vstack([Z,v.conj()[None,:]]),np.r_[v,0],rcond=1e-12)[0]
    detchecks=[]
    for z in [eplambda+M('.13')+M('.19')*mp.j,M('-.5')-mp.j*(OMG+M('1.2')),M('.3')+M('.5')*mp.j]:
        pmat=mp.det(z*mp.eye(6)-A);pform=fullchar(z)
        detchecks.append(abs(pmat-pform)/max(abs(pform),1))
    # Cross-check old audited real-state source, using its native conventions.
    oldm=replace(em.Material(),BE=float(m['BE']),BA=float(m['BA']),alpha=float(m['alpha']),gamma_GHz_T=float(GAMMA))
    olds=replace(em.Setup(),fc=float(w),Qc=float(w/(2*kappa)),gilbert=True)
    oldA=em.matrix(float(g),0,olds,oldm)
    r=float(r);L=np.zeros((6,6));L[0,[0,2]]=1/np.sqrt(2*r);L[1,[1,3]]=1/np.sqrt(2*r)
    L[2,[0,2]]=[np.sqrt(r/2),-np.sqrt(r/2)];L[3,[1,3]]=[np.sqrt(r/2),-np.sqrt(r/2)]
    L[4,4]=L[5,5]=np.sqrt(2*float(GAMMA)*oldm.BE/float(w))
    matcherr=float(np.max(abs(Af-L@oldA@np.linalg.inv(L))))
    A0=np.array(matrices(m,0,w,kappa)[3].tolist(),dtype=float)
    free=np.linalg.eigvals(A0[:4,:4]);target=np.array([-float(gamma)+1j*float(OMG),-float(gamma)-1j*float(OMG)]*2)
    ii,jj=linear_sum_assignment(abs(free[:,None]-target[None,:]))
    row=dict(r=r,G_U_GHz=float(g),cavity_frequency_GHz=float(w),cavity_offset_from_Omega_MHz=float((w-OMG)*1000),
      Qc=float(w/(2*kappa)),kappa_GHz=float(kappa),Gamma_GHz=float(gamma),Omega_GHz=float(OMG),
      alpha=float(m['alpha']),omega_m_GHz=float(m['wm']),BE_T=float(m['BE']),BA_T=float(m['BA']),
      numerator_lambda=float(m['a']),numerator_constant_GHz=float(m['b']),numerator_ratio_per_GHz=float(m['a']/m['b']),
      ep_lambda_GHz=cm(eplambda),ep_eigenvalues_exact=[cm(eplambda),cm(mp.conj(eplambda))],
      G_decimal=mp.nstr(g,65),wc_decimal=mp.nstr(w,65),ep_lambda_decimal=[mp.nstr(mp.re(eplambda),65),mp.nstr(mp.im(eplambda),65)],
      normalized_p=mp.nstr(abs(fullchar(eplambda))/OMG**6,8),normalized_dp=mp.nstr(abs(mp.diff(fullchar,eplambda))/OMG**5,8),
      normalized_ddp=mp.nstr(abs(mp.diff(fullchar,eplambda,2))/OMG**4,14),
      bright_quartic_double_root_residual=mp.nstr(max(abs(P(eplambda))/OMG**4,abs(mp.diff(P,eplambda))/OMG**3),8),
      dark_factor_at_ep_normalized=mp.nstr(abs(D(eplambda))/OMG**2,14),
      matrix_characteristic_max_relative_error=mp.nstr(max(detchecks),8),
      min_energy_eigenvalue_GHz=float(np.linalg.eigvalsh(Kf).min()),max_pole_real_GHz=float(vals.real.max()),
      shifted_singular_values_GHz=sv.tolist(),eigenvector_residual=float(np.linalg.norm(Z@v)),
      jordan_chain_residual=float(np.linalg.norm(Z@chain-v)),audited_source_matrix_max_difference_GHz=matcherr,
      free_spin_pole_match_error_Hz=float(np.max(abs(free[ii]-target[jj]))*1e9),
      lyapunov_identity_relative_residual=float(np.linalg.norm(Af.T@Kf+Kf@Af+2*Kf@Rf@Kf)/np.linalg.norm(Kf@Af)),
      energy_positive_coupling_ratio=float(8*g*g/(w*m['wm'])))
    assert g>0 and w>0 and c>(gamma+kappa)**2/4
    assert float(row['normalized_p'])<1e-55 and float(row['normalized_dp'])<1e-55
    assert float(row['normalized_ddp'])>1e-8 and float(row['dark_factor_at_ep_normalized'])>1e-8
    assert row['min_energy_eigenvalue_GHz']>0 and row['energy_positive_coupling_ratio']<1
    assert matcherr<1e-8 and row['jordan_chain_residual']<1e-8 and sv[-2]>.01
    return row

def response_test(rows):
    ref=next(x for x in rows if x['r']==float(R0));g=ref['G_U_GHz'];w=ref['cavity_frequency_GHz'];k=float(KAP)
    f=np.linspace(float(OMG)-4,float(OMG)+4,16001);z=-1j*f
    D=z*z+2*float(GAM)*z+float(OMG*OMG+GAM*GAM)
    curves={};records=[]
    for r in [M('.014'),R0,M('.5'),M('.99')]:
        m=magnetic(r,GAM);N=float(m['a'])*z+float(m['b'])
        P=D*(z*z+2*k*z+w*w)-8*w*g*g*N
        series=-2*.5*k*z*D/P
        Af=np.array(matrices(m,M(str(g)),M(str(w)))[3].tolist(),dtype=float)
        checks=[]
        for index in np.linspace(0,len(f)-1,17,dtype=int):
            rr=np.linalg.inv(z[index]*np.eye(6)-Af)
            checks.append(abs(series[index]-(-2*.5*k*rr[5,5])))
        curves[str(float(r))]=series
        records.append({'r':float(r),'direct_resolvent_max_abs_difference':max(checks)})
        assert max(checks)<1e-11
    refS=curves[str(float(R0))]
    for rec in records:
        cur=curves[str(rec['r'])]
        rec['max_complex_difference_percent_of_reference_peak']=float(100*np.max(abs(cur-refS))/np.max(abs(refS)))
    np.savez_compressed(HERE/'same_operating_point_response.npz',frequency_GHz=f,**{f'S21_r_{r}':v for r,v in curves.items()})
    return {'fixed_G_U_GHz':g,'fixed_cavity_frequency_GHz':w,'fixed_kappa_GHz':k,'eta_external':.5,
     'B0_T':0,'frequency_window_GHz':[f[0],f[-1]],'samples':len(f),
     'normalization':'100 max|S_r-S_ref| / max|S_ref|; same physical drive/readout; this is not an experimental detection claim.',
     'records':records}

def main():
    rgrid=sorted(set([*np.geomspace(.014,.99,33),float(R0),.1,.2,.5]))
    rows=[ep(M(str(r))) for r in rgrid]
    limits=[]
    for scale in [1.,.1,.01,.001,0.]:
        rr=rows if scale==1 else [ep(M(str(r)),GAM*M(str(scale))) for r in [.014,float(R0),.2,.5,.99]]
        gs=[x['G_U_GHz'] for x in rr];fs=[x['cavity_frequency_GHz'] for x in rr]
        limits.append({'target_Gamma_scale':scale,'target_Gamma_GHz':float(GAM*M(str(scale))),
          'samples':len(rr),'G_span_Hz':(max(gs)-min(gs))*1e9,'G_span_percent':100*(max(gs)/min(gs)-1),
          'cavity_frequency_span_Hz':(max(fs)-min(fs))*1e9,
          'scope':'Omega and kappa fixed; Gamma reduced as a controlled family. Not r->0 at fixed Gamma.',
          'rows':rr if scale!=1 else []})
    response=response_test(rows)
    write('results.json',{'units':'Angular rates divided by 2*pi GHz; G_U is per chiral branch from common energy.',
      'dps':75,'target_Gamma_GHz':float(GAM),'target_Omega_GHz':float(OMG),'fixed_kappa_GHz':float(KAP),
      'bias_T':0,'reference_r':float(R0),'r_grid':rgrid,'rows':rows,'regular_damping_limit':limits,'same_operating_point_response':response})
    write('input_manifest.json',{'frozen_energy_model_sha256':digest(FROZEN),'frozen_input_path':'inputs/energy_model.py',
       'script_sha256':digest(Path(__file__)),'target_pole_origin':'printed native-Gilbert zero-field target; last retained decimal is input precision, not experimental precision',
       'runtime':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__,'matplotlib':matplotlib.__version__},
       'bath':'total-field native Gilbert plus cavity viscous energy-gradient bath',
       'scope':'New analytic/CPU EP verification. No fresh GPU trajectories. No local-amplitude-bath equivalence claimed.'})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(12,3.55),constrained_layout=True)
    rs=np.array([x['r'] for x in rows]);ref=next(x for x in rows if x['r']==float(R0))
    axs[0].plot(rs,[x['cavity_offset_from_Omega_MHz'] for x in rows]);axs[0].set(xlabel=r'Squeezing ratio $r$',ylabel=r'$(f_{c,EP}-f_{d0})$ (MHz)',title='(a) Same entire free-spin spectrum')
    axs[1].plot(rs,[(x['G_U_GHz']/ref['G_U_GHz']-1)*1e6 for x in rows],color='tab:orange');axs[1].set(xlabel=r'Squeezing ratio $r$',ylabel=r'$(G_{EP}/G_{EP,ref}-1)$ (ppm)',title='(b) EP coupling at fixed cavity loss')
    dat=np.load(HERE/'same_operating_point_response.npz');rf=dat[f'S21_r_{float(R0)}']
    for rv in [.014,.5,.99]:
        d=dat[f'S21_r_{rv}']-rf
        axs[2].plot(dat['frequency_GHz']-float(OMG),100*abs(d)/max(abs(rf)),label=f'r={rv:g}')
    axs[2].set(xlabel=r'$f-f_{d0}$ (GHz)',ylabel='Complex transmission difference (%)',title='(c) Same cavity and coupling')
    axs[2].legend(frameon=False,fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(HERE/'zero_field_ep.png',dpi=220);fig.savefig(HERE/'zero_field_ep.svg')
    print(json.dumps({'r_points':len(rows),'endpoints':[rows[0],rows[-1]],'limits':[{k:v for k,v in x.items() if k!='rows'} for x in limits],'response':response},indent=2))

if __name__=='__main__':main()
