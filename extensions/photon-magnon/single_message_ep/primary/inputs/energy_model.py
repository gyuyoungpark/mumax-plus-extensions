"""Energy-derived AFM + primary cavity (+ auxiliary), without an RWA.

Time unit: 1/(2*pi*1 GHz). Couplings/rates/frequencies below are angular
frequencies divided by 2*pi*1 GHz. Magnetic bias is in tesla.

State: [mAx,mAy,mBx,mBy,Re(a)/s_a,Im(a)/s_a,(Re(d)/s_d,Im(d)/s_d)].
E/E0 = x.T K x/2, E0=mu_sub*B_E. Every coupling is in the same real energy.
A=(Omega-R)K. Spin damping is the coded LL convention by default; the
Gilbert denominator can be enabled. Photon baths are passive viscous
quadrature baths: dot(Im(a)) contains -2*kappa*Im(a), with amplitude
decay kappa. This is an explicit bath assumption, not device calibration.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class Material:
    BE: float = 4 * .84e-12 / (624e3 * (3.13e-10)**2)
    BA: float = 2 * 245.4e3 / 624e3
    gamma_GHz_T: float = 1.76e11 / (2*np.pi*1e9)
    alpha: float = .001
    mu_sub: float = 624e3 * (40e-9)*(40e-9)*(10e-9)

    @property
    def r(self):
        return np.sqrt(self.BA/(2*self.BE+self.BA))

    @property
    def fm(self):
        return self.gamma_GHz_T*np.sqrt(self.BA*(2*self.BE+self.BA))


@dataclass(frozen=True)
class Setup:
    fc: float = Material().fm + 5.
    Qc: float = 500.
    auxiliary: bool = False
    delta_d: float = .2
    kappa_d: float = 1.333
    J: float = .4
    g_d: float = .4
    gilbert: bool = False
    geometry: str = 'uniform'
    photon_bath: str = 'viscous'


def components(g_a, bias, setup=Setup(), material=Material(), mp=None):
    """g_a and g_d are physical uniform-field per-branch Zeeman couplings.

    For staggered controls the same B_zpf is retained, so the actual
    branch coupling is g_a/r. No legacy g0 mapping is assumed.
    """
    scalar=(lambda x:mp.mpf(str(x))) if mp else float
    sqrt=mp.sqrt if mp else np.sqrt
    zeros=mp.zeros if mp else (lambda n:np.zeros((n,n)))
    n=8 if setup.auxiliary else 6
    K,O,R=zeros(n),zeros(n),zeros(n)
    be,ba,gg,alpha=map(scalar,(material.BE,material.BA,material.gamma_GHz_T,material.alpha))
    bias=scalar(bias) if not mp else bias
    g_a=scalar(g_a) if not mp else g_a
    fc=scalar(setup.fc)
    rr=sqrt(ba/(2*be+ba))
    sg=1 if setup.geometry=='uniform' else -1
    for i in (0,1): K[i,i]=(be+ba+bias)/be
    for i in (2,3): K[i,i]=(be+ba-bias)/be
    for i,j in ((0,2),(1,3)): K[i,j]=K[j,i]=1
    norm=1+alpha**2 if setup.gilbert else 1
    for q,p,sign in ((0,1,-1),(2,3,1)):
        O[q,p]=sign*gg*be/norm
        O[p,q]=-sign*gg*be/norm
        R[q,q]=R[p,p]=alpha*gg*be/norm
    photons=[(4,fc,g_a,fc/(2*scalar(setup.Qc)))]
    if setup.auxiliary:
        photons.append((6,fc+scalar(setup.delta_d),scalar(setup.g_d),scalar(setup.kappa_d)))
    for q,freq,g,kappa in photons:
        K[q,q]=K[q+1,q+1]=2
        O[q,q+1]=freq/2
        O[q+1,q]=-freq/2
        R[q+1,q+1]=kappa
        coupling=-2*g*sqrt(2/(gg*be*freq*rr))
        K[0,q]=K[q,0]=coupling
        K[2,q]=K[q,2]=sg*coupling
    if setup.auxiliary:
        coupling=4*scalar(setup.J)/sqrt(fc*(fc+scalar(setup.delta_d)))
        K[4,6]=K[6,4]=coupling
    A=(O-R)*K if mp else (O-R)@K
    if setup.photon_bath=='local_amplitude':
        # Sensitivity control matching the submitted cavity decay convention.
        # This version is not asserted to satisfy the total-energy Lyapunov law.
        A=O*K if mp else O@K
        spinR=zeros(n)
        for i in range(4): spinR[i,i]=R[i,i]
        A-=spinR*K if mp else spinR@K
        for q,freq,g,kappa in photons:
            A[q,q]-=kappa
            A[q+1,q+1]-=kappa
    return K,O,R,A


def matrix(g_a,bias,setup=Setup(),material=Material()):
    return components(g_a,bias,setup,material)[3]


def determinant_and_derivative(z):
    det=np.linalg.det(z)
    der=sum(np.linalg.det(np.delete(np.delete(z,i,axis=0),i,axis=1)) for i in range(len(z)))
    return det,der


def solve_ep(setup=Setup(),material=Material(),seed=None):
    if seed is None:
        bc=(setup.fc-material.fm)/material.gamma_GHz_T
        gamma=material.alpha*material.gamma_GHz_T*(material.BA+material.BE)*(1+(setup.fc-material.fm)/material.fm)
        kap=setup.fc/(2*setup.Qc)
        seed=[abs(gamma-kap)/2,bc*1e3,-(gamma+kap)/2,0]
    n=8 if setup.auxiliary else 6
    scale=(2*setup.fc)**(n//2)*10
    def fun(x):
        A=matrix(x[0],x[1]*1e-3,setup,material)
        eta=x[2]+1j*x[3]
        z=eta*np.eye(n)-(A+1j*setup.fc*np.eye(n))
        f,df=determinant_and_derivative(z)
        return [f.real/scale,f.imag/scale,df.real/scale,df.imag/scale]
    sol=root(fun,seed,tol=1e-10)
    return {'x':sol.x,'success':bool(sol.success),'residual':float(np.linalg.norm(fun(sol.x))),
            'message':sol.message}


def refine_ep(initial,setup=Setup(),material=Material(),dps=55):
    import mpmath as mp
    with mp.workdps(dps):
        n=8 if setup.auxiliary else 6
        fc=mp.mpf(str(setup.fc))
        scale=(2*fc)**(n//2)*10
        def equations(g,b,er,ei):
            A=components(g,b/1000,setup,material,mp)[3]
            eta=mp.mpc(er,ei)
            def characteristic(w):
                return mp.det(w*mp.eye(n)-(A+mp.j*fc*mp.eye(n)))/scale
            f=characteristic(eta)
            df=mp.diff(characteristic,eta)
            return mp.re(f),mp.im(f),mp.re(df),mp.im(df)
        start=tuple(mp.mpf(str(float(x))) for x in initial)
        answer=mp.findroot(equations,start,tol=mp.mpf('1e-42'),maxsteps=30)
        residual=max(abs(v) for v in equations(*answer))
        A=components(answer[0],answer[1]/1000,setup,material,mp)[3]
        eta=mp.mpc(answer[2],answer[3])
        p2=mp.diff(lambda w: mp.det(w*mp.eye(n)-(A+mp.j*fc*mp.eye(n)))/scale,eta,2)
        return {'x':[float(v) for v in answer],
                'x_decimal':[mp.nstr(v,45) for v in answer],
                'normalized_double_root_residual':mp.nstr(residual,8),
                'normalized_characteristic_second_derivative_abs':mp.nstr(abs(p2),12),'dps':dps}


def bzpf(g_a,material=Material()):
    hbar=1.054571817e-34
    gstar=g_a*2*np.pi*1e9/np.sqrt(material.r)
    return gstar/np.sqrt(material.gamma_GHz_T*2*np.pi*1e9*material.mu_sub/(2*hbar))


def pole_record(g_a,bias,setup=Setup(),material=Material(),ep_eta=None):
    K,O,R,A=components(g_a,bias,setup,material)
    w,V=np.linalg.eig(A)
    order=np.argsort(-w.imag)
    rec={'g_U_GHz':float(g_a),'B_T':float(bias),'B_zpf_T':float(bzpf(g_a,material)),
         'geometry':setup.geometry,
         'physical_branch_coupling_GHz':float(g_a if setup.geometry=='uniform' else g_a/material.r),
         'poles_GHz':[{'real':float(v.real),'imag':float(v.imag)} for v in w[order]],
         'max_real_GHz':float(w.real.max()),'trace_GHz':float(np.trace(A)),
         'K_eigen_min':float(np.linalg.eigvalsh(K).min()),
         'R_eigen_min':float(np.linalg.eigvalsh(R).min()),
         'energy_dissipation_identity_relative_residual':float(np.linalg.norm(A.T@K+K@A+2*K@R@K)/np.linalg.norm(K@A))}
    if ep_eta is not None:
        lam=ep_eta-1j*setup.fc
        z=A-lam*np.eye(len(A))
        _,s,vh=np.linalg.svd(z)
        v=vh.conj().T[:,-1]
        chain=np.linalg.lstsq(np.vstack((z,v.conj()[None,:])),np.r_[v,0],rcond=1e-13)[0]
        ids=np.argsort(abs(w-lam))[:2]
        rec.update({'ep_lambda_GHz':[float(lam.real),float(lam.imag)],
                    'direct_double_precision_pair_gap_Hz':float(abs(w[ids[0]]-w[ids[1]])*1e9),
                    'shifted_singular_values_GHz':s.tolist(),
                    'eigenvector_residual':float(np.linalg.norm(z@v)),
                    'jordan_chain_relative_residual':float(np.linalg.norm(z@chain-v)),
                    'other_positive_frequency_poles_GHz':[
                        [float(vv.real),float(vv.imag)] for i,vv in enumerate(w) if vv.imag<0 and i not in ids]})
    return rec


def schur(A,lam):
    """Exact auxiliary elimination, retaining both auxiliary quadratures."""
    App,Apd,Adp,Add=A[:6,:6],A[:6,6:],A[6:,:6],A[6:,6:]
    Sigma=Apd@np.linalg.solve(lam*np.eye(2)-Add,Adp)
    return App+Sigma,Sigma


def two_mode(g_a,bias,setup=Setup(),material=Material(),include_lower=False):
    if setup.auxiliary:
        raise ValueError('Two-mode calibration is for the no-auxiliary baseline.')
    A0=matrix(0,bias,setup,material)
    w,V=np.linalg.eig(A0)
    Vinv=np.linalg.inv(V)
    cav=np.sum(abs(V[4:6,:])**2,axis=0)/np.sum(abs(V)**2,axis=0)>.5
    ic=[i for i in range(6) if cav[i] and w[i].imag<0][0]
    im=max([i for i in range(6) if not cav[i] and w[i].imag<0],key=lambda i:-w[i].imag)
    derivative=Vinv@(matrix(1,bias,setup,material)-A0)@V
    ids=[ic,im]
    if include_lower:
        ids += [i for i in range(6) if not cav[i] and w[i].imag<0 and i!=im]
    return np.diag(w[ids])+g_a*derivative[np.ix_(ids,ids)]


def solve_two_mode(setup=Setup(),material=Material(),seed=None):
    if seed is None:
        seed=solve_ep(setup,material)['x'][:2]
    def fun(x):
        M=two_mode(x[0],x[1]*1e-3,setup,material)
        d=(M[0,0]-M[1,1])**2+4*M[0,1]*M[1,0]
        return [d.real,d.imag]
    sol=root(fun,seed,tol=1e-10)
    return {'g_U_GHz':float(sol.x[0]),'B_mT':float(sol.x[1]),
            'success':bool(sol.success),'residual':float(np.linalg.norm(fun(sol.x)))}


def mode_weights(v,bias,setup=Setup(),material=Material()):
    """Positive/negative bare conservative mode weights, energy normalized."""
    bare=replace(setup,J=0,g_d=0,photon_bath='viscous')
    mat=replace(material,alpha=0)
    K,O,_,_=components(0,bias,bare,mat)
    w,V=np.linalg.eig(O@K)
    for i in range(len(w)):
        V[:,i]/=np.sqrt(np.vdot(V[:,i],K@V[:,i]).real)
    coeff=np.linalg.solve(V,v)
    wt=abs(coeff)**2
    wt/=wt.sum()
    result={}
    for i in range(len(w)):
        vc=V[:,i]
        photon=np.linalg.norm(vc[4:6])>np.linalg.norm(vc[:4])
        aux=len(vc)>6 and np.linalg.norm(vc[6:])>max(np.linalg.norm(vc[:4]),np.linalg.norm(vc[4:6]))
        if aux:label='auxiliary'
        elif photon:label='primary'
        else:label='upper_magnon' if abs(w[i].imag)>material.fm else 'lower_magnon'
        label+=('_positive' if w[i].imag<0 else '_negative')
        result[label]=float(wt[i])
    return result


def ep_weights(x,setup=Setup(),material=Material()):
    A=matrix(x[0],x[1]*1e-3,setup,material)
    lam=x[2]+1j*(x[3]-setup.fc)
    _,_,vh=np.linalg.svd(A-lam*np.eye(len(A)))
    return mode_weights(vh.conj().T[:,-1],x[1]*1e-3,setup,material)


def continue_auxiliary(setup,material=Material(),seed=None,steps=16):
    base=replace(setup,auxiliary=False)
    initial=solve_ep(base,material,seed)
    x=initial['x']
    path=[]
    for t in np.linspace(0,1,steps+1):
        current=replace(setup,J=setup.J*t,g_d=setup.g_d*t)
        result=solve_ep(current,material,x)
        if result['residual']>1e-8 or result['x'][0]<=0:
            return {'success':False,'failed_t':float(t),'path':path,'last':result}
        x=result['x']
        path.append({'t':float(t),'x':x.tolist(),'residual':result['residual']})
    return {'success':True,'path':path,'x':x}


def nonlinear_rhs(x,g_a,bias,setup=Setup(),material=Material()):
    """Independent torque/oscillator evaluation for checking the linearization."""
    K,_,_,_=components(g_a,bias,setup,material)
    ma=np.r_[x[:2],np.sqrt(1-x[0]**2-x[1]**2)]
    mb=np.r_[x[2:4],-np.sqrt(1-x[2]**2-x[3]**2)]
    ha=-material.BE*sum(K[0,q]*x[q] for q in range(4,len(x),2))
    hb=-material.BE*sum(K[2,q]*x[q] for q in range(4,len(x),2))
    ba=-material.BE*mb+np.array([ha,0,bias+material.BA*ma[2]])
    bb=-material.BE*ma+np.array([hb,0,bias+material.BA*mb[2]])
    norm=1+material.alpha**2 if setup.gilbert else 1
    fa=-material.gamma_GHz_T*(np.cross(ma,ba)+material.alpha*np.cross(ma,np.cross(ma,ba)))/norm
    fb=-material.gamma_GHz_T*(np.cross(mb,bb)+material.alpha*np.cross(mb,np.cross(mb,bb)))/norm
    out=np.zeros_like(x)
    out[:2],out[2:4]=fa[:2],fb[:2]
    grad=K@x
    for q,f,k in [(4,setup.fc,setup.fc/(2*setup.Qc))]+(
            [(6,setup.fc+setup.delta_d,setup.kappa_d)] if setup.auxiliary else []):
        out[q]=f*x[q+1]
        out[q+1]=-f*grad[q]/2-2*k*x[q+1]
        if setup.photon_bath=='local_amplitude':
            out[q]-=k*x[q]
            out[q+1]+=k*x[q+1]
    return out


def bare_modal_matrix(g_a,bias,setup=Setup(),material=Material()):
    bare=replace(setup,J=0,g_d=0,photon_bath='viscous')
    K,O,_,_=components(0,bias,bare,replace(material,alpha=0))
    w,V=np.linalg.eig(O@K)
    labels={}
    for i in range(len(w)):
        V[:,i]/=np.sqrt(np.vdot(V[:,i],K@V[:,i]).real)
        v=V[:,i]
        if setup.auxiliary and np.linalg.norm(v[6:])>max(np.linalg.norm(v[:4]),np.linalg.norm(v[4:6])):
            name='d'
        elif np.linalg.norm(v[4:6])>np.linalg.norm(v[:4]):name='c'
        else:name='u' if abs(w[i].imag)>material.fm else 'l'
        name+= '+' if w[i].imag<0 else '-'
        labels[name]=i
    order=[labels[k] for k in ['c+','u+','l+','c-','u-','l-']+(['d+','d-'] if setup.auxiliary else [])]
    V=V[:,order]
    return np.linalg.solve(V,matrix(g_a,bias,setup,material)@V),V


def self_energy_diagnostics(g_a,bias,setup,material=Material(),probe=None):
    probe=setup.fc if probe is None else probe
    M,V=bare_modal_matrix(g_a,bias,setup,material)
    eff,S=schur(M,-1j*probe)
    def pair(z):return [float(z.real),float(z.imag)]
    basef=1j*M[0,1];baseb=1j*M[1,0]
    gf=1j*eff[0,1];gb=1j*eff[1,0]
    return {'probe_GHz':probe,
            'induced_primary_amplitude_loss_MHz':float(-S[0,0].real*1e3),
            'induced_upper_amplitude_loss_MHz':float(-S[1,1].real*1e3),
            'induced_primary_frequency_shift_MHz':float(-S[0,0].imag*1e3),
            'induced_upper_frequency_shift_MHz':float(-S[1,1].imag*1e3),
            'forward_H_GHz':pair(gf),'reverse_H_GHz':pair(gb),
            'forward_phase_change_deg':float(np.angle(gf/basef,deg=True)),
            'reverse_phase_change_deg':float(np.angle(gb/baseb,deg=True)),
            'coupling_product_magnitude_GHz':float(np.sqrt(abs(gf*gb))),
            'full_modal_self_energy_real':S.real.tolist(),'full_modal_self_energy_imag':S.imag.tolist()}


def solve_positive_three(setup=Setup(),material=Material(),seed=None):
    if seed is None:seed=solve_ep(setup,material)['x']
    def fun(x):
        M=two_mode(x[0],x[1]*1e-3,setup,material,include_lower=True)
        lam=x[2]+1j*(x[3]-setup.fc)
        f,d=determinant_and_derivative(lam*np.eye(3)-M)
        return [f.real/10,f.imag/10,d.real/10,d.imag/10]
    sol=root(fun,seed,tol=1e-10)
    return {'x':sol.x,'success':bool(sol.success),'residual':float(np.linalg.norm(fun(sol.x)))}
