import sys, numpy as np
sys.path.insert(0,"D:/mumax-plus-dev/extension/SAW-magnonics/revision_check")
import saw_analysis as sa
np.set_printoptions(precision=4,suppress=False,linewidth=200)
D="D:/mumax-plus-dev/extension/SAW-magnonics/data/sim40_checkpoints/sim40_full_2fK_eps7e-05.npz"
d=np.load(D); m=d['my_xt'].astype(np.float64); dx=float(d['CX']); dt=float(d['DT_REC'])
nt,nx=m.shape; L=nx*dx; dk=2*np.pi/L
# per-bin time trace: complex amplitude a_j(t) = fft_x(m)(t, j)
X=np.fft.fft(m-m.mean(axis=1,keepdims=True),axis=1)/nx
t=np.arange(nt)*dt
print("per-bin |a_j| at selected times (x1e3):")
hdr="  t[ns] "+"".join("  bin%-2d "%j for j in range(0,13))
print(hdr)
for i in range(0,nt,50):
    print("  %5.2f "%(t[i]*1e9)+"".join(" %6.3f"%(abs(X[i,j])*1e3) for j in range(0,13)))
# k=0 separately (mean over x)
m0=m.mean(axis=1)
print("k0 |m0| at same times x1e3:", " ".join("%.3f"%(abs(m0[i])*1e3) for i in range(0,nt,50)))
# growth rates per bin over 2-7 ns  (fit on envelope via |analytic|)
from numpy.fft import rfft, irfft
for j in [1,2,3,4,5,6,7,8,9]:
    a=np.abs(X[:,j])
    # sliding max envelope
    g,b,r2=sa.growth_fit(t,a,t0=2e-9,t1=7e-9)
    g2,b2,r22=sa.growth_fit(t,a,t0=7.5e-9,t1=15e-9)
    print("bin %2d  Gamma(2-7ns)=%.4e r2=%.3f   Gamma(7.5-15ns)=%.4e r2=%.3f"%(j,g,r2,g2,r22))
