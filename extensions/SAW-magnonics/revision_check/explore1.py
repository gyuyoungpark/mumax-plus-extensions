import sys, numpy as np
sys.path.insert(0,"D:/mumax-plus-dev/extension/SAW-magnonics/revision_check")
import saw_analysis as sa

D="D:/mumax-plus-dev/extension/SAW-magnonics/data/sim40_checkpoints/sim40_full_2fK_eps7e-05.npz"
d=np.load(D)
m=d['my_xt'].astype(np.float64); dx=float(d['CX']); dt=float(d['DT_REC'])
nt,nx=m.shape
L=nx*dx; dk=2*np.pi/L
q=sa.pump_wavevector(float(d['f_saw']),3500.0)
print("nt,nx",nt,nx,"dx",dx,"dt",dt)
print("L um",L*1e6,"dk um^-1",sa.to_inv_um(dk),"q um^-1",sa.to_inv_um(q),"qL/2pi",q*L/(2*np.pi))
print("kSAW/2",sa.to_inv_um(q/2), " q/dk =", q/dk)
print("bin4,5:",sa.to_inv_um(4*dk),sa.to_inv_um(5*dk),"sum",sa.to_inv_um(9*dk))
print("m range", m.min(), m.max(), "rms", m.std())
# time trace amplitude growth of volume average
print("mean|m| per 50 steps:", [float(np.abs(m[i]).mean()) for i in range(0,nt,75)])

k,f,M = sa.spectrum(m,dx,dt,window='hann',trange=(375,751),window_x='rect')
print("df bin", f[1]-f[0])
i,sl = sa.f_slice(M,f,3.0e9)
print("f slice at",f[i])
P=np.abs(sl)**2
kum=sa.to_inv_um(k)
order=np.argsort(P)[::-1][:20]
tot=P.sum()
for j in order:
    print("  k=%9.4f  P/tot=%.5e  |M|=%.4e  bin=%+d"%(kum[j],P[j]/tot,np.abs(sl[j]), round(k[j]/dk)))
