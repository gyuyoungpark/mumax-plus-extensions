import numpy as np
V,F=3500.0,6.0e9
q=2*np.pi*F/V
def check(NX,dx,label):
    L=NX*dx; x=(np.arange(NX)+0.5)*dx          # kernel: coord=(i+0.5)*cs.x
    eps=np.sin(q*x)                             # t=0, phi=0
    S=np.abs(np.fft.fft(eps))**2; S/=S.sum()
    k=2*np.pi*np.fft.fftfreq(NX,d=dx); dk=2*np.pi/L
    order=np.argsort(-S)[:6]
    print("%-28s L=%7.4f um dk=%.6f /um  q/dk=%.4f" % (label,L*1e6,dk/1e6,q/dk))
    for j in order:
        print("      bin %+5d  k=%+9.6f /um  frac=%.6f" % (round(k[j]/dk), k[j]/1e6, S[j]))
    tot2=sum(S[j] for j in order[:2])
    print("      top-2 fraction = %.9f ; leakage outside top-2 = %.3e\n" % (tot2,1-tot2))
check(1024,5e-9,"sim40 (incommensurate)")
check(1400,5e-9,"n=12 commensurate")
check(2100,5e-9,"n=18 commensurate")
check(1050,5e-9,"n=9 commensurate but ODD")
