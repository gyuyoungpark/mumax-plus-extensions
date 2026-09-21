"""Seed calibration: what magnon temperature does a given seed amplitude correspond to?

Rayleigh-Jeans equipartition for one spin-wave mode of a discretised film.
A single Fourier mode m_perp(x) = a cos(kx + phi) in a magnet of total volume
V_tot carries  E = (1/4) Ms V_tot B_k a^2  (two transverse quadratures, so
<E> = k_B T by equipartition):

        a_th(k, T) = sqrt( 4 k_B T / (Ms V_tot B_k) ),
        B_k = sqrt( (B0 + 2A k^2/Ms) (B0 + mu0 Ms + 2A k^2/Ms) )   [Kittel form]

CPU only.  Run:  python seed_calibration.py
"""
import numpy as np
KB, MU0, GAMMA = 1.380649e-23, 4e-7*np.pi, 1.76e11
V_SAW, F_SAW = 3500.0, 6.0e9
q = 2*np.pi*F_SAW/V_SAW

def B_k(k, B0, Ms, A):
    ex = 2*A*k**2/Ms
    return np.sqrt((B0+ex)*(B0+MU0*Ms+ex))

def a_th(k, T, B0, Ms, A, V_tot):
    return np.sqrt(4*KB*T/(Ms*V_tot*B_k(k,B0,Ms,A)))

MS, AEX, B0 = 140e3, 3.65e-12, 50.629e-3            # YIG, sim40 B0
VCELL = 5e-9*10e-9*20e-9
print("YIG  Ms=%.0f kA/m  A=%.2f pJ/m  B0=%.3f mT  V_cell=%.3e m^3" % (MS/1e3,AEX*1e12,B0*1e3,VCELL))
print("B_k(0)=%.4f T   B_k(q/2)=%.4f T   B_k(q)=%.4f T" % (B_k(0,B0,MS,AEX),B_k(q/2,B0,MS,AEX),B_k(q,B0,MS,AEX)))

print("\n--- thermal amplitude of the SINGLE mode at k=q/2, per box ---")
print("%-22s %8s %12s %12s %12s" % ("box","N cells","V_tot(m^3)","a_th(300K)","a_th(1K)"))
for lbl,NX,NY in (("sim40 1024x8",1024,8),("n=12 1400x8",1400,8),("n=18 2100x8",2100,8),("n=6 700x8",700,8)):
    V=NX*NY*VCELL
    print("%-22s %8d %12.4e %12.4e %12.4e" % (lbl,NX*NY,V,a_th(q/2,300,B0,MS,AEX,V),a_th(q/2,1.0,B0,MS,AEX,V)))

print("\n--- what T does the sim43-49 seed correspond to? ---")
print("sim43/44/46/47/48/49: per-cell Gaussian sigma=1e-4 on m_y and m_z, NX=1024, NY=8.")
print("k-flat over ALL N=NX*NY modes -> per-mode rms amplitude sigma/sqrt(N):")
for NX,NY in ((1024,8),(1400,8)):
    N=NX*NY; V=N*VCELL
    a_seed=1e-4/np.sqrt(N)
    # a_th ~ sqrt(T)  ->  T_eq = 300 * (a_seed/a_th(300))^2
    Teq=300*(a_seed/a_th(q/2,300,B0,MS,AEX,V))**2
    print("  NX=%4d NY=%d : a_seed=%.3e   a_th(q/2,300K)=%.3e   ratio=%.2e   T_equiv=%.3e K"
          % (NX,NY,a_seed,a_th(q/2,300,B0,MS,AEX,V),a_seed/a_th(q/2,300,B0,MS,AEX,V),Teq))
    print("        -> the seed is ln(a_th/a_seed) = %.2f e-foldings BELOW the 300 K bath"
          % np.log(a_th(q/2,300,B0,MS,AEX,V)/a_seed))

print("\n--- band-limited y-uniform seed: per-cell sigma for a given per-mode amplitude ---")
print("Seed = sum over 0<|k|<=k_cut of a*cos(kx+phi_k), uniform in y,z (k_y=k_z=0).")
print("n_modes(+k only) = floor(k_cut/dk);  per-cell variance = n_modes * a^2/2 per component.")
for lbl,NX,dx in (("n=12 1400x8",1400,5e-9),("n=18 2100x8",2100,5e-9)):
    L=NX*dx; dk=2*np.pi/L; V=NX*8*VCELL
    for kc_lab,kc in (("2q",2*q),("1.5q",1.5*q)):
        nm=int(np.floor(kc/dk)); a=a_th(q/2,300,B0,MS,AEX,V)
        sig=a*np.sqrt(nm/2)
        print("  %-12s k_cut=%-5s n_modes=%3d  a=a_th(300K)=%.3e -> per-cell sigma=%.4f  (max canting ~%.3f)"
              % (lbl,kc_lab,nm,a,sig,sig*3.5))

print("\n--- linear window: how long before a seed of amplitude a reaches m_y ~ 0.1 ---")
Aa,Bb = 3.2850e5,-19.433   # Gamma[1/us] = Aa*eps0 + Bb  (fit to the three audit-measured Gamma)
print("Gamma[1/us] = %.4e*eps_0 %+0.3f  (fit to sim48 measured Gamma at 1e-4/2e-4/3e-4)" % (Aa,Bb))
print("%10s %10s %12s %12s %12s" % ("eps_0","G[1/us]","a=1.4e-2","a=1.4e-4","a=9.4e-7"))
for e in (7e-5,1e-4,2e-4,3e-4):
    g=(Aa*e+Bb)*1e6
    row=["%10.0e"%e,"%10.2f"%(g/1e6)]
    for a in (1.4e-2,1.4e-4,9.4e-7):
        row.append("%10.0f ns"%(np.log(0.1/a)/g*1e9) if g>0 else "%12s"%"decays")
    print(" ".join(row))
