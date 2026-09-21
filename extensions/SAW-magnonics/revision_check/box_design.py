"""Box-design arithmetic for the commensurate GPU campaign.  CPU only.

Everything printed here is derived from first principles in this file; no
number is copied from the manuscript.  Run:  python box_design.py
"""
import numpy as np

V_SAW = 3500.0          # m/s   (src/sim40_eps_kresolved.py: V_SAW = 3500.0)
F_SAW = 6.0e9           # Hz    (sim40 'full_2fK' = 2 * F_REF, F_REF = 3.0e9)
MU0   = 4e-7*np.pi
GAMMA = 1.76e11

lam = V_SAW / F_SAW
q   = 2*np.pi*F_SAW / V_SAW
print("lambda_SAW = %.6f nm" % (lam*1e9))
print("q          = %.6f rad/um" % (q/1e6))

# --- the incommensurate box actually used (sim40) ---
L40 = 1024*5e-9
print("\nsim40 box: NX=1024 dx=5nm  L=%.3f um  dk=%.7f /um  q/dk=%.4f (NOT integer)"
      % (L40*1e6, 2*np.pi/L40/1e6, q/(2*np.pi/L40)))
k4, k5 = 4*2*np.pi/L40, 5*2*np.pi/L40
print("  bin4=%.6f  bin5=%.6f  sum=%.6f = 9*dk ; q=%.6f ; diff=%.6f (%.2f%%)"
      % (k4/1e6, k5/1e6, (k4+k5)/1e6, q/1e6, (k4+k5-q)/1e6, 100*(k4+k5-q)/q))

# --- n-period boxes ---
print("\n%3s %12s %12s %10s %12s %10s %10s" %
      ("n","L (um)","NX@dx=5nm","integer?","dk (/um)","q bin","q/2 bin"))
for n in (6,8,10,12,16,18,24):
    L  = n*lam
    nx = L/5e-9
    dk = 2*np.pi/L
    print("%3d %12.6f %12.4f %10s %12.7f %10s %10s" %
          (n, L*1e6, nx, "YES" if abs(nx-round(nx))<1e-9 else "no",
           dk/1e6, "%d"%round(q/dk), ("%d"%(n//2)) if n%2==0 else "n/2 NOT integer"))

print("\nlambda/dx at dx=5nm = %.6f cells  -> NX=n*lambda/dx integer iff n %% 3 == 0"
      % (lam/5e-9))
print("even AND multiple of 3  ->  n in {6,12,18,24,...}")

# alternative cell size that makes every n commensurate
dx2 = lam/125
print("\nalternative dx = lambda/125 = %.6f nm -> NX = 125*n for every n" % (dx2*1e9))
for n in (8,10,12,16):
    print("   n=%2d  L=%.6f um  NX=%d" % (n, n*lam*1e6, 125*n))

# --- exchange length ---
for name,A,Ms in (("YIG(sim40/43)",3.65e-12,140e3), ("CoFeB(sim43)",19e-12,1.0e6)):
    lex = np.sqrt(2*A/(MU0*Ms**2))
    print("\nl_ex(%s) = sqrt(2A/mu0 Ms^2) = %.3f nm   dx=5nm -> dx/l_ex = %.3f"
          % (name, lex*1e9, 5e-9/lex))

# --- magnon wavelength at q/2, and Nyquist ---
print("\nlambda_magnon(q/2) = %.4f um = %.1f cells at 5 nm" % (2*np.pi/(q/2)*1e6, 2*np.pi/(q/2)/5e-9))
print("k_Nyquist = pi/dx = %.1f rad/um  (= %.1f x q)" % (np.pi/5e-9/1e6, (np.pi/5e-9)/q))

# --- LLG stability: fastest exchange mode at k_Nyquist ---
for name,A,Ms in (("YIG",3.65e-12,140e3), ("CoFeB",19e-12,1.0e6)):
    kN = np.pi/5e-9
    Bex = 2*A*kN**2/Ms
    w   = GAMMA*Bex
    print("%-6s B_ex(k_Ny)=%.2f T  omega=%.3e rad/s  T=%.4f ps  dt=0.2ps -> dt/T=%.3f"
          % (name, Bex, w, 2*np.pi/w*1e12, 0.2/(2*np.pi/w*1e12)))

# --- commensurate frequencies available on a FIXED box ---
print("\nOn a FIXED box, f_SAW must satisfy q L = 2 pi n  ->  f_n = n*v/L:")
for L,nx in ((7.0e-6,1400),(14.0e-6,2800),(3.5e-6,700)):
    print("  L=%5.2f um (NX=%4d): f_1 = %.4f GHz ; EVEN-n steps = %.4f GHz ; n(6GHz)=%.3f"
          % (L*1e6, nx, V_SAW/L/1e9, 2*V_SAW/L/1e9, F_SAW*L/V_SAW))
print("  => a fine f_SAW scan is INCOMPATIBLE with a fixed commensurate box.")
print("     Detuning must be swept with B0 (Kittel), which does not touch the box.")

# --- growth-rate model anchored on the audit's MEASURED Gamma ---
print("\n--- Gamma(eps_0) from the three audit-reported measured values ---")
eps = np.array([1e-4,2e-4,3e-4]); G = np.array([13.6,45.9,79.3])   # per us, sim48 f6.10GHz
Aa,Bb = np.polyfit(eps,G,1)
print("  linear fit  Gamma[1/us] = %.4e * eps_0 %+0.3f ; predicted at 2e-4 = %.2f (meas 45.9)"
      % (Aa,Bb, Aa*2e-4+Bb))
print("  threshold eps_th = %.3e ;  alpha*omega_K (alpha=5e-4,f=3.05GHz) = %.2f per us"
      % (-Bb/Aa, 5e-4*2*np.pi*3.05e9/1e6))
seed=1e-4
for NX,NY in ((1024,8),(1400,8)):
    print("  per-mode seed amp for sigma=1e-4, NX=%d NY=%d : %.3e" % (NX,NY,seed/np.sqrt(NX*NY)))
s0 = seed/np.sqrt(1400*8)
print("\n  e-foldings and T from the fit (NX=1400, per-mode seed %.2e):" % s0)
print("  %8s %10s %12s %12s %12s" % ("eps_0","G[1/us]","T(Gt=1.5)","T->1e-3","T->5e-4 max"))
for e in (5e-5,7e-5,1e-4,2e-4,3e-4,5e-4):
    g = (Aa*e+Bb)*1e6
    if g<=0:
        print("  %8.0e %10.2f %12s %12s %12s" % (e,(Aa*e+Bb),"decays","decays","decays")); continue
    print("  %8.0e %10.2f %10.0f ns %10.0f ns %10.0f ns"
          % (e,(Aa*e+Bb), 1.5/g*1e9, np.log(1e-3/s0)/g*1e9, np.log(5e-4/s0)/g*1e9))

# --- MEL drive field and the one measured anchor at toy B1 ---
print("\n--- b_mel = 2|B1| eps_0 / Ms  (Tesla; from chiralsawfield.cu H = -2 b1 eps m/Ms) ---")
for nm,B1,Ms,e in (("sim40 toy B1=8.8MJ",8.8e6,140e3,7e-5),("lit YIG B1=0.35MJ",0.35e6,140e3,1e-4)):
    b = 2*B1*e/Ms
    print("  %-22s eps=%.0e -> b=%.4e T ; gamma*b/4 = %.3e 1/s" % (nm,e,b,GAMMA*b/4))
print("  sim40 measured Gamma(bin4)=3.057e8, Gamma(bin5)=2.992e8 (CONVENTIONS.md sec.7)")
print("  ratio measured/(gamma*b/4) = %.3f" % (3.02e8/(GAMMA*(2*8.8e6*7e-5/140e3)/4)))

# --- wall clock from the run logs ---
print("\n--- throughput from data/sim4*_run.log (NX=1024, NY=8, dt_step=0.2ps) ---")
obs = [("sim41 T=0",100,27.7),("sim42",50,14.0),("sim44",40,10.9),
       ("sim46",40,14.4),("sim47",40,14.3),("sim48",40,14.4),("sim45 noSAW",20,6.5)]
r=[]
for nm,T,mn in obs:
    r.append(mn/T); print("  %-12s %4d ns -> %5.1f min = %.3f min/ns" % (nm,T,mn,mn/T))
r=np.array(r)
print("  range %.3f - %.3f min/ns ; median %.3f" % (r.min(),r.max(),np.median(r)))
print("  scaled to NX=1400 (linear in NX): %.3f - %.3f min/ns"
      % (r.min()*1400/1024, r.max()*1400/1024))
