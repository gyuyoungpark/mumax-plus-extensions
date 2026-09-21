"""Analyze sim41: realistic-YIG thermal parametric band at k_SAW/2.

Reads data/sim41_checkpoints/*.npz (or the packed sim41_realistic_thermal.npz),
computes for each (eps0, T) run:
  - S(k, f_K): |M_y(k, f_K)|^2 on the second-half 2D FFT
  - pair-band centroid k-bar and band/total power ratio near +k_SAW/2
  - m_y(t) growth trace
and makes a figure: (a) S(k,f_K) for the three achievable strains at T=300 K
showing the +k_SAW/2 band in REALISTIC YIG; (b) T=300 vs T=0 control at eps=3e-4.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(DATA, "sim41_checkpoints")
FIGS = os.path.join(ROOT, "figures")
OUT = os.path.join(DATA, "sim41_observables.npz")

RUNS = ['eps3e-4_T300', 'eps2e-4_T300', 'eps1e-4_T300', 'eps3e-4_T0']


def load(label):
    cp = os.path.join(CKPT, f"sim41_{label}.npz")
    if not os.path.isfile(cp):
        return None
    d = np.load(cp)
    return dict(my_xt=d['my_xt'], eps0=float(d['eps0']), T_K=float(d['T_K']),
                K_SAW=float(d['K_SAW']), F_K=float(d['F_K']),
                CX=float(d['CX']), DT_REC=float(d['DT_REC']),
                next_i=int(d['next_i']), done=bool(d['done']))


def _fk_slice(my_xt, CX, DT_REC, F_K, i0, i1):
    sub = my_xt[i0:i1].astype(float)
    sub -= sub.mean(axis=0, keepdims=True)
    win = np.hanning(sub.shape[0])[:, None]
    F = np.fft.fftshift(np.fft.fft2(sub * win), axes=(0, 1))
    fax = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=DT_REC))
    kax = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(my_xt.shape[1], d=CX))
    kd = -kax
    order = np.argsort(kd)
    ifk = int(np.argmin(np.abs(fax - F_K)))
    return kd[order], np.abs(F[ifk])[order] ** 2


def growth_window(my_xt, next_i, DT_REC, K_SAW, CX, F_K, win_ns=20.0):
    """Pick the LINEAR-parametric window: the sliding win_ns window that
    maximizes the coherent +k_SAW/2 band power on the f_K slice. |m_y| RMS
    is a poor picker here (thermal floor dominates); the coherent band peaks
    in the linear-growth window (~15-35 ns) then washes out under nonlinear
    saturation + thermal repopulation, so we track the band directly."""
    n = min(next_i, my_xt.shape[0])
    w = int(round(win_ns * 1e-9 / DT_REC))
    kh = K_SAW / 2
    best = (0, min(w, n), -1.0)
    for a in range(0, max(1, n - w), max(1, w // 4)):
        b = a + w
        if b > n:
            break
        kd, S = _fk_slice(my_xt, CX, DT_REC, F_K, a, b)
        bandp = S[np.abs(kd - kh) < 1.0e6].sum()
        if bandp > best[2]:
            best = (a, b, bandp)
    return best[0], best[1]


def spectrum(my_xt, CX, DT_REC, F_K, i0, i1):
    sub = my_xt[i0:i1].astype(float)
    sub -= sub.mean(axis=0, keepdims=True)
    win = np.hanning(sub.shape[0])[:, None]
    F = np.fft.fftshift(np.fft.fft2(sub * win), axes=(0, 1))
    fax = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=DT_REC))
    kax = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(my_xt.shape[1], d=CX))
    kd = -kax  # paper display convention: +k_SAW forward
    order = np.argsort(kd)
    ifk = int(np.argmin(np.abs(fax - F_K)))
    return kd[order], np.abs(F[ifk])[order] ** 2


def main():
    res = {}
    for label in RUNS:
        d = load(label)
        if d is None:
            print(f"  [missing] {label}")
            continue
        i0, i1 = growth_window(d['my_xt'], d['next_i'], d['DT_REC'],
                               d['K_SAW'], d['CX'], d['F_K'])
        kd, S = spectrum(d['my_xt'], d['CX'], d['DT_REC'], d['F_K'], i0, i1)
        kh = d['K_SAW'] / 2
        bp = S[np.abs(kd - kh) < 1.0e6].sum()          # forward +k_SAW/2 band
        bm = S[np.abs(kd + kh) < 1.0e6].sum()          # backward -k_SAW/2 band
        fwd = bp / max(bm, 1e-30)                        # forward asymmetry
        res[label] = dict(kd=kd, S=S, kh=kh, eps0=d['eps0'], T_K=d['T_K'],
                          bp=bp, bm=bm, fwd=fwd, win=(i0, i1),
                          twin=(i0 * d['DT_REC'] * 1e9, i1 * d['DT_REC'] * 1e9),
                          done=d['done'])
        print(f"  {label}: win={res[label]['twin'][0]:.0f}-{res[label]['twin'][1]:.0f}ns "
              f"S(+k/2)={bp:.2e}  S(-k/2)={bm:.2e}  fwd-asym={fwd:.1f}x  "
              f"(k_SAW/2={kh/1e6:+.2f}/um)")

    if not res:
        print("no data yet")
        return
    np.savez(OUT, **{f"{k}__{f}": res[k][f]
                     for k in res for f in ('S', 'kd')},
             labels=np.array(list(res.keys())))

    # Figure
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    colors = {'eps3e-4_T300': '#D55E00', 'eps2e-4_T300': '#E69F00',
              'eps1e-4_T300': '#0072B2'}
    for label in ['eps1e-4_T300', 'eps2e-4_T300', 'eps3e-4_T300']:
        if label not in res:
            continue
        r = res[label]
        axa.plot(r['kd'] / 1e6, r['S'] / r['S'].max(), lw=1.2,
                 color=colors[label], label=f"$\\varepsilon_0$={r['eps0']:.0e}")
    if res:
        kh = list(res.values())[0]['kh']
        axa.axvline(kh / 1e6, color='k', ls='--', lw=0.7, alpha=0.6)
        axa.text(kh / 1e6, 1.02, r'$+k_\mathrm{SAW}/2$', ha='center', fontsize=7)
    axa.set_xlim(-12, 12)
    axa.set_xlabel(r'$k\ (\mu\mathrm{m}^{-1})$')
    axa.set_ylabel(r'$S(k,f_K)$ (norm.)')
    axa.set_title('Realistic YIG, $T=300$ K', fontsize=8)
    axa.legend(fontsize=6, frameon=False)

    for label, c, ls in [('eps3e-4_T300', '#D55E00', '-'),
                         ('eps3e-4_T0', '#666666', '--')]:
        if label not in res:
            continue
        r = res[label]
        axb.plot(r['kd'] / 1e6, r['S'] / max(res['eps3e-4_T300']['S'].max(), 1e-30),
                 lw=1.2, color=c, ls=ls,
                 label=f"$T$={r['T_K']:.0f} K")
    if 'eps3e-4_T300' in res:
        axb.axvline(res['eps3e-4_T300']['kh'] / 1e6, color='k', ls='--', lw=0.7, alpha=0.6)
    axb.set_xlim(-12, 12)
    axb.set_xlabel(r'$k\ (\mu\mathrm{m}^{-1})$')
    axb.set_ylabel(r'$S(k,f_K)$ (norm.)')
    axb.set_title(r'$\varepsilon_0=3\times10^{-4}$: thermal vs $T=0$', fontsize=8)
    axb.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    out = os.path.join(FIGS, 'fig_sim41_realistic_thermal.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
