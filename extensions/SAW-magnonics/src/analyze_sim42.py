"""Analyze sim42: realistic thermal ENSEMBLE — band + reversal + genuine-ensemble C(k1).

For each config (YIG_fwd, YIG_rev, CoFeB_fwd), across independent thermal seeds:
  - ensemble-mean S(k, f_K) with spread -> finite-momentum band at direction*k_SAW/2
  - forward/backward asymmetry with seed error bars
  - GENUINE-ensemble pair coherence
        C(k1) = |<M_s(k1) M_s(kpump-k1)>_s| / sqrt(<|M_s(k1)|^2><|M_s(kpump-k1)|^2>)
    where the ensemble members s are INDEPENDENT thermal realizations (not Welch
    segments of one run) -> fixes audit issue #2. Incoherent baseline = 1/sqrt(N).

Figure: (a) YIG mean S(k): +k_SAW/2 (fwd) mirrors to -k_SAW/2 (rev);
        (b) CoFeB mean S(k) band at its own +k_SAW/2;
        (c) ensemble C(k1) for YIG_fwd vs 1/sqrt(N) baseline.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
CKPT = os.path.join(DATA, "sim42_checkpoints")
FIGS = os.path.join(ROOT, "figures")
OUT = os.path.join(DATA, "sim42_observables.npz")

CONFIGS = [('YIG_fwd', +1, 5), ('YIG_rev', -1, 3), ('CoFeB_fwd', +1, 3)]
WIN_NS = 20.0


def load_seeds(label, n):
    out = []
    for s in range(n):
        cp = os.path.join(CKPT, f"sim42_{label}_seed{s}.npz")
        if os.path.isfile(cp) and bool(np.load(cp)['done']):
            out.append(np.load(cp))
    return out


def fk_complex(my_xt, CX, DT_REC, F_K, i0, i1):
    """Complex M_y(k, f_K) over window [i0,i1); returns kd (sorted, display sign) and complex row."""
    sub = my_xt[i0:i1].astype(float)
    sub -= sub.mean(axis=0, keepdims=True)
    win = np.hanning(sub.shape[0])[:, None]
    F = np.fft.fftshift(np.fft.fft2(sub * win), axes=(0, 1))
    fax = np.fft.fftshift(np.fft.fftfreq(sub.shape[0], d=DT_REC))
    kax = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(my_xt.shape[1], d=CX))
    kd = -kax
    order = np.argsort(kd)
    ifk = int(np.argmin(np.abs(fax - F_K)))
    return kd[order], F[ifk][order]


def pick_window(seeds, direction, k_saw, CX, DT_REC, F_K):
    """Linear window: sliding WIN_NS window maximizing the band power at the
    EXPECTED centre (direction*k_SAW/2), averaged over seeds."""
    NT = seeds[0]['my_xt'].shape[0]
    w = int(round(WIN_NS * 1e-9 / DT_REC))
    kc = direction * k_saw / 2
    best = (0, min(w, NT), -1.0)
    for a in range(0, max(1, NT - w), max(1, w // 4)):
        b = a + w
        if b > NT:
            break
        p = 0.0
        for d in seeds:
            kd, M = fk_complex(d['my_xt'], CX, DT_REC, F_K, a, b)
            p += (np.abs(M)[np.abs(kd - kc) < 1.0e6] ** 2).sum()
        if p > best[2]:
            best = (a, b, p)
    return best[0], best[1]


def main():
    res = {}
    for label, direction, n in CONFIGS:
        seeds = load_seeds(label, n)
        if not seeds:
            print(f"  [missing] {label}")
            continue
        d0 = seeds[0]
        CX, DT_REC, F_K = float(d0['CX']), float(d0['DT_REC']), float(d0['f_K'])
        k_saw = float(d0['k_saw'])
        i0, i1 = pick_window(seeds, direction, k_saw, CX, DT_REC, F_K)

        Ms = []       # complex M_y(k,f_K) per seed
        Ss = []       # power spectra per seed
        for d in seeds:
            kd, M = fk_complex(d['my_xt'], CX, DT_REC, F_K, i0, i1)
            Ms.append(M)
            Ss.append(np.abs(M) ** 2)
        kd = kd
        Ss = np.array(Ss)               # (N, NX)
        Smean = Ss.mean(0)
        Sstd = Ss.std(0)
        kc = direction * k_saw / 2
        bp = Smean[np.abs(kd - kc) < 1.0e6].sum()
        bm = Smean[np.abs(kd + kc) < 1.0e6].sum()

        # genuine-ensemble pair coherence at k1 = +k_SAW/2 partner pairing to k_pump
        kpump = direction * k_saw
        Marr = np.array(Ms)             # (N, NX)
        j2 = np.array([np.argmin(np.abs(kd - (kpump - kk))) for kk in kd])
        num = np.abs((Marr * Marr[:, j2]).mean(0))
        den = np.sqrt((np.abs(Marr) ** 2).mean(0) * (np.abs(Marr[:, j2]) ** 2).mean(0)) + 1e-30
        C = num / den
        floor = 1.0 / np.sqrt(len(seeds))
        jc = int(np.argmin(np.abs(kd - kc)))
        res[label] = dict(kd=kd, Smean=Smean, Sstd=Sstd, C=C, floor=floor,
                          kc=kc, k_saw=k_saw, bp=bp, bm=bm, n=len(seeds),
                          twin=(i0 * DT_REC * 1e9, i1 * DT_REC * 1e9),
                          Cpeak=float(C[jc]))
        print(f"  {label} (N={len(seeds)}, win {res[label]['twin'][0]:.0f}-"
              f"{res[label]['twin'][1]:.0f}ns): band@{kc/1e6:+.2f}/um "
              f"fwd/bwd={bp/max(bm,1e-30):.1f}x  C({kc/1e6:+.1f})={res[label]['Cpeak']:.2f} "
              f"(1/sqrtN={floor:.2f})")

    if not res:
        print("no data")
        return
    np.savez(OUT, **{f"{k}__{fld}": res[k][fld]
                     for k in res for fld in ('kd', 'Smean', 'Sstd', 'C')},
             labels=np.array(list(res.keys())))

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7))
    # (a) YIG fwd vs rev mirror
    ax = axes[0]
    for label, c, lab in [('YIG_fwd', '#D55E00', r'$+\mathbf{k}_\mathrm{SAW}$'),
                          ('YIG_rev', '#0072B2', r'$-\mathbf{k}_\mathrm{SAW}$')]:
        if label not in res:
            continue
        r = res[label]
        n = r['Smean'] / r['Smean'].max()
        ax.plot(r['kd'] / 1e6, n, color=c, lw=1.3, label=lab)
    if 'YIG_fwd' in res:
        kh = res['YIG_fwd']['k_saw'] / 2e6
        for xx, cc in [(kh, '#D55E00'), (-kh, '#0072B2')]:
            ax.axvline(xx, color=cc, ls='--', lw=0.6, alpha=0.6)
    ax.set_xlim(-12, 12); ax.set_xlabel(r'$k\ (\mu\mathrm{m}^{-1})$')
    ax.set_ylabel(r'ensemble $\langle S(k,f_K)\rangle$ (norm.)')
    ax.set_title('Realistic YIG: direction reversal', fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.text(-0.18, 1.03, '(a)', transform=ax.transAxes, fontweight='bold', fontsize=11)

    # (b) CoFeB band
    ax = axes[1]
    if 'CoFeB_fwd' in res:
        r = res['CoFeB_fwd']
        ax.plot(r['kd'] / 1e6, r['Smean'] / r['Smean'].max(), color='#009E73', lw=1.3)
        ax.axvline(r['k_saw'] / 2e6, color='#009E73', ls='--', lw=0.6, alpha=0.6)
        ax.text(r['k_saw'] / 2e6, 1.02, r'$+k_\mathrm{SAW}/2$', ha='center', fontsize=7)
    ax.set_xlim(-25, 25); ax.set_xlabel(r'$k\ (\mu\mathrm{m}^{-1})$')
    ax.set_ylabel(r'$\langle S(k,f_K)\rangle$ (norm.)')
    ax.set_title('Realistic CoFeB', fontsize=8)
    ax.text(-0.18, 1.03, '(b)', transform=ax.transAxes, fontweight='bold', fontsize=11)

    # (c) ensemble coherence
    ax = axes[2]
    if 'YIG_fwd' in res:
        r = res['YIG_fwd']
        ax.plot(r['kd'] / 1e6, r['C'], color='#D55E00', lw=1.2, label='YIG ensemble $C(k_1)$')
        ax.axhline(r['floor'], color='k', ls=':', lw=0.8, label=r'$1/\sqrt{N}$ baseline')
        ax.axvline(r['kc'] / 1e6, color='#D55E00', ls='--', lw=0.6, alpha=0.6)
    ax.set_xlim(-12, 12); ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$k_1\ (\mu\mathrm{m}^{-1})$')
    ax.set_ylabel(r'ensemble pair coherence $C(k_1)$')
    ax.set_title('Genuine-ensemble coherence', fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc='upper right')
    ax.text(-0.18, 1.03, '(c)', transform=ax.transAxes, fontweight='bold', fontsize=11)

    fig.tight_layout()
    out = os.path.join(FIGS, 'fig_sim42_ensemble.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"saved {out}")


if __name__ == "__main__":
    main()
