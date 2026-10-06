"""Figures and tables for revision 5.
Outputs (directory set by the environment variable CAC_OUT, default 'output'):
figs/fig_running.pdf, figS3_kround.pdf, figS4_stopping.pdf, figS5_multihop.pdf,
figS6_hetero.pdf; tab_sparse_power.tex, tab_sparse_cost.tex, tab_geo.tex, tab_mhglob.tex."""
import numpy as np
import sys
sys.path.insert(0, 'sim')
from capfirst import caption_first
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = 'sim/results/'
import os
OUTDIR = os.environ.get('CAC_OUT', 'output')
OUT = OUTDIR + '/figs/'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'lines.markersize': 4})


def se(x, axis=0):
    return x.std(axis=axis, ddof=1) / np.sqrt(x.shape[axis])


def fmt(mean, s, digits=3):
    """0.542(3): SE in units of the last digit."""
    unit = 10 ** (-digits)
    k = int(round(s / unit))
    return f'{mean:.{digits}f}({max(k, 0)})' if k < 10 else f'{mean:.{digits}f}({k})'


def fmt_signed(mean, s, digits=3):
    unit = 10 ** (-digits); k = int(round(s / unit))
    return f'{mean:+.{digits}f}({k})'.replace('+', '$+$').replace('-', '$-$')


# ------------------------------------------------------------ running example
def fig_running(fname='fig_running.pdf', h=2.45):
    """Three panels: (a) round-1 grid, zoom trapped in [0, alpha/3]; (b) outside query at 0.1,
    no redirection (output and window unchanged); (c) same query with redirection."""
    a = 0.2; p = np.array([0.081, 0.083, 0.085, 0.087, 0.089, 0.25, 0.3, 0.4, 0.5, 0.6]); m = len(p)
    F = lambda t: np.mean(p <= t)
    t = np.linspace(0, a, 2001)
    Z1 = [0, a / 3, 2 * a / 3, a]; Z2 = [0, a / 3, 0.1, 2 * a / 3, a]
    panels = [(Z1, None, False, '(a) round-1 grid, window $[0,0.0667]$'),
              (Z2, 0.1, False, '(b) outside query at $0.1$, no redirection'),
              (Z2, 0.1, True, '(c) outside query at $0.1$, redirection')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, h), sharey=True)
    for ax, (Z, q, redirect, title) in zip(axes, panels):
        Z = np.array(Z); FZ = np.array([F(z) for z in Z])
        floor = np.array([FZ[np.searchsorted(Z, x, side='right') - 1] for x in t])
        ceil = np.array([FZ[min(np.searchsorted(Z, x, side='left'), len(Z) - 1)] for x in t])
        # output threshold: zoom chain (round-1 crossing on Z1) or largest known crossing (redirection)
        Zc = Z if redirect else np.array(Z1); FZc = np.array([F(z) for z in Zc])
        cross = [z for z, f in zip(Zc, FZc) if f >= z / a - 1e-12]; zI = max(cross); th = a * F(zI)
        # window: known bin starting at zI
        nxt = min(z for z in Z if z > zI + 1e-12); L, U = zI, nxt
        ax.axvspan(L, U, color='0.85', lw=0, label='search window $[L,U]$')
        ax.fill_between(t, floor, ceil, color='C0', alpha=0.15, lw=0, label='floor--ceiling band')
        ax.plot(t, [F(x) for x in t], color='k', lw=0.8, label='pooled $F$ (for explanation; unknown to the center)')
        ax.plot(t, floor, color='C0', lw=1.2, label='floor $\\hat F$')
        ax.plot(t, ceil, color='C0', lw=1.2, ls='--', label='ceiling $\\bar F$')
        ax.plot(t, t / a, color='C3', lw=1.0, label='BH line $t/\\alpha$')
        ax.plot(Z, FZ, 'o', color='C0', ms=4, label='queried counts')
        if q is not None:
            ax.plot([q], [F(q)], 's', color='C1', ms=5, mfc='none', mew=1.2, label='outside query')
        tb = th
        for b in range(1, len(Z)):
            if a * FZ[b] > Z[b - 1] + 1e-12:
                tb = max(tb, min(Z[b], a * FZ[b]))
        ax.axvline(th, color='C2', lw=0.9, ls=':'); ax.axvline(tb, color='C1', lw=0.9, ls=':')
        ax.text(th + 0.003, 0.93, '$\\hat\\tau=%.2g$' % th, color='C2', fontsize=7)
        ax.text(tb + 0.003, 0.80, '$\\bar\\tau=%.2g$' % tb, color='C1', fontsize=7)
        mC = (m / a) * (tb - th)
        ax.text(0.125, 0.30, '$mC=%d$' % round(mC), fontsize=7.5)
        ax.set_xlim(0, a); ax.set_ylim(0, 1.0); ax.set_xlabel('$t$'); ax.set_title(title, fontsize=7.5)
    axes[0].set_ylabel('CDF value')
    hh, ll = axes[2].get_legend_handles_labels()
    fig.legend(hh, ll, loc='lower center', ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14 * 2.45 / h, 1, 1)); fig.savefig(OUT + fname); plt.close(fig)


# ------------------------------------------------------------ sparse tables
S1_LABELS = {'BH': 'Pooled BH', 'uniform12': 'Uniform, $M=12$', 'geo12': 'Geometric, $M=12$',
             'twopass77': 'Two-pass $(7,7)$', 'zoom4x5': 'Zoom, $M=4$, $K=5$', 'bisect3x10': 'Bisection, $K=10$',
             'hybrid77': 'Geometric $7$ + refine $7$', 'cert_fixed': 'Certified, no redirection',
             'cert_fixed_geo': 'Certified, no redir., geo.\\ start', 'cert_redirect_abstain': 'Certified, redirect + abstain',
             'cert_fixed_levelcorr': 'Certified, no redir., level $\\alpha/(1{+}\\varepsilon)$'}
PIS = (0.3, 0.1, 0.03, 0.01)


def tab_sparse():
    D = [np.load(R + f'expS1_pi{pi}.npz', allow_pickle=True) for pi in PIS]
    methods = list(D[0]['methods'])
    # power table
    rows = []
    for c, name in enumerate(methods):
        cells = []
        for d in D:
            o = d['out']; pw = o[:, c, 0]; diff = pw - o[:, 0, 0]
            if name == 'BH':
                cells += [fmt(pw.mean(), se(pw)), '---']
            else:
                cells += [fmt(pw.mean(), se(pw)), fmt_signed(diff.mean(), se(diff))]
        rows.append(S1_LABELS[name] + ' & ' + ' & '.join(cells) + r' \\')
        if name in ('BH', 'hybrid77'):
            rows.append(r'\midrule')
    tau = ' & '.join(r'\multicolumn{2}{c}{%.4f}' % d['tau_bh'].mean() for d in D)
    tex = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{3.5pt}
\footnotesize
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{2}{c}{$\pi_1=0.3$} & \multicolumn{2}{c}{$\pi_1=0.1$} & \multicolumn{2}{c}{$\pi_1=0.03$} & \multicolumn{2}{c}{$\pi_1=0.01$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
 & power & $-$BH & power & $-$BH & power & $-$BH & power & $-$BH \\
\midrule
Mean $\tau_{\mathrm{BH}}$ & """ + tau + r""" \\
\midrule
""" + '\n'.join(rows[:-1] if rows[-1] == r'\midrule' else rows) + r"""
\bottomrule
\end{tabular}
\caption{Sparse regime, power (400 trials per column). Model: $N=300$ nodes,
$m^{(i)}=\max(1,\mathrm{Pois}(10))$, so $m\approx 3000$; node $i$'s non-null
means are drawn uniformly from $\mu^{(i)}\pm 0.5$ with $\mu^{(i)}=2.5+i/N$;
two-sided $z$-tests; $\alpha=0.2$. ``$-$BH'' is the paired difference to
pooled BH on the same data. Parentheses give Monte Carlo standard errors in
units of the last digit. Fixed-budget methods (middle block) request exactly
$12$ counts per node with endpoint reuse and no outside queries
($\rho=0$); certified methods (bottom block)
use $\varepsilon=0.2$, $M=4$ per round, $K_{\max}=8$, $\rho=1$. Costs and
certification are in Table~\ref{tab:sparse-cost}.}
\label{tab:sparse-power}
\end{table*}
"""
    open(OUTDIR + '/tab_sparse_power.tex', 'w').write(caption_first(tex))
    # cost / certification table
    rows = []
    for c, name in enumerate(methods):
        if name == 'BH':
            continue
        cells = []
        for d in D:
            o = d['out'][:, c]; mu = o.mean(0)
            cells += [f'{mu[4]:.1f}/{mu[5]:.1f}', f'{100*(mu[6]+mu[7]):.0f} ({100*mu[7]:.0f})', f'{100*mu[3]:.0f}']
        rows.append(S1_LABELS[name] + ' & ' + ' & '.join(cells) + r' \\')
        if name == 'hybrid77':
            rows.append(r'\midrule')
    tex = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{2.2pt}
\scriptsize
\begin{tabular}{@{}lcccccccccccc@{}}
\toprule
 & \multicolumn{3}{c}{$\pi_1=0.3$} & \multicolumn{3}{c}{$\pi_1=0.1$} & \multicolumn{3}{c}{$\pi_1=0.03$} & \multicolumn{3}{c}{$\pi_1=0.01$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
 & $S$/exch. & cert. & rec. & $S$/exch. & cert. & rec. & $S$/exch. & cert. & rec. & $S$/exch. & cert. & rec. \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Sparse regime, same runs as Table~\ref{tab:sparse-power}: mean
counts $S$ and exchanges per node; percentage of runs certified at
$\varepsilon=0.2$ by Lemma~\ref{lem:cert} (in parentheses: exactly certified,
$mC<1$); percentage of runs that actually recover pooled BH's rejection
set, evaluated against pooled BH in the simulation (exact certification
identifies a verifiable subset of these runs). For the fixed-budget methods the
certificate is computed from the counts they already transmit, with no
additional queries; it is an audit, not a stopping rule. For the
level-corrected run, certification refers to BH at level $\alpha/(1+\varepsilon)$
and recovery to BH at level $\alpha$.}
\label{tab:sparse-cost}
\end{table*}
"""
    open(OUTDIR + '/tab_sparse_cost.tex', 'w').write(caption_first(tex))


# ------------------------------------------------------------ Table II
def tab_geo():
    Ss = (6, 8, 10, 12, 16, 20)
    rows = []
    for S in Ss:
        o = np.load(R + f'expS2_S{S}.npz', allow_pickle=True)['out']
        pw = o[:, :, 0]
        cells = [fmt(pw[:, c].mean(), se(pw[:, c])) for c in (1, 2, 3, 4, 5, 0)]
        rows.append(f'{S} & ' + ' & '.join(cells) + r' \\')
    tex = r"""\begin{table}[t]
\centering
\setlength{\tabcolsep}{2.5pt}
\footnotesize
\begin{tabular}{@{}lcccccc@{}}
\toprule
$S$ & Unif. & Geo.\ $\bar m$ & Geo.\ $\sqrt{\bar m}$ & Two-pass & Geo.+ref. & BH \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\caption{Weak-signal star network: power at $S$ counts per node with
endpoint reuse and no outside queries (2000 trials; standard errors in
units of the last digit).
Geometric grids~\eqref{eq:geogrid} with $S$ points and floor $\alpha/c$,
$c\in\{\bar m,\sqrt{\bar m}\}$, $\bar m=N\lambda=300$; two-pass: $(M,M)$ with
$M=S/2+1$, i.e.\ $2M-2=S$ counts; hybrid: geometric first grid with $M$
points and floor $\alpha/\sqrt{\bar m}$, then $M$ uniform refinement points.
No compressed method has estimated FDR above $0.137$.}
\label{tab:geo}
\end{table}
"""
    open(OUTDIR + '/tab_geo.tex', 'w').write(caption_first(tex))


# ------------------------------------------------------------ fixed-K zoom
def fig_S3():
    """Column-width figure (3.45 in): power (top), certificate tightness (bottom)."""
    from matplotlib.lines import Line2D
    uni = np.load(R + 'expS3_uniform.npz')['out']
    with plt.rc_context({'font.size': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7}):
        fig, ax = plt.subplots(2, 1, figsize=(3.45, 4.3), sharex=True)
        ax[0].errorbar(uni[:, 0], uni[:, 1], yerr=uni[:, 2], fmt='k--s', ms=3, capsize=1.5, label='uniform single pass')
        bh = None
        for M, mk in zip((3, 4, 5), ('o', '^', 'v')):
            o = np.load(R + f'expS3_M{M}.npz')['out']; mu = o.mean(0); sd = se(o)
            ax[0].errorbar(mu[1:, 4], mu[1:, 0], yerr=sd[1:, 0], marker=mk, ms=3.5, capsize=1.5,
                           color=f'C{M-3}', label=f'zoom, $M={M}$')
            ax[1].plot(mu[1:, 4], mu[1:, 8], marker=mk, ms=3.5, ls='-', color=f'C{M-3}')
            ax[1].plot(mu[1:, 4], mu[1:, 9], marker=mk, ms=3.5, ls='--', color=f'C{M-3}')
            bh = mu[0, 0]
        ax[0].axhline(bh, color='gray', lw=0.8, ls=':', label='pooled BH')
        ax[0].set_ylabel('power'); ax[0].legend(loc='lower right', frameon=False)
        ax[1].set_yscale('log'); ax[1].set_xlabel('counts requested per node $S$')
        ax[1].set_ylabel('missed BH discoveries')
        h = [Line2D([], [], color='k', ls='-'), Line2D([], [], color='k', ls='--')]
        ax[1].legend(h, ['certified bound $mC$', 'actual'], loc='upper right', frameon=False,
                     title='colors as above', title_fontsize=6.5)
        fig.tight_layout(); fig.savefig(OUT + 'figS3_kround.pdf'); plt.close(fig)


# ------------------------------------------------------------ certified stopping (4 panels)
def fig_S4(fname='figS4_stopping.pdf', grid2x2=False):
    V = {v: np.load(R + f'expS4_{v}.npz') for v in ('fixed', 'redirect_abstain', 'fixed_levelcorr')}
    e = V['fixed']['eps_grid']
    if grid2x2:
        fig, ax = plt.subplots(2, 2, figsize=(3.45, 3.65)); ax = ax.ravel()
    else:
        fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1))
    o = V['fixed']['out']; mu = o.mean(0); s = se(o)
    ax[0].plot(e, mu[1:, 4], 'o-', label='counts'); ax[0].plot(e, mu[1:, 5], 's--', label='exchanges')
    ax[0].set_title('(a) communication', fontsize=7.5)
    lab = {'fixed': 'no redirection', 'redirect_abstain': 'redirect + abstain', 'fixed_levelcorr': 'no redir., level $\\alpha/(1{+}\\varepsilon)$'}
    for v, st in zip(V, ('o-', 'd:', 'v-.')):
        oo = V[v]['out']; m_ = oo.mean(0); s_ = se(oo)
        ax[1].errorbar(e, m_[1:, 0], yerr=s_[1:, 0], fmt=st, capsize=1.5, label=lab[v])
    ax[1].axhline(mu[0, 0], color='gray', ls=':', lw=0.8, label='pooled BH')
    ax[1].set_title('(b) power', fontsize=7.5); ax[1].set_ylim(0.08, 0.34)
    ax[2].plot(e, mu[1:, 6] + mu[1:, 7], 'o-', label='certified (rel.+exact)')
    ax[2].plot(e, mu[1:, 7], 's--', label='exactly certified')
    ax[2].plot(e, mu[1:, 3], 'd:', label='actual recovery (simulation benchmark)')
    ax[2].set_title('(c) certification', fontsize=7.5); ax[2].set_ylim(0, 1.05)
    ax[3].plot(e, mu[1:, 8], 'o-', label='certified bound $mC$'); ax[3].plot(e, mu[1:, 9], 's--', label='actual missed')
    ax[3].set_yscale('log'); ax[3].set_title('(d) certificate tightness', fontsize=7.5)
    if grid2x2:
        hh, ll = [], []
        for a_, ttl in zip(ax, ('(a) communication', '(b) power', '(c) certification', '(d) certificate tightness')):
            a_.set_xscale('log'); a_.set_xlabel('$\\varepsilon$', labelpad=1); a_.set_title(ttl, fontsize=7)
            h_, l_ = a_.get_legend_handles_labels(); hh += h_; ll += l_
        ax[1].set_ylim(0.08, 0.34)
        ll = [l_.replace(' (simulation benchmark)', '') for l_ in ll]
        fig.legend(hh, ll, loc='lower center', ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.005),
                   handlelength=1.6, columnspacing=0.8, handletextpad=0.5)
        fig.tight_layout(rect=(0, 0.17, 1, 1), h_pad=0.6, w_pad=0.8); fig.savefig(OUT + fname); plt.close(fig); return
    for a_ in ax:
        a_.set_xscale('log'); a_.set_xlabel('$\\varepsilon$')
        a_.legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), fontsize=6.5, ncol=1, frameon=False)
    fig.set_size_inches(7.2, 2.9); fig.tight_layout(); fig.savefig(OUT + fname); plt.close(fig)


# ------------------------------------------------------------ multihop
def fig_S5(fname='figS5_multihop.pdf', side_by_side=False):
    """Column-width figure (3.45 in): power (top), certificate tightness (bottom).
    side_by_side=True gives a 1x2 variant of the same height as one panel, for the TSP main text."""
    from matplotlib.lines import Line2D
    with plt.rc_context({'font.size': 7.5, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5 if side_by_side else 7,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7}):
        if side_by_side:
            fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.0))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(3.45, 4.3), sharex=True)
        for deg, c in zip((2, 4, 8, 16), ('C0', 'C1', 'C2', 'C3')):
            o = np.load(R + f'expS5_deg{deg}.npz')['out']; mu = o.mean(0); sd = se(o)
            ax[0].errorbar(mu[1:5, 4], mu[1:5, 0], yerr=sd[1:5, 0], fmt='o-', ms=3.5, color=c, capsize=1.5, label=f'$d={deg}$')
            ax[0].axhline(mu[0, 0], color=c, ls=':', lw=0.9)
            ax[1].plot(mu[1:5, 4], mu[1:5, 2], 'o-', ms=3.5, color=c)
            ax[1].plot(mu[1:5, 4], mu[1:5, 3], 's--', ms=3.5, color=c)
        ax[0].set_ylabel('power')
        ax[1].set_yscale('log')
        if side_by_side:
            for a_ in ax: a_.set_xlabel('counts per edge $S$')
            ax[1].set_ylabel('missed QuTE disc.')
            hh, ll = ax[0].get_legend_handles_labels()
            hh += [Line2D([], [], color='k', ls=':', lw=0.9), Line2D([], [], color='k', ls='-', marker='o', ms=3),
                   Line2D([], [], color='k', ls='--', marker='s', ms=3)]
            ll += ['uncompressed QuTE', 'certified $\\sum_x D^{(x)}$', 'actual']
            fig.legend(hh, ll, loc='lower center', ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.01),
                       handlelength=1.8, columnspacing=1.0)
            fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.savefig(OUT + fname); plt.close(fig); return
        else:
            ax[0].legend(loc='lower right', frameon=False, ncol=2,
                         title='dotted: uncompressed QuTE', title_fontsize=6.5)
            ax[1].set_xlabel('counts requested per edge $S$ ($M=4$, $K=1,\\ldots,4$)')
            ax[1].set_ylabel('missed QuTE discoveries')
        h = [Line2D([], [], color='k', ls='-', marker='o', ms=3), Line2D([], [], color='k', ls='--', marker='s', ms=3)]
        ax[1].legend(h, ['certified $\\sum_x D^{(x)}$', 'actual'], loc='upper right', frameon=False,
                     title='colors as above', title_fontsize=6.5)
        fig.tight_layout(); fig.savefig(OUT + fname); plt.close(fig)


# ------------------------------------------------------------ heterogeneous
def fig_S6():
    Bs = (8, 10, 14)
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.3))
    names = ['pooled BH', 'equal split', '$\\sqrt{m^{(i)}}$ rule', 'oracle (data-dependent)']
    for c, (nm, st) in enumerate(zip(names, ('k:', 'o-', 's-', 'd--'))):
        mus = []; ses = []
        for B in Bs:
            o = np.load(R + f'expS6_B{B}.npz')['out']; mus.append(o[:, c, 0].mean()); ses.append(se(o[:, c, 0]))
        ax.errorbar(Bs, mus, yerr=ses, fmt=st, capsize=1.5, label=nm)
    ax.set_xlabel('mean grid points per node per pass'); ax.set_ylabel('power'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT + 'figS6_hetero.pdf'); plt.close(fig)




def tab_mhglob():
    degs = (2, 4, 6, 8, 12, 16, 20)
    O = [np.load(R + f'expS7_deg{d}.npz')['out'] for d in degs]
    row = lambda c: ' & '.join(fmt(o[:, c, 0].mean(), se(o[:, c, 0])) for o in O)
    diff = ' & '.join(fmt_signed((o[:, 2, 0] - o[:, 1, 0]).mean(), se(o[:, 2, 0] - o[:, 1, 0])) for o in O)
    tex = r"""\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4pt}
\footnotesize
\begin{tabular}{@{}lccccccc@{}}
\toprule
Nominal degree $d$ & 2 & 4 & 6 & 8 & 12 & 16 & 20 \\
\midrule
Global grid $[0,\alpha]$, $10$ counts & """ + row(0) + r""" \\
Local grid $[0,\alpha^{(x)}]$, $10$ counts & """ + row(1) + r""" \\
Local, two-pass $(5,5)$, $8$ counts & """ + row(2) + r""" \\
\midrule
Two-pass $-$ local (paired) & """ + diff + r""" \\
\bottomrule
\end{tabular}
\caption{Multihop power versus nominal degree $d$ ($N=50$, $\lambda=3$,
1000 trials; graph model of Fig.~\ref{fig:S5}, realized average degrees
$2.0$, $4.2$, $6.0$, $8.6$, $12.6$, $14.2$ and $19.4$). The global grid is
the multihop procedure of~\cite{pournaderi2022isit}; counts are per edge
with endpoint reuse. Standard errors in units of the last digit. All
three control FDR (at most $0.092$).}
\label{tab:mhglob}
\end{table*}
"""
    open(OUTDIR + '/tab_mhglob.tex', 'w').write(caption_first(tex))


if __name__ == '__main__':
    fig_running(); fig_running('fig_running_tsp.pdf', 2.0); tab_sparse(); tab_geo(); fig_S3(); fig_S4(); fig_S4('figS4_stopping_tsp.pdf', True); fig_S5(); fig_S5('figS5_multihop_tsp.pdf', True); fig_S6(); tab_mhglob(); print('done')
