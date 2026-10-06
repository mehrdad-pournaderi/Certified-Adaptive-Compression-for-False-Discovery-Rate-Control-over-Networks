# sim/plot.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'lines.linewidth': 1.2,
    'figure.dpi': 150,
})

FIGS = os.environ.get('CAC_OUT', 'output') + '/figs'
RES = 'sim/results'

METHOD_STYLE = {
    'BH':   dict(color='black', linestyle='--', marker='s', label='Pooled BH'),
    'Bonf': dict(color='gray',  linestyle=':',  marker='^', label='Bonferroni'),
    'Orig': dict(color='C0',    linestyle='-',  marker='o', label='single-pass S&F'),
    'Adap': dict(color='C3',    linestyle='-',  marker='D', label='two-pass S&F'),
}


def _panel(ax, x, y, methods, ylabel, xlabel, title=None, ylim=None):
    for j, key in enumerate(methods):
        ax.plot(x, y[:, j], **METHOD_STYLE[key], markersize=3)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.3)


def plot_headline_strong_panel():
    """Supplementary: two rows (FDR / Power) x four cols (E1 strong–E4)."""
    os.makedirs(FIGS, exist_ok=True)
    e1 = np.load(f'{RES}/exp1.npz')
    e2 = np.load(f'{RES}/exp2.npz')
    e3 = np.load(f'{RES}/exp3.npz')
    e4 = np.load(f'{RES}/exp4.npz')

    fig, axes = plt.subplots(2, 4, figsize=(11, 4.2), constrained_layout=True)
    methods = ['BH', 'Bonf', 'Orig', 'Adap']
    titles = [
        r'(a) vary $M$ (strong signal, reproduces [PX23])',
        r'(b) vary $\lambda$',
        r'(c) vary $N$',
        r'(d) vary $\rho$',
    ]
    xs = [e1['M_grid'], e2['lam_grid'], e3['N_grid'], e4['rho_grid']]
    xlabels = [r'$M$ (total)', r'$\lambda$', r'$N$', r'$\rho$']
    fdrs = [e1['fdr'], e2['fdr'], e3['fdr'], e4['fdr']]
    powers = [e1['power'], e2['power'], e3['power'], e4['power']]

    fig.suptitle('Strong-signal regime (reproducing [PX23])', fontsize=10)

    for c in range(4):
        _panel(axes[0, c], xs[c], fdrs[c], methods, 'FDR', xlabels[c],
               title=titles[c], ylim=(0, 0.45))
        _panel(axes[1, c], xs[c], powers[c], methods, 'Power', xlabels[c],
               ylim=(0, 1.0))
        axes[0, c].axhline(0.2, color='red', linewidth=0.8, linestyle=':', label=r'$\alpha=0.2$')

    axes[0, 0].legend(loc='upper left', fontsize=7, ncol=2)
    fig.savefig(f'{FIGS}/fig_headline_strong.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_headline_strong.pdf')


def plot_headline_panel():
    """Headline: two rows (FDR / Power) x five cols from weak-signal experiments."""
    os.makedirs(FIGS, exist_ok=True)
    e1w = np.load(f'{RES}/exp1_weak.npz')
    e2w = np.load(f'{RES}/exp2_weak.npz')
    e3w = np.load(f'{RES}/exp3_weak.npz')
    e4w = np.load(f'{RES}/exp4_weak.npz')
    eaw = np.load(f'{RES}/exp_alpha_weak.npz')

    # drawn at the printed width (7.1 in) so that 7-pt lettering stays 7 pt
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'legend.fontsize': 7})
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 2.1), constrained_layout=True)
    for a_ in axes.ravel():
        a_.tick_params(labelsize=6.5)
    methods = ['BH', 'Bonf', 'Orig', 'Adap']
    titles = [
        r'(a) vary $M$',
        r'(b) vary $\lambda$',
        r'(c) vary $N$',
        r'(d) vary $\rho$',
        r'(e) vary $\alpha$',
    ]
    xs = [e1w['M_grid'], e2w['lam_grid'], e3w['N_grid'], e4w['rho_grid'], eaw['alpha_grid']]
    xlabels = [r'$M$ (total)', r'$\lambda$', r'$N$', r'$\rho$', r'$\alpha$']
    fdrs   = [e1w['fdr'],   e2w['fdr'],   e3w['fdr'],   e4w['fdr'],   eaw['fdr']]
    powers = [e1w['power'], e2w['power'], e3w['power'], e4w['power'], eaw['power']]

    for c in range(5):
        _panel(axes[0, c], xs[c], fdrs[c], methods, 'FDR', xlabels[c],
               title=titles[c], ylim=(0, 0.30))
        axes[0, c].axhline(0.2, color='red', linewidth=0.8, linestyle=':', label=r'$\alpha=0.2$')
        max_pow = max(float(powers[c].max()) + 0.05, 0.4)
        _panel(axes[1, c], xs[c], powers[c], methods, 'Power', xlabels[c],
               ylim=(0, max_pow))

    hh, ll = axes[0, 0].get_legend_handles_labels()
    fig.legend(hh, ll, loc='lower center', ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(f'{FIGS}/fig_headline.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_headline.pdf')


def plot_e1_weak():
    """Standalone two-panel figure (FDR + Power) for weak-signal E1 data."""
    os.makedirs(FIGS, exist_ok=True)
    e1w = np.load(f'{RES}/exp1_weak.npz')

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), constrained_layout=True)
    methods = ['BH', 'Bonf', 'Orig', 'Adap']

    _panel(axes[0], e1w['M_grid'], e1w['fdr'], methods,
           'FDR', r'$M$ (total)',
           title=r'(a) FDR — vary $M$ (weak signal)',
           ylim=(0, 0.25))
    axes[0].axhline(0.2, color='red', linewidth=0.8, linestyle=':', label=r'$\alpha=0.2$')
    axes[0].legend(loc='upper left', fontsize=7)

    _panel(axes[1], e1w['M_grid'], e1w['power'], methods,
           'Power', r'$M$ (total)',
           title=r'(b) Power — vary $M$ (weak signal)',
           ylim=(0, 1.0))

    fig.savefig(f'{FIGS}/fig_e1_weak.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_e1_weak.pdf')


def plot_e5_split():
    os.makedirs(FIGS, exist_ok=True)
    e5 = np.load(f'{RES}/exp5.npz')

    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)
    for M_total in [4, 6, 8, 10]:
        splits = e5[f'M{M_total}_splits']
        pw = e5[f'M{M_total}_power']
        ax.plot(splits[:, 0], pw, marker='o', label=f'$M_{{\\rm tot}}={M_total}$')
    ax.set_xlabel(r'$M_1$')
    ax.set_ylabel('Power')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(f'{FIGS}/fig_e5_split.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_e5_split.pdf')


def plot_e6_inflation():
    os.makedirs(FIGS, exist_ok=True)
    e6 = np.load(f'{RES}/exp6.npz')

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
    for bi, b in enumerate(e6['b_grid']):
        axes[0].plot(e6['lam_grid'], e6['fdr'][:, bi], marker='o', label=f'$b={b}$')
        axes[1].plot(e6['lam_grid'], e6['power'][:, bi], marker='o', label=f'$b={b}$')
    axes[0].axhline(0.2, color='red', linewidth=0.8, linestyle=':')
    axes[0].set_ylabel('FDR')
    axes[1].set_ylabel('Power')
    axes[0].set_ylim(0, 0.45)
    axes[1].set_ylim(0, 1.0)
    for a in axes:
        a.set_xlabel(r'$\lambda$')
        a.grid(alpha=0.3)
        a.legend()
    fig.savefig(f'{FIGS}/fig_e6_inflation.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_e6_inflation.pdf')


def plot_e7_iterative():
    os.makedirs(FIGS, exist_ok=True)
    e7 = np.load(f'{RES}/exp7.npz')

    fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)
    K = e7['K_grid']
    M = int(e7['M_per_round'])
    ax2 = ax.twinx()
    ax.plot(K, e7['fdr'],   marker='o', color='C0', label='FDR')
    ax2.plot(K, e7['power'], marker='D', color='C3', label='Power')
    ax.axhline(0.2, color='red', linewidth=0.8, linestyle=':')
    ax.set_xlabel(f'Rounds $K$ ({M} samples/round)')
    ax.set_ylabel('FDR', color='C0')
    ax2.set_ylabel('Power', color='C3')
    ax.set_ylim(0, 0.45)
    ax2.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    fig.savefig(f'{FIGS}/fig_e7_iterative.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_e7_iterative.pdf')


MULTIHOP_STYLE = {
    'Orig': dict(color='C0', linestyle='-',  marker='o', label='QuTE-S&F (orig)'),
    'Adap': dict(color='C3', linestyle='-',  marker='D', label='QuTE-S&F (adap)'),
    'BH':   dict(color='black', linestyle='--', marker='s', label='Pooled BH (ceiling)'),
}


def plot_multihop_vary_N():
    """Supplementary: FDR and Power vs N for cycle multihop.
    Saves to <CAC_OUT>/figs/fig_multihop_cycle.pdf."""
    os.makedirs(FIGS, exist_ok=True)
    d = np.load(f'{RES}/exp_multihop_vary_N.npz')
    N_grid    = d['N_grid']
    fdr_orig  = d['fdr_orig'];  fdr_adap  = d['fdr_adap']
    pow_orig  = d['pow_orig'];  pow_adap  = d['pow_adap']

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

    axes[0].plot(N_grid, fdr_orig, **MULTIHOP_STYLE['Orig'], markersize=4)
    axes[0].plot(N_grid, fdr_adap, **MULTIHOP_STYLE['Adap'], markersize=4)
    axes[0].axhline(0.2, color='red', linewidth=0.8, linestyle=':', label=r'$\alpha=0.2$')
    axes[0].set_xlabel(r'$N$ (cycle size)')
    axes[0].set_ylabel('FDR')
    axes[0].set_title('(a) FDR vs $N$ — cycle multihop')
    axes[0].set_ylim(0, 0.30)
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].plot(N_grid, pow_orig, **MULTIHOP_STYLE['Orig'], markersize=4)
    axes[1].plot(N_grid, pow_adap, **MULTIHOP_STYLE['Adap'], markersize=4)
    axes[1].set_xlabel(r'$N$ (cycle size)')
    axes[1].set_ylabel('Power')
    axes[1].set_title('(b) Power vs $N$ — cycle multihop')
    axes[1].set_ylim(0, max(float(pow_adap.max()) + 0.1, 0.5))
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.savefig(f'{FIGS}/fig_multihop_cycle.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_multihop_cycle.pdf')


def plot_multihop():
    """Three-panel multihop figure:
    (a) Power vs avg_degree (orig vs adap)
    (b) Power vs M (orig, adap, pooled-BH ceiling)
    (c) FDR vs avg_degree (orig vs adap, showing FDR control)
    Saves to <CAC_OUT>/figs/fig_multihop.pdf."""
    os.makedirs(FIGS, exist_ok=True)
    dd = np.load(f'{RES}/exp_multihop_degree.npz')
    dm = np.load(f'{RES}/exp_multihop_M.npz')

    degree_grid = dd['degree_grid']
    M_grid      = dm['M_grid']

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)

    # Panel (a): Power vs avg_degree
    ax = axes[0]
    ax.plot(degree_grid, dd['pow_orig'], **MULTIHOP_STYLE['Orig'], markersize=4)
    ax.plot(degree_grid, dd['pow_adap'], **MULTIHOP_STYLE['Adap'], markersize=4)
    ax.set_xlabel('Avg degree')
    ax.set_ylabel('Power')
    ax.set_title('(a) Vary graph degree\n(cycle baseline at left)')
    ax.set_ylim(0, max(float(dd['pow_adap'].max()) + 0.1, 0.4))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel (b): Power vs M with pooled-BH ceiling
    ax = axes[1]
    ax.plot(M_grid, dm['pow_orig'], **MULTIHOP_STYLE['Orig'], markersize=4)
    ax.plot(M_grid, dm['pow_adap'], **MULTIHOP_STYLE['Adap'], markersize=4)
    ax.plot(M_grid, dm['pow_pooled_bh'], **MULTIHOP_STYLE['BH'], markersize=4)
    ax.set_xlabel(r'$M$ (per-node budget)')
    ax.set_ylabel('Power')
    ax.set_title('(b) Vary per-node budget')
    ax.set_ylim(0, max(float(dm['pow_pooled_bh'].max()) + 0.1, 0.5))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # Panel (c): FDR vs avg_degree
    ax = axes[2]
    ax.plot(degree_grid, dd['fdr_orig'], **MULTIHOP_STYLE['Orig'], markersize=4)
    ax.plot(degree_grid, dd['fdr_adap'], **MULTIHOP_STYLE['Adap'], markersize=4)
    ax.axhline(0.2, color='red', linewidth=0.8, linestyle=':', label=r'$\alpha=0.2$')
    ax.set_xlabel('Avg degree')
    ax.set_ylabel('FDR')
    ax.set_title('(c) FDR vs degree\n(both control FDR)')
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.savefig(f'{FIGS}/fig_multihop.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_multihop.pdf')


def plot_multihop_grid():
    """Bar chart: 2x2 (orig vs adap) x (FDR vs Power) from grid experiment.
    Saves to <CAC_OUT>/figs/fig_multihop_grid.pdf."""
    os.makedirs(FIGS, exist_ok=True)
    d = np.load(f'{RES}/exp_multihop_grid.npz')
    fdr_orig = float(d['fdr_orig']); fdr_adap = float(d['fdr_adap'])
    pow_orig = float(d['pow_orig']); pow_adap = float(d['pow_adap'])
    side = int(d['side']); M_total = int(d['M_total'])

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

    labels = ['QuTE-S&F (orig)', 'QuTE-S&F (adap)']
    colors = ['C0', 'C3']

    # FDR bar chart
    ax = axes[0]
    bars = ax.bar(labels, [fdr_orig, fdr_adap], color=colors, width=0.5, alpha=0.85)
    ax.axhline(0.2, color='red', linewidth=1.0, linestyle=':', label=r'$\alpha=0.2$')
    ax.set_ylabel('FDR')
    ax.set_title(f'(a) FDR — {side}x{side} grid, $M={M_total}$')
    ax.set_ylim(0, 0.25)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, [fdr_orig, fdr_adap]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # Power bar chart
    ax = axes[1]
    bars = ax.bar(labels, [pow_orig, pow_adap], color=colors, width=0.5, alpha=0.85)
    ax.set_ylabel('Power')
    ax.set_title(f'(b) Power — {side}x{side} grid, $M={M_total}$')
    ax.set_ylim(0, max(pow_adap + 0.1, 0.3))
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, [pow_orig, pow_adap]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    fig.savefig(f'{FIGS}/fig_multihop_grid.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGS}/fig_multihop_grid.pdf')


if __name__ == '__main__':
    plot_headline_panel()          # fig_headline.pdf       (weak, 2x5, new headline)
    plot_headline_strong_panel()   # fig_headline_strong.pdf (strong, supplementary)
    plot_e1_weak()                 # fig_e1_weak.pdf
    plot_e5_split()                # fig_e5_split.pdf
    plot_e6_inflation()            # fig_e6_inflation.pdf
    plot_e7_iterative()            # fig_e7_iterative.pdf
    plot_multihop()                # fig_multihop.pdf        (3-panel: degree, M, FDR)
    plot_multihop_grid()           # fig_multihop_grid.pdf   (2x2 bar chart)
    plot_multihop_vary_N()         # fig_multihop_cycle.pdf  (supplementary, cycle vary-N)
    print('All figures generated.')
