import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

def q_sigma0(alpha):
    gamma = 1.0 / alpha
    am = (1 - np.sqrt(gamma))**2
    ap = (1 + np.sqrt(gamma))**2
    dens = lambda x: alpha * np.sqrt(max((ap-x)*(x-am), 0)) / (2*np.pi*x)
    E_sqrt, _ = integrate.quad(lambda x: np.sqrt(x)*dens(x),
                                am, ap, limit=600, points=[am, ap])
    return E_sqrt**2

def D_sigma0(alpha):
    gamma = 1.0 / alpha
    am = (1 - np.sqrt(gamma))**2
    ap = (1 + np.sqrt(gamma))**2
    dens = lambda x: alpha * np.sqrt(max((ap-x)*(x-am), 0)) / (2*np.pi*x)
    E_log, _ = integrate.quad(lambda x: np.log(x)*dens(x),
                               am, ap, limit=600, points=[am, ap])
    return -0.5 * E_log

def inf_sigma0(alpha):
    # formula chiusa: 1/2 * gamma/(1-gamma)
    gamma = 1.0 / alpha
    return 0.5 * gamma / (1 - gamma)

alphas   = np.linspace(1.03, 20, 300)

inf_arr  = np.array([inf_sigma0(a)               for a in alphas])
D_arr    = np.array([D_sigma0(a)                 for a in alphas])
logq_arr = np.array([-0.5 * np.log(q_sigma0(a)) for a in alphas])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# pannello sinistro: scala log, le tre quantità
ax = axes[0]
ax.semilogy(alphas, inf_arr,  lw=2.5, color='#2ca02c',
            label=r'$\bar\Delta/d = \frac{\gamma}{2(1-\gamma)}$')
ax.semilogy(alphas, D_arr,    lw=2.5, color='#ff7f0e', ls='-.',
            label=r'$\mathcal{D}$ (KL divergence)')
ax.semilogy(alphas, logq_arr, lw=2.5, color='#1f77b4', ls='--',
            label=r'$-\frac{1}{2}\log q$')
ax.set_xlabel(r'$\alpha = n/d$', fontsize=12)
ax.set_ylabel('(log scale)', fontsize=12)
ax.set_title(r'Sandwich $-\frac{1}{2}\log q \leq \mathcal{D} \leq \bar\Delta/d$', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.25)

# pannello destro: quanto è stretto il sandwich?
ax = axes[1]
ax.plot(alphas, inf_arr / D_arr,  lw=2.5, color='#2ca02c',
        label=r'$(\bar\Delta/d)\;/\;\mathcal{D}$')
ax.plot(alphas, D_arr / logq_arr, lw=2.5, color='#1f77b4', ls='--',
        label=r'$\mathcal{D}\;/\;(-\frac{1}{2}\log q)$')
ax.axhline(1, color='gray', ls=':', lw=1)
ax.set_xlabel(r'$\alpha = n/d$', fontsize=12)
ax.set_ylabel('ratio', fontsize=12)
ax.set_title('Quanto è stretto il sandwich?', fontsize=12)
ax.set_ylim(0.9, 6)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)

fig.suptitle(
    r'$\sigma^2=0$: sandwich $-\frac{1}{2}\log q \leq \mathcal{D} \leq \bar\Delta/d$'
    r'  in funzione di $\alpha = n/d$', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/sandwich_alpha.png', dpi=160)
