import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# ── integrali MP ─────────────────────────────────────────────────────────────
def mp_moment(alpha, lam, func, limit=600):
    gc = 1.0 / alpha
    am = (1 - np.sqrt(gc))**2
    ap = (1 + np.sqrt(gc))**2
    pm = max(0.0, 1 - alpha)
    dens = lambda x: alpha * np.sqrt(max((ap-x)*(x-am), 0)) / (2*np.pi*x)
    bulk, _ = integrate.quad(
        lambda x: func((x+lam)/(1+lam)) * dens(x),
        am, ap, limit=limit, points=[am, ap]
    )
    return bulk + pm * func(lam/(1+lam))

def q_of(a, s):
    return mp_moment(a, s, lambda nu: np.sqrt(nu))**2

def inf_of(a, s):
    return 0.5 * (mp_moment(a, s, lambda nu: 1/nu) - 1)

def V_of(a, s):
    return mp_moment(a, s, lambda nu: (nu-1)**2)

# ── griglia ───────────────────────────────────────────────────────────────────
alphas = np.geomspace(0.01, 500, 500)
s = 0.01

q_arr   = np.array([q_of(a, s)   for a in alphas])
inf_arr = np.array([inf_of(a, s) for a in alphas])
V_arr   = np.array([V_of(a, s)   for a in alphas])

# Nuova definizione di rho
rho_arr = 1.0 / (1.0 + 2.0 * inf_arr)
one_minus_rho_arr = 1.0 - rho_arr

# ── plot 1: α → ∞ ─────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))

# Espansioni per grandi campioni
plt.loglog(alphas, 1 - q_arr,       lw=2.5, color='#1f77b4', label=r'$1-q$')
plt.loglog(alphas, one_minus_rho_arr, lw=2.5, color='#2ca02c', label=r'$1-\rho$')
plt.loglog(alphas, inf_arr,       lw=2.5, color='#d62728', ls='--', label=r'$\mathcal{I}$')

# Asintoti teorici (usando V_arr come approssimazione lineare di V)
plt.loglog(alphas, V_arr/4,         lw=1.5, color='#1f77b4', ls=':', label=r'$V/4$')
plt.loglog(alphas, V_arr,           lw=1.5, color='#2ca02c', ls=':', label=r'$V$')
plt.loglog(alphas, V_arr/2,         lw=1.5, color='#d62728', ls=':', label=r'$V/2$')
plt.axvline(1, color='gray', ls=':', lw=1)
plt.xlabel(r'$\alpha = n/d$', fontsize=11)
plt.title(r'$\alpha\to\infty$: $1-q\approx V/4$, $1-\rho\approx V$, $\mathcal{I}\approx V/2$', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, which='both', alpha=0.25)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/taylor_asymptotics_large.png', dpi=160)
plt.close()

# ── plot 2: α → 0 ─────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))

plt.loglog(alphas, q_arr,   lw=2.5, color='#1f77b4', label=r'$q$')
plt.loglog(alphas, inf_arr, lw=2.5, color='#d62728', ls='--', label=r'$\mathcal{I}$')
plt.loglog(alphas, rho_arr, lw=2.5, color='#2ca02c', ls='--', label=r'$\rho$')

# Nuovi sviluppi rigorosi per piccoli alpha
q_asymp = (s + 2*np.sqrt(s)*np.sqrt(alphas) + alphas*(1 - 2*s)) / (1 + s)


plt.loglog(alphas, q_asymp,   lw=1.5, color='#1f77b4', ls=':', label=r'$1/(1+\sigma^2)(2\sigma\sqrt{\alpha} + \sigma^2)$')
plt.axhline(1/(2*s), lw=1.5, color='#d62728', ls=':', label=r'$1/(2\sigma^2)$')
plt.axhline(s/(1+s),          lw=1.5, color='#2ca02c', ls=':', label=r'$\sigma^2/(1+\sigma^2)$')

plt.axvline(1, color='gray', ls=':', lw=1)
plt.xlabel(r'$\alpha = n/d$', fontsize=11)
plt.title(r'$\alpha\to 0$: Rigorous Marchenko-Pastur Expansion', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, which='both', alpha=0.25)
plt.ylim(1e-4, 100)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/taylor_asymptotics_small.png', dpi=160)
plt.close()