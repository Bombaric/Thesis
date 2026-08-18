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
    # Var(nu) esatta dalla distribuzione MP
    return mp_moment(a, s, lambda nu: (nu-1)**2)

# ── griglia ───────────────────────────────────────────────────────────────────
alphas = np.geomspace(0.01, 500, 500)
s = 0.01

q_arr   = np.array([q_of(a, s)   for a in alphas])
inf_arr = np.array([inf_of(a, s) for a in alphas])
one_minus_rho_arr = np.array([inf_of(a, s) / (1 + inf_of(a, s)) for a in alphas])
V_arr   = np.array([V_of(a, s)   for a in alphas])

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── pannello sinistro: α → ∞ ─────────────────────────────────────────────────
# Taylor: ν = 1+u, E[u]=0, E[u²]=V
#   √ν ≈ 1 + u/2 - u²/8  →  E[√ν] ≈ 1 - V/8  →  q ≈ 1 - V/4
#   ν⁻¹ ≈ 1 - u + u²      →  E[ν⁻¹] ≈ 1 + V   →  Inf ≈ V/2
ax = axes[0]
ax.loglog(alphas, 1 - q_arr,              lw=2.5, color='#1f77b4',
          label=r'$1-q$')
ax.loglog(alphas, inf_arr,    lw=2.5, color='#d62728', ls='--',
          label=r'$\mathcal{I}$')
ax.loglog(alphas, one_minus_rho_arr, lw=2.5, color='#2ca02c', ls='--',
          label=r'$1-\rho$')
ax.loglog(alphas, V_arr/4,                lw=1.5, color='#1f77b4', ls=':',
          label=r'$V/4$')
ax.loglog(alphas, V_arr/2,                lw=1.5, color='#d62728', ls=':',
          label=r'$V/2$')
ax.axvline(1, color='gray', ls=':', lw=1)
ax.set_xlabel(r'$\alpha = n/d$', fontsize=12)
ax.set_title(
    r'$\alpha\to\infty$: $\nu=1+\epsilon$,  $1-q\approx V/4$,  $\mathcal{I}\approx V/2$',
    fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.25)

# ── pannello destro: α → 0 ───────────────────────────────────────────────────
# Due componenti:
#   bulk  (peso α,   ν_bulk ~ 1/(α(1+σ²)))  →  q ≈ α/(1+σ²)
#   zero-mode (peso 1-α, ν₀ = σ²/(1+σ²))   →  Inf ≈ 1/(2σ²)  →  1/(1+Inf) ≈ 2σ²
ax = axes[1]
ax.loglog(alphas, q_arr,              lw=2.5, color='#1f77b4',
          label=r'$q$')
ax.loglog(alphas, inf_arr,      lw=2.5, color='#d62728', ls='--',
          label=r'$\mathcal{I}$')
ax.loglog(alphas, 1-one_minus_rho_arr, lw=2.5, color='#2ca02c', ls='--',
          label=r'$\rho$')
eig_q = (np.sqrt(alphas/(1+s)) + (1-alphas)*np.sqrt((s/(1+s))))**2
ax.loglog(alphas, eig_q,       lw=1.5, color='#1f77b4', ls=':',
          label=r'$\alpha/(1+\sigma^2)$')
ax.axhline(1/(2*s),                       lw=1.5, color='#d62728', ls=':',
           label=r'$1/(2\sigma^2)$')
ax.axhline(2*s,                          lw=1.5, color='#2ca02c', ls=':',
           label=r'$2\sigma^2$')
ax.axvline(1, color='gray', ls=':', lw=1)
ax.set_xlabel(r'$\alpha = n/d$', fontsize=12)
ax.set_title(
    r'$\alpha\to 0$',
    fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.25)
ax.set_ylim(1e-4, 100)

fig.suptitle(r'Asymptotic behaviour: $\alpha \to \infty$ and $\alpha \to 0$ ($\sigma^2=0.01$)', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/taylor_asymptotics.png', dpi=160)