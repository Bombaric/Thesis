import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# BLOCCO 1: aspettazione rispetto alla MP
# Calcola E_MP[f(nu)] dove nu = (lambda + sigma^2) / (1 + sigma^2)
# e lambda segue la distribuzione di Marchenko-Pastur con ratio alpha = n/d
# ──────────────────────────────────────────────
def mp_moment(alpha, lam, func, limit=600):
    gc = 1.0 / alpha                               # gamma = d/n
    am = (1 - np.sqrt(gc))**2                      # bordo sinistro supporto MP
    ap = (1 + np.sqrt(gc))**2                      # bordo destro supporto MP
    pm = max(0.0, 1 - alpha)                       # massa puntuale a lambda=0 (se n < d)

    # densità continua di Marchenko-Pastur
    dens = lambda x: alpha * np.sqrt(max((ap - x) * (x - am), 0)) / (2 * np.pi * x)

    # integrale sulla parte continua
    bulk, _ = integrate.quad(
        lambda x: func((x + lam) / (1 + lam)) * dens(x),
        am, ap, limit=limit, points=[am, ap]
    )

    # contributo dell'atomo a lambda=0 (nu_0 = sigma^2 / (1+sigma^2))
    atom = pm * func(lam / (1 + lam))

    return bulk + atom


# ──────────────────────────────────────────────
# BLOCCO 2: le quattro quantità fisiche
# ──────────────────────────────────────────────

def I_of(alpha, sigma2):
    # spectral fill: E[lambda/(lambda+sigma^2)]
    # quante direzioni sono effettivamente apprese
    # ricavato da  I = 1 - sigma^2/(1+sigma^2) * E[nu^{-1}]
    return 1.0 - (sigma2 / (1 + sigma2)) * mp_moment(alpha, sigma2, lambda nu: 1 / nu)

def h_of(alpha, sigma2):
    # leverage media: h = gamma * I,  sempre in [0, 1)
    return I_of(alpha, sigma2) / alpha

def q_of(alpha, sigma2):
    # convergence overlap della paper: q = (E[sqrt(nu)])^2
    return mp_moment(alpha, sigma2, lambda nu: np.sqrt(nu))**2

def inf_of(alpha, sigma2):
    # influence density: Delta_bar / d = h*I / (2*(1-h))
    # grande per alpha < 1, va a 0 per alpha grande
    h = h_of(alpha, sigma2)
    I = I_of(alpha, sigma2)
    return 0.5 * h * I / (1 - h)


# ──────────────────────────────────────────────
# BLOCCO 3: le due rho bounded in [0,1]
#
#   rho_A = 1 / (1 + Inf)      trasformazione di Mobius diretta
#   rho_B = I * (1 + sigma^2)  spectral fill normalizzato
# ──────────────────────────────────────────────

alphas = np.linspace(0.01, 7.5, 300)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, s in zip(axes, [0.1, 0.01]):

    q_arr    = np.array([q_of(a, s)              for a in alphas])
    rhoA_arr = np.array([1 / (1 + 2*inf_of(a, s))  for a in alphas])
    #rhoB_arr = np.array([I_of(a, s) * (1 + s)    for a in alphas])

    ax.plot(alphas, q_arr,    lw=2.5, color='#1f77b4',
            label=r'$q$')
    ax.plot(alphas, rhoA_arr, lw=2.5, color='#d62728',
            label=r'$\rho = \frac{1}{1+2\mathcal{I}}$')
    #ax.plot(alphas, rhoB_arr, lw=2.5, color='#2ca02c', ls='--',
            #label=r'$\tilde\rho_B = I\,(1{+}\sigma^2)$')

    ax.axvline(1, color='gray', ls=':', lw=1)   # soglia alpha = 1
    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$\alpha = n/d$', fontsize=12)
    ax.set_title(fr'$\sigma^2 = {s}$', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel('value', fontsize=12)
fig.suptitle(r'Bounded objects in $[0,1]$ derived from $q$ and $\mathcal{I}$', fontsize=12)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/bounded_objects.png', dpi=160)