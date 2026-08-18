import numpy as np

import matplotlib.pyplot as plt

import scipy.integrate as integrate


# --- Theoretical Functions ---

def analytical_q_factor(alpha, lam):

    gamma = 1.0 / alpha

    lam_minus = (1.0 - np.sqrt(gamma))**2

    lam_plus = (1.0 + np.sqrt(gamma))**2

   

    def integrand(x):

        radicand = np.clip((lam_plus - x) * (x - lam_minus), 0, None)

        density = np.sqrt(radicand) / (2 * np.pi * gamma * x)

        return np.sqrt(x + lam) * density


    bulk_integral, _ = integrate.quad(integrand, lam_minus, lam_plus, limit=200)

    point_mass = (1.0 - 1.0 / gamma) * np.sqrt(lam) if gamma > 1.0 else 0.0

    return (1.0 / (1.0 + lam)) * ((bulk_integral + point_mass)**2)


def exact_rho(alpha, lam):

    """rho = I * (1 + sigma^2). Using h = gamma * E[lambda/(lambda+sigma^2)]"""

    gamma = 1.0 / alpha

    b = lam - gamma + 1.0

    m = (-b + np.sqrt(b**2 + 4.0 * gamma * lam)) / (2.0 * gamma * lam)

    I = (1.0 - lam * m)

    # p = h / (gamma * (1-h)) - 1

    # Influence I = 0.5 * p. rho = I * (1 + lam)

    # rho = 0.5 * (1 + lam) * (h / (gamma * (1-h)) - 1)

   

    return gamma *I / (1.0 - I *(1.0 - gamma))


# --- Plotting Configuration ---

sigmas = [0.1, 0.01]

alphas = np.logspace(-2, 2, 100) # Log scale for better visibility of alpha=1

fig, axes = plt.subplots(1, 2, figsize=(16, 6))


for i, lam in enumerate(sigmas):

    q_vals = [analytical_q_factor(a, lam) for a in alphas]

    rho_vals = [exact_rho(a, lam) for a in alphas]

   

    ax = axes[i]

    ax.semilogx(alphas, q_vals, label=r'$q$ (convergence overlap)', linewidth=2.5)

    ax.semilogx(alphas, rho_vals, label=r'$\rho = I(1+\sigma^2)$ (spectral fill)', linewidth=2.5)

    #y in logscale


    # Formatting

    ax.set_title(rf'$\sigma^2 = {lam}$', fontsize=14)

    ax.axvline(1.0, color='gray', linestyle=':', alpha=0.6)

    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.6)

    ax.set_xlabel(r'$\alpha = n/d$', fontsize=12)

    ax.set_ylim([-0.05, 1.6])

    ax.grid(True, which="both", linestyle='-', alpha=0.2)

    ax.legend(fontsize=11)


axes[0].set_ylabel('value', fontsize=12)

plt.tight_layout()

plt.savefig('/home/ceci/Thesis/Plots/q_vs_I.png', dpi=300) 