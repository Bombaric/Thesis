import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Funzioni Teoriche (Marchenko-Pastur) ---

def mp_density(lam, gamma):
    if gamma == 0:
        return 0
    lam_plus = (1 + np.sqrt(gamma))**2
    lam_minus = (1 - np.sqrt(gamma))**2
    if lam < lam_minus or lam > lam_plus:
        return 0
    return np.sqrt((lam_plus - lam) * (lam - lam_minus)) / (2 * np.pi * gamma * lam)

def compute_spectral_values(gamma, sigma2):
    """Calcola i valori attesi teorici basati sulla distribuzione spettrale."""
    lam_plus = (1 + np.sqrt(gamma))**2
    lam_minus = (1 - np.sqrt(gamma))**2

    I_integrand = lambda l: (l / (l + sigma2)) * mp_density(l, gamma)
    I_val, _ = quad(I_integrand, lam_minus, lam_plus)

    KL_integrand = lambda l: np.log(l + sigma2) * mp_density(l, gamma)
    log_exp_bulk, _ = quad(KL_integrand, lam_minus, lam_plus)

    if gamma > 1.0:
        point_mass_weight = 1.0 - (1.0 / gamma)
        point_mass_contribution = point_mass_weight * np.log(sigma2)
        log_exp = log_exp_bulk + point_mass_contribution
    else:
        log_exp = log_exp_bulk

    spectral_delta = 0.5 * (gamma * I_val**2) / (1 - gamma * I_val)
    spectral_kl = 0.5 * (sigma2 - log_exp)

    return spectral_delta, spectral_kl

# --- Configurazione Simulazione ---
d = 400
n_values = np.linspace(10, 3000, 60).astype(int)
alphas = n_values / d
gammas = d / n_values

sigma2_list = [0.001, 0.01, 0.1]
colors = ['tab:blue', 'tab:orange', 'tab:green']

emp_delta_dict = {}
theory_delta_dict = {}

for sigma2 in sigma2_list:
    emp_delta = []
    theory_delta = []

    for n in n_values:
        g = d / n

        X = np.random.randn(n, d)
        Sigma_n = (X.T @ X) / n + sigma2 * np.eye(d)

        V = np.linalg.solve(Sigma_n, X.T)
        h_vec = np.sum(X * V.T, axis=1) / n
        r = n / (n - 1)
        term_det = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
        term_quad = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)
        emp_delta.append(np.mean(term_det + term_quad) / d)

        s_delta, _ = compute_spectral_values(g, sigma2)
        theory_delta.append(s_delta)

    emp_delta_dict[sigma2] = emp_delta
    theory_delta_dict[sigma2] = theory_delta

# --- Plot 1: in funzione di gamma ---
plt.figure(figsize=(10, 6))
for sigma2, color in zip(sigma2_list, colors):
    plt.plot(gammas, emp_delta_dict[sigma2], 'o', color=color,
             label=fr'Influence Density ($\sigma^2$={sigma2})', markersize=4)
    plt.plot(gammas, theory_delta_dict[sigma2], '-', color=color,
             label=fr'Spectral asymptotic limit ($\sigma^2$={sigma2})', alpha=0.7)
plt.yscale('log')
plt.xlabel(r'$\gamma = d/n$')
plt.ylabel(r'$\mathcal{I}$ (Log Scale)')
plt.title(r'Influence density vs spectral asymptotic limit — vs $\gamma$')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/validation_spectral_gamma.png', dpi=300)
plt.close()

# --- Plot 2: in funzione di alpha ---
plt.figure(figsize=(10, 6))
for sigma2, color in zip(sigma2_list, colors):
    plt.plot(alphas, emp_delta_dict[sigma2], 'o', color=color,
             label=fr'Influence Density ($\sigma^2$={sigma2})', markersize=4)
    plt.plot(alphas, theory_delta_dict[sigma2], '-', color=color,
             label=fr'Spectral asymptotic limit ($\sigma^2$={sigma2})', alpha=0.7)
plt.yscale('log')
plt.xlabel(r'$\alpha = n/d$')
plt.ylabel(r'$\mathcal{I}$ (Log Scale)')
plt.title(r'Influence density vs spectral asymptotic limit — vs $\alpha$')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/validation_spectral_alpha.png', dpi=300)
plt.close()