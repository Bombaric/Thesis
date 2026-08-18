'''import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Funzioni Teoriche (Marchenko-Pastur) ---

def mp_density(lam, gamma):
    if gamma == 0: return 0
    lam_plus = (1 + np.sqrt(gamma))**2
    lam_minus = (1 - np.sqrt(gamma))**2
    if lam < lam_minus or lam > lam_plus:
        return 0
    return np.sqrt((lam_plus - lam) * (lam - lam_minus)) / (2 * np.pi * gamma * lam)

def compute_spectral_values(gamma, sigma2):
    """Calcola i valori attesi teorici basati sulla distribuzione spettrale."""
    lam_plus = (1 + np.sqrt(gamma))**2
    lam_minus = (1 - np.sqrt(gamma))**2
    
    # 1. Calcolo di I = E[ lambda / (lambda + sigma2) ]
    I_integrand = lambda l: (l / (l + sigma2)) * mp_density(l, gamma)
    I_val, _ = quad(I_integrand, lam_minus, lam_plus)
    
    # Risultati asintotici normalizzati per d (solo Influence Density)
    spectral_delta = 0.5 * (gamma * I_val**2) / (1 - gamma * I_val)
    
    return spectral_delta

# --- Configurazione Simulazione ---
d = 400
# Starting from 10 is safe for r = n/(n-1)
n_values = np.linspace(10, 3000, 60).astype(int) 
alphas = n_values / d  # Sample complexity n/d

sigma2_list = [0.001, 0.01, 0.1]

# Setup the figure for a single plot
plt.figure(figsize=(10, 6))

print("Running simulations for Influence Density. This may take a moment...")

for idx, sigma2 in enumerate(sigma2_list):
    emp_delta = []
    theory_delta = []
    theory_dkl = []
    
    color = plt.cm.tab10(idx)
    
    for n in n_values:
        g = d / n
        
        # --- 1. Parte Empirica (Simulazione) ---
        X = np.random.randn(n, d)
        Sigma_n = np.dot(X.T, X) / n + sigma2 * np.eye(d)
        
        # Influence Function
        V = np.linalg.solve(Sigma_n, X.T)
        h_vec = np.sum(X * V.T, axis=1) / n
        r = n / (n - 1)
        term_det = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
        term_quad = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)
        emp_delta.append(np.mean(term_det + term_quad) / d)
        c = 0.5 * (np.trace(Sigma_n) / d - 1 - np.log(np.linalg.det(Sigma_n)) / d)
        theory_dkl.append(c)

        
        # --- 2. Parte Teorica (Formule Spettrali) ---
        s_delta = compute_spectral_values(g, sigma2)
        theory_delta.append(s_delta)

    gammas = d / n_values
        
    # --- Plot: Influence Density ---
    plt.plot(alphas, emp_delta, 'o', color=color, markersize=4, alpha=0.7, label=r' Influence Density ($\sigma^2$= '+f'{sigma2})')
    plt.plot(alphas, theory_delta, '-', color=color, linewidth=2, label=r'Spectral asymptotic limit $\mathcal{I}$ ($\sigma^2$= '+f'{sigma2})')
    plt.plot(alphas, theory_dkl, '--', color=color, linewidth=2, label=r'Excess KL Divergence $\mathcal{D}$ ($\sigma^2$= '+f'{sigma2})')

# Formatting
plt.yscale('log')
plt.xlabel(r' $\alpha = n/d$')
plt.ylabel(r'$\mathcal{I}$ (Log Scale)')
plt.title(r'Influence density (computed and spectral asymptotic limit $\mathcal{I}$) vs $\mathcal{D}$ (excess KL divergence)')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.tight_layout()

plt.savefig('/home/ceci/Thesis/Plots/validation_spectral_influence_only.png', dpi=300)
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Funzioni Teoriche (Marchenko-Pastur) ---

def mp_density(lam, gamma):
    if gamma == 0: return 0
    lam_plus = (1 + np.sqrt(gamma))**2
    lam_minus = (1 - np.sqrt(gamma))**2
    if lam < lam_minus or lam > lam_plus:
        return 0
    return np.sqrt((lam_plus - lam) * (lam - lam_minus)) / (2 * np.pi * gamma * lam)

def compute_spectral_values(gamma, sigma2):
    """Calcola i valori attesi teorici basati sulla distribuzione spettrale."""
    lam_plus = (1 + np.sqrt(gamma))**2
    lam_minus = (1 - np.sqrt(gamma))**2
    
    # 1. Calcolo di I = E[ lambda / (lambda + sigma2) ]
    I_integrand = lambda l: (l / (l + sigma2)) * mp_density(l, gamma)
    I_val, _ = quad(I_integrand, lam_minus, lam_plus)
    
    # Risultati asintotici normalizzati per d (solo Influence Density)
    spectral_delta = 0.5 * (gamma * I_val**2) / (1 - gamma * I_val)
    
    return spectral_delta

# --- Configurazione Simulazione ---
d = 400
# Starting from 10 is safe for r = n/(n-1)
n_values = np.linspace(10, 3000, 60).astype(int) 
alphas = n_values / d  # Sample complexity n/d

sigma2_list = [0.001, 0.01, 0.1]

# Setup the figure for 3 distinct subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

print("Running simulations for Influence Density. This may take a moment...")

for idx, sigma2 in enumerate(sigma2_list):
    emp_delta = []
    theory_delta = []
    theory_dkl = []
    
    color = plt.cm.tab10(idx)
    ax = axes[idx]
    
    for n in n_values:
        g = d / n
        
        # --- 1. Parte Empirica (Simulazione) ---
        X = np.random.randn(n, d)
        Sigma_n = np.dot(X.T, X) / n + sigma2 * np.eye(d)
        
        # Influence Function
        V = np.linalg.solve(Sigma_n, X.T)
        h_vec = np.sum(X * V.T, axis=1) / n
        r = n / (n - 1)
        term_det = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
        term_quad = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)
        emp_delta.append(np.mean(term_det + term_quad) / d)
        
        # Safe log-determinant calculation to prevent numerical underflow
        _, logdet_sig = np.linalg.slogdet(Sigma_n)
        c = 0.5 * (np.trace(Sigma_n) / d - 1 - logdet_sig / d)
        theory_dkl.append(c)
        
        # --- 2. Parte Teorica (Formule Spettrali) ---
        s_delta = compute_spectral_values(g, sigma2)
        theory_delta.append(s_delta)
        
    # --- Plotting su Subplot Individuale ---
    ax.plot(alphas, emp_delta, 'o', color=color, markersize=4, alpha=0.7, label=r'Empirical Influence')
    ax.plot(alphas, theory_delta, '-', color=color, linewidth=2, label=r'Spectral Limit $\mathcal{I}$')
    ax.plot(alphas, theory_dkl, '--', color=color, linewidth=2, label=r'Excess KL $\mathcal{D}$')

    # Formatting Subplot
    ax.set_yscale('log')
    ax.set_xlabel(r'Sample Complexity $\alpha = n/d$')
    if idx == 0:
        ax.set_ylabel('Value (Log Scale)')
    ax.set_title(rf'Generalization Metrics ($\sigma^2 = {sigma2}$)')
    ax.legend(fontsize=14)
    ax.grid(True, which="both", linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/validation_spectral_separated.png', dpi=300)
print("Plots saved successfully!")