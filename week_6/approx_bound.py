import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Funzioni Teoriche (Marchenko-Pastur) ---

def mp_density(lam, gamma):
    """Densità di Marchenko-Pastur per il rapporto d/n = gamma."""
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
    # Nota: Usiamo la media su d, quindi integriamo rispetto alla densità MP
    I_integrand = lambda l: (l / (l + sigma2)) * mp_density(l, gamma)
    I_val, _ = quad(I_integrand, lam_minus, lam_plus)
    
    # 2. Calcolo di E[ ln(lambda + sigma2) ] per la KL
    KL_integrand = lambda l: np.log(l + sigma2) * mp_density(l, gamma)
    log_exp, _ = quad(KL_integrand, lam_minus, lam_plus)
    
    # Risultati asintotici normalizzati per d
    # Influenza (Delta/d)
    spectral_delta = 0.5 * (gamma * I_val**2) / (1 - gamma * I_val)
    # KL (D/d)
    spectral_kl = 0.5 * (sigma2 - log_exp)
    
    return spectral_delta, spectral_kl

# --- Configurazione Simulazione ---
d = 400
n_values = np.linspace(450, 3000, 30).astype(int) # Evitiamo n < d per stabilità numerica se sigma2 è piccolo
alphas = n_values / d  # Sample complexity n/d
gamma_vals = 1 / alphas # d/n

sigma2 = 0.1 # Regolarizzazione (lambda nel tuo codice)

# Liste per i risultati
emp_delta = []
emp_kl = []
theory_delta = []
theory_kl = []

for n in n_values:
    g = d / n
    
    # --- 1. Parte Empirica (Simulazione) ---
    X = np.random.randn(n, d)
    Sigma_n = (X.T @ X) / n + sigma2 * np.eye(d)
    
    # Influence Function
    V = np.linalg.solve(Sigma_n, X.T)
    h_vec = np.sum(X * V.T, axis=1) / n
    r = n / (n - 1)
    term_det = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
    term_quad = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)
    emp_delta.append(np.mean(term_det + term_quad) / d)
    
    # KL Divergence
    _, logdet_sig = np.linalg.slogdet(Sigma_n)
    emp_kl.append(0.5 * (np.trace(Sigma_n)/d - 1 - logdet_sig/d))
    
    # --- 2. Parte Teorica (Formule Spettrali) ---
    s_delta, s_kl = compute_spectral_values(g, sigma2)
    theory_delta.append(s_delta)
    theory_kl.append(s_kl)

# --- Plot ---
plt.figure(figsize=(10, 6))

# Influenza
plt.plot(alphas, emp_delta, 'ro', label=f'Empirical Influence (σ²={sigma2})', markersize=4)
plt.plot(alphas, theory_delta, 'r-', label=f'Theoretical Spectral Influence', alpha=0.7)

# KL
plt.plot(alphas, emp_kl, 'bs', label=f'Empirical KL (σ²={sigma2})', markersize=4)
plt.plot(alphas, theory_kl, 'b-', label=f'Theoretical Spectral KL', alpha=0.7)

# Riferimento base lambda=0 (solo per confronto visivo)
plt.plot(alphas, 0.5 * (1/alphas / (1 - 1/alphas)), 'k--', label='Base Limit (σ²=0)', alpha=0.3)

plt.yscale('log')
plt.xlabel(r'Sample Complexity $\alpha = n/d$')
plt.ylabel('Value / d (Log Scale)')
plt.title('Validation: Empirical vs Spectral Formula (Marchenko-Pastur)')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/validation_spectral.png', dpi=150)