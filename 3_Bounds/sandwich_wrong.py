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

# --- Configurazione ---
# --- Configurazione ---
d = 400
n_values = np.linspace(20, 3000, 50).astype(int)
alphas = n_values / d  # n/d

# Regularization list (using sigma2 notation)
sigma2 = 0.01

print("Generating data and computing the failed pointwise bound...")

# Only need these flat lists now!
theory_delta = []
avg_delta_vals = []
kl_id_vals = []
c_wrong_bound_vals = []

for n in n_values:
        gamma = d / n # Aspect ratio gamma
        
        # 1. Data generation
        X = np.random.randn(n, d)
        
        # 2. Regularized empirical covariance
        Sigma_n = (X.T @ X) / n + sigma2 * np.eye(d)
        
        # 3. Average Influence Function (Rescaled 1/d)
        V = np.linalg.solve(Sigma_n, X.T)
        h_vec = np.sum(X * V.T, axis=1) / n
        r = n / (n - 1)
        
        term_det_inf = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
        term_quad_inf = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)
        
        avg_delta = np.mean(term_det_inf + term_quad_inf) / d
        avg_delta_vals.append(avg_delta)
        
        # 4. Shifted KL Divergence
        trace_sig = np.trace(Sigma_n)
        _, logdet_sig = np.linalg.slogdet(Sigma_n)
        logdet_sigma = d * np.log(1 + sigma2)
        
        kl_id = 0.5 * ( (1/d) * (1+sigma2)**(-1) * trace_sig - 1 + (1/d)*logdet_sigma - (1/d) * logdet_sig )
        kl_id_vals.append(kl_id)
        
        # 5. Calculation of the "Wrong" Constant C from the pointwise inequality
        lambda_min = (1 - np.sqrt(gamma))**2 if gamma <= 1 else 0.0
        nu_min = (lambda_min + sigma2) / (1 + sigma2)
        
        if np.isclose(nu_min, 1.0):
            C_wrong = 1.0 
        else:
            C_wrong = (-nu_min * np.log(nu_min)) / (1 - nu_min)
            
        c_wrong_bound_vals.append(C_wrong * avg_delta)

        s_delta = compute_spectral_values(gamma, sigma2)
        theory_delta.append(s_delta)      
        
        # BUG WAS HERE: Removed the indented _list appends entirely

# --- Plot ---
plt.figure(figsize=(10, 6))

color = plt.cm.tab10(2)
    
# Plotting the flat lists directly
plt.plot(alphas, avg_delta_vals, 'o', label=rf'Influence density $\mathcal{{I}}$ ($\sigma^2={sigma2}$)', markersize=4)
plt.plot(alphas, theory_delta, '-', label=rf'Spectral influence $\mathcal{{I}}$ ($\sigma^2={sigma2}$)', markersize=4)
...


plt.plot(alphas, kl_id_vals, '--', label=rf'excess KL divergence $\mathcal{{D}}$ ($\sigma^2={sigma2}$)', markersize=4)
    
plt.plot(alphas, c_wrong_bound_vals, '^:', color='red', label=rf'Failed Pointwise Bound $C \mathcal{{I}}$ ($\sigma^2={sigma2}$)', markersize=5)

plt.yscale('log')
plt.xlabel(r'$\alpha = n/d$')
plt.ylabel('Value (log scale)')
plt.title('Failure of the pointwise inequality bound')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/pointwise_bound_failure.png', dpi=300)
print("Plot saved to: /home/ceci/Thesis/Plots/pointwise_bound_failure.png")