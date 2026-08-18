import numpy as np
import matplotlib.pyplot as plt

# --- Configurazione ---
d = 400
n_values = np.linspace(10, 3000, 50).astype(int)
alphas = n_values / d  # n/d

# Regularization level chosen to clearly show the curve crossing
lam = 0.4

avg_delta_vals = []
naive_kl_vals = []
excess_kl_vals = []

for n in n_values:
    # 1. Generazione dati (True covariance is I_d)
    X = np.random.randn(n, d)
    
    # 2. Covarianza empirica regolarizzata
    Sigma_n = (X.T @ X) / n + lam * np.eye(d)
    
    # 3. Influence Function Media (Riscalata 1/d)
    V = np.linalg.solve(Sigma_n, X.T)
    h_vec = np.sum(X * V.T, axis=1) / n
    r = n / (n - 1)
    
    # Formula esatta della Delta (puntuale)
    term_det_inf = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
    term_quad_inf = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)
    
    # Media delle Delta e riscalamento per d
    avg_delta = np.mean(term_det_inf + term_quad_inf) / d
    avg_delta_vals.append(avg_delta)
    
    # 4. Calcolo Trace e LogDet per le divergenze KL
    trace_sig = np.trace(Sigma_n)
    _, logdet_sig = np.linalg.slogdet(Sigma_n)
    
    # --- Naive KL Divergence ---
    # Come da immagine formula: KL(N(0, Sigma_n) || N(0, I_d))
    # D = 1/2 * (1/d * Tr(Sigma_n) - 1 - 1/d * ln(det Sigma_n))
    naive_kl = 0.5 * ( (1/d) * trace_sig - 1 - (1/d) * logdet_sig )
    naive_kl_vals.append(naive_kl)
    
    # --- Excess KL Divergence ---
    # Target regolarizzato: Sigma* + \sigma^2 I_d = (1 + lam) * I_d
    # KL(N(0, Sigma_n) || N(0, (1+lam)I_d)) 
    excess_kl = 0.5 * ( (1/d) * (trace_sig / (1 + lam)) - 1 + np.log(1 + lam) - (1/d) * logdet_sig )
    excess_kl_vals.append(excess_kl)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Sinistra: Naive KL vs Influence Density
ax1.plot(alphas, avg_delta_vals, 'o-', label=r'Influence Density ($\mathcal{I}$)', color='tab:blue', markersize=4)
ax1.plot(alphas, naive_kl_vals, 's--', label=r'Naive KL ($\mathcal{D}$)', color='tab:orange', markersize=4)
ax1.set_yscale('log')
ax1.set_xlabel(r'Sample Complexity $\alpha = n/d$')
ax1.set_ylabel('Value (log scale)')
ax1.set_title(rf'Naive KL vs. Influence Density ($\sigma^2={lam}$)')
ax1.legend()
ax1.grid(True, which="both", linestyle='--', alpha=0.4)

# Destra: Excess KL vs Influence Density
ax2.plot(alphas, avg_delta_vals, 'o-', label=r'Influence Density ($\mathcal{I}$)', color='tab:blue', markersize=4)
ax2.plot(alphas, excess_kl_vals, 's--', label=r'Excess KL', color='tab:green', markersize=4)
ax2.set_yscale('log')
ax2.set_xlabel(r'Sample Complexity $\alpha = n/d$')
ax2.set_title(rf'Excess KL vs. Influence Density ($\sigma^2={lam}$)')
ax2.legend()
ax2.grid(True, which="both", linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/asymptotic_behavior_double.png', dpi=300)