import numpy as np
import matplotlib.pyplot as plt

# --- Configurazione ---
d = 400
n_values = np.linspace(500, 3000, 50).astype(int)
alphas = n_values / d  # n/d
#reg_list = [0.001, 0.01, 0.1]
reg_list = [0]
avg_delta_vals_list = []
kl_id_vals_list = []

for lam in reg_list:
    avg_delta_vals = []
    kl_id_vals = []
    for n in n_values:
   
        # 1. Generazione dati
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
        
        # 4. KL Divergenza (Sigma_n || Id) riscalata 1/d
        # KL = 1/2 * (1/d * Tr(Sigma) - 1 - 1/d * ln(det Sigma))
        trace_sig = np.trace(Sigma_n)
        _, logdet_sig = np.linalg.slogdet(Sigma_n)
        
        kl_id = 0.5 * ( (1/d) * trace_sig - 1 - (1/d) * logdet_sig )
        kl_id_vals.append(kl_id)
       
    avg_delta_vals_list.append(avg_delta_vals)
    kl_id_vals_list.append(kl_id_vals)
print(len(avg_delta_vals_list), len(kl_id_vals_list))
print(len(avg_delta_vals_list[0]), len(kl_id_vals_list[0]))
#print(len(avg_delta_vals_list[1]), len(kl_id_vals_list[1]))
# --- Plot ---
plt.figure(figsize=(10, 6))
approx_function = 0.5 * (1/alphas / (1 - 1/alphas))  # Approssimazione teorica per λ=0

for _ in range(len(reg_list)):
    plt.plot(alphas, avg_delta_vals_list[_], 'o-', label=f'Influence Density (λ={reg_list[_]})', markersize=3)
    plt.plot(alphas, kl_id_vals_list[_], 's--', label=f'KL Divergence (λ={reg_list[_]})', markersize=3)
plt.plot(alphas, approx_function, '--', color='black',
         label=r'Approximation: ($\lambda=0$): $\frac{1}{2} \frac{\gamma}{1-\gamma}$')
plt.yscale('log')
plt.xlabel(r'Sample Complexity $\alpha = 1/\gamma = n/d$')
plt.ylabel('$\Delta$ / d (Log Scale)')
plt.title('Influence Function vs. KL Divergence to Identity')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('approximation_gamma.png', dpi=150)