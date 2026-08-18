import numpy as np
import matplotlib.pyplot as plt

# --- Configurazione ---
d = 400
n_values = np.linspace(401, 3000, 150).astype(int)
alphas = n_values / d      # n/d
gammas = d / n_values      # d/n
reg_list = [0]

avg_delta_vals_list = []

for lam in reg_list:
    avg_delta_vals = []
    for n in n_values:
        # 1. Generazione dati
        X = np.random.randn(n, d)

        # 2. Covarianza empirica regolarizzata
        Sigma_n = (X.T @ X) / n + lam * np.eye(d)

        # 3. Influence Function Media (Riscalata 1/d)
        V = np.linalg.solve(Sigma_n, X.T)
        h_vec = np.sum(X * V.T, axis=1) / n
        r = n / (n - 1)

        term_det_inf = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
        term_quad_inf = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)

        avg_delta = np.mean(term_det_inf + term_quad_inf) / d
        avg_delta_vals.append(avg_delta)

    avg_delta_vals_list.append(avg_delta_vals)

approx_function_gamma = 0.5 * (gammas / (1 - gammas))  # in terms of gamma, gamma<1
approx_function_alpha = 0.5 * (1 / alphas / (1 - 1 / alphas))  # equivalent, in terms of alpha

# --- Plot 1: in funzione di gamma = d/n ---
plt.figure(figsize=(10, 6))
for i, lam in enumerate(reg_list):
    plt.plot(gammas, avg_delta_vals_list[i], 'o', color='red',
             label=fr'Influence density ($\sigma^2$={lam})', markersize=3)
plt.plot(gammas, approx_function_gamma, '--', color='black',
         label=r'asymptotic limit ($\sigma^2=0$): $\frac{1}{2}\frac{\gamma}{1-\gamma}$')
plt.yscale('log')
plt.xlabel(r'$\gamma = d/n$')
plt.ylabel(r'$\mathcal{I}$ (Log Scale)')
plt.title(r'Influence density vs spectral asymptotic limit ($\sigma^2=0$) — vs $\gamma$')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/approximation_gamma.png', dpi=150)
plt.close()

# --- Plot 2: in funzione di alpha = n/d ---
plt.figure(figsize=(10, 6))
for i, lam in enumerate(reg_list):
    plt.plot(alphas, avg_delta_vals_list[i], 'o', color='red',
             label=fr'Influence density ($\sigma^2$={lam})', markersize=3)
plt.plot(alphas, approx_function_alpha, '--', color='black',
         label=r'asymptotic limit ($\sigma^2=0$): $\frac{1}{2(\alpha-1)}$')
plt.yscale('log')
plt.xlabel(r'$\alpha = n/d$')
plt.ylabel(r'$\mathcal{I}$ (Log Scale)')
plt.title(r'Influence density vs spectral asymptotic limit ($\sigma^2=0$) — vs $\alpha$')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/approximation_alpha.png', dpi=150)
plt.close()