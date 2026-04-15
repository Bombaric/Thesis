import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# --- Configurazione ---
d = 400
n_values = np.linspace(300, 3000, 40).astype(int) # Evitiamo n < d per stabilità con lambda piccoli
alphas = n_values / d 
# Generiamo molti lambda per una GIF fluida (es. scala logaritmica)
reg_list = np.logspace(-3, -0.1, 100) 

filenames = []

print("Generazione frame per la GIF...")

for i, lam in enumerate(reg_list):
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
        
        term_det_inf = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
        term_quad_inf = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)
        
        avg_delta = np.mean(term_det_inf + term_quad_inf) / d
        avg_delta_vals.append(avg_delta)
        
        # 4. KL Divergenza (Sigma_n || Id) riscalata 1/d
        trace_sig = np.trace(Sigma_n)
        _, logdet_sig = np.linalg.slogdet(Sigma_n)
        kl_id = 0.5 * ( (1/d) * trace_sig - 1 - (1/d) * logdet_sig )
        kl_id_vals.append(kl_id)

    # --- Plot del Frame ---
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, avg_delta_vals, 'o-', label=f'Influence Density $\\bar{{\Delta}}/d$', color='tab:blue')
    plt.plot(alphas, kl_id_vals, 's--', label=f'KL Divergence $\\mathcal{{D}}$', color='orange')
    
    plt.yscale('log')
    plt.ylim(1e-3, 2) # Fissiamo l'asse Y per evitare che "salti" nella GIF
    plt.xlabel(r'Sample Complexity $\alpha = n/d$')
    plt.ylabel('Value / d (Log Scale)')
    plt.title(f'Influence vs KL Divergence (Regularization $\lambda = {lam:.4f}$)')
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # Salvataggio frame temporaneo
    filename = f'frame_{i}.png'
    plt.savefig(filename, dpi=150)
    filenames.append(filename)
    plt.close()

# --- Creazione GIF ---
with imageio.get_writer('influence_evolution.gif', mode='I', duration=0.5) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Pulizia file temporanei
for filename in filenames:
    os.remove(filename)

print("GIF creata con successo: influence_evolution.gif")
