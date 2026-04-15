import numpy as np
import matplotlib.pyplot as plt

# 1. Configurazione per il limite termodinamico
d = 1000  
n_values = np.linspace(100, 3000, 60).astype(int)
gammas = n_values / d
lambdas = [0.01, 0.1] # Blu per regolarizzazione bassa, Arancio per alta

fig, ax = plt.subplots(figsize=(12, 8))

for lam in lambdas:
    delta_avg_vals = []
    kl_vals = []
    
    for n in n_values:
        # --- STEP 1: Estrazione n campioni puri N(0, Id) ---
        X = np.random.randn(n, d) 
        
        # --- STEP 2: Matrice di scatter empirica Sigma_hat ---
        # Normalizzata per n per stabilità nel limite termodinamico
        A = (X.T @ X) / n 
        Sigma_hat = A + lam * np.eye(d) 
        
        # --- STEP 3: Media dell'Influenza (Expected Leave-One-Out) ---
        # Invece di un solo punto, mediamo su un batch per pulire il grafico
        n_batch = min(n, 100) 
        X_batch = X[:n_batch].T # Prendiamo i primi 100 punti, dimensione (d, n_batch)
        
        # Calcolo di g per tutti i punti del batch contemporaneamente (Vettorizzato)
        # g_all = (1/n) * diag(X_batch.T @ Sigma_hat^-1 @ X_batch)
        g_all = np.sum(X_batch * np.linalg.solve(Sigma_hat, X_batch), axis=0) / n
        delta_all = 0.5 * np.abs(np.log(1 - g_all) + (g_all / (1 - g_all)))
        # Applichiamo la formula dell'influenza a tutto il batch
        delta_avg_vals.append(np.mean(np.abs(delta_all))) # Media del batch per stabilità
        
        # --- STEP 4: KL Divergenza Esatta (Diretta: Modello || Verità) ---
        _, logdet = np.linalg.slogdet(Sigma_hat)
        tr_sigma = np.trace(Sigma_hat) / d
        kl = 0.5 * (tr_sigma - 1 - logdet / d)
        kl_vals.append(kl)

    # --- Plotting ---
    color = 'tab:blue' if lam == 0.01 else 'tab:orange'
    
    # Influenza (Media del batch)
    ax.plot(gammas, delta_avg_vals, '-', linewidth=2, color=color, 
            label=f'Expected Influence |Δ| (λ={lam})')
    
    # KL Divergenza
    ax.plot(gammas, kl_vals, '--', color=color, alpha=0.6, 
            label=f'KL Divergence (λ={lam})')

# --- Estetica Finale ---
ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.8, label='Critical Point n=d')
ax.set_yscale('log') 
ax.set_xlabel(r'Sample Complexity $\gamma = n/d$', fontsize=13)
ax.set_ylabel('Value (Log Scale)', fontsize=13)

# Legenda pulita
ax.legend(loc='upper right', frameon=True, fontsize=10)
plt.title(r'Phase Transition: Expected Influence vs. KL Divergence ($d=1000$)', fontsize=15)
plt.grid(True, which="both", ls="-", alpha=0.15)
plt.tight_layout()

# Salvataggio
plt.savefig('final_comparison_averaged.png', dpi=300)
plt.show()