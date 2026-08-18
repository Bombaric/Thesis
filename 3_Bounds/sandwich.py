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
d = 400
n_values = np.linspace(20, 3000, 50).astype(int)
alphas = n_values / d  # n/d
#reg_list = [0.0, 0.01, 0.1] # Lista di regolarizzazioni da testare
reg_list = [0.01]

avg_delta_vals_list = []
kl_id_vals_list = []
c_bound_vals_list = [] # Updated list for the rigorous C_gamma bound
#C non rigoroso
theory_delta = [compute_spectral_values(d/n, lam) for lam in reg_list for n in n_values]
print("Generazione dati e calcolo dei bound rigorosi...")

for lam in reg_list:
    avg_delta_vals = []
    kl_id_vals = []
    c_bound_vals = []
    c_non_rigorous_vals = []

    for n in n_values:
        gamma = d / n # Aspect ratio gamma
        
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
        
        # 4. KL Divergenza SHIFTATA (Correzione applicata)
        trace_sig = np.trace(Sigma_n)
        _, logdet_sig = np.linalg.slogdet(Sigma_n)
        logdet_sigma = d * np.log(1 + lam)
        
        kl_id = 0.5 * ( (1/d) * (1+lam)**(-1) * trace_sig - 1 + (1/d)*logdet_sigma - (1/d) * logdet_sig )
        kl_id_vals.append(kl_id)
        
        # 5. Calcolo del Lower Bound Rigoroso (C_gamma * Delta/d)
        # Bordo inferiore dello spettro di Marchenko-Pastur
        #lambda_min = (1 - np.sqrt(gamma))**2 if gamma <= 1 else 0.0
        lambda_min = 0
        # Autovalore riscalato minimo, diverso da zero con regolarizzazione!!
        nu_min = (lambda_min + lam) / (1 + lam)
        
        # Costante di bound C_gamma (Rigorosa)
        if np.isclose(nu_min, 1.0):
            C_gamma = 0.5 # Il limite asintotico esatto di Taylor
        else:
            num = nu_min * (nu_min - 1 - np.log(nu_min))
            den = (nu_min - 1)**2
            C_gamma = num / den
            
        c_bound_vals.append(C_gamma * avg_delta)
        
    avg_delta_vals_list.append(avg_delta_vals)
    kl_id_vals_list.append(kl_id_vals)
    c_bound_vals_list.append(c_bound_vals)

# --- Plot ---
plt.figure(figsize=(10, 6))

for i, lam in enumerate(reg_list):
    color = plt.cm.tab10(i)
    
    # Plot delle curve principali
    plt.plot(alphas, theory_delta[i*len(n_values):(i+1)*len(n_values)], '-', label=rf'Spectral influence $\mathcal{{I}}$ ($\sigma^2$={lam})', markersize=4)
    plt.plot(alphas, avg_delta_vals_list[i], 'o', label=rf'Influence density $\mathcal{{I}}$ ($\sigma^2$={lam})', markersize=4)
    plt.plot(alphas, kl_id_vals_list[i], '--', label=rf'excess KL divergence $\mathcal{{D}}$ ($\sigma^2$={lam})', markersize=4)
    
    # Plot del lower bound analitico rigoroso
    plt.plot(alphas, c_bound_vals_list[i], '^:', label=rf'Lower Bound $C \mathcal{{I}}$ ($\sigma^2$={lam})', markersize=4)
    
    # Shading dell'area del teorema "Sandwich"
    #plt.fill_between(alphas, c_bound_vals_list[i], avg_delta_vals_list[i], color=color, alpha=0.1)

plt.yscale('log')
plt.xlabel(r'Sample Complexity $\alpha = n/d$')
plt.ylabel('Value (Log Scale)')
plt.title(r'Rigorous sandwich bound: $\mathcal{I}$, $\mathcal{D}$, and $C \mathcal{I}$')
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/asymptotic_sandwich_bound_rigorous.png', dpi=300)
print("Plot salvato in: /home/ceci/Thesis/Plots/asymptotic_sandwich_bound_rigorous.png")