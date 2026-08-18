import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

def theoretical_c_alpha(alphas, lam):
    """
    Computes the rigorous theoretical bound C as a function of alpha.
    """
    gamma = 1.0 / alphas
    
    # 1. Bordo inferiore dello spettro di Marchenko-Pastur
    lambda_min = np.where(gamma <= 1.0, (1.0 - np.sqrt(gamma))**2, 0.0)
    
    # 2. Autovalore riscalato minimo
    nu_min = (lambda_min + lam) / (1.0 + lam)
    
    # 3. Costante di bound rigorosa (vettorializzata)
    C_val = np.zeros_like(nu_min)
    
    # Gestione sicura del limite di Taylor (evita divisione per zero)
    is_one = np.isclose(nu_min, 1.0)
    not_one = ~is_one
    
    C_val[is_one] = 0.5 
    
    nu = nu_min[not_one]
    C_val[not_one] = (nu * (nu - 1.0 - np.log(nu))) / ((nu - 1.0)**2)
    
    return C_val

# --- Configurazione ---
d = 400
n_values = np.linspace(300, 3000, 50).astype(int)
alphas = n_values / d  # n/d

# Creiamo un range logaritmico per lambda: da 0.001 a 0.1 (30 frame)
reg_list = np.logspace(-3, -1, 30)

print(f"Precalcolo dei dati per {len(reg_list)} frame (questo richiederà un minuto)...")

precomputed_data = []
global_min = float('inf')
global_max = float('-inf')

for lam in reg_list:
    avg_delta_vals = []
    kl_id_vals = []
    
    for n in n_values:
        # 1. Generazione dati empirici
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
        
        # 4. KL Divergenza SHIFTATA 
        trace_sig = np.trace(Sigma_n)
        _, logdet_sig = np.linalg.slogdet(Sigma_n)
        logdet_sigma = d * np.log(1 + lam)
        
        kl_id = 0.5 * ( (1/d) * (1+lam)**(-1) * trace_sig - 1 + (1/d)*logdet_sigma - (1/d) * logdet_sig )
        kl_id_vals.append(kl_id)
        
    avg_delta_vals = np.array(avg_delta_vals)
    kl_id_vals = np.array(kl_id_vals)
    
    # Calcolo Teorico
    C_alpha_theoretical = theoretical_c_alpha(alphas, lam)
    theoretical_lower_bound = C_alpha_theoretical * avg_delta_vals
    
    # Aggiorna i limiti globali per bloccare l'asse Y
    current_min = np.min(theoretical_lower_bound)
    current_max = np.max(avg_delta_vals)
    if current_min < global_min: global_min = current_min
    if current_max > global_max: global_max = current_max
    
    precomputed_data.append({
        'lam': lam,
        'inf': avg_delta_vals,
        'kl': kl_id_vals,
        'bound': theoretical_lower_bound
    })

print("Precalcolo completato. Generazione dell'animazione in corso...")

# --- Animazione ---
fig, ax = plt.subplots(figsize=(10, 6))

def update(frame_idx):
    ax.clear() # Pulisce il frame precedente per permettere allo shading di aggiornarsi
    
    data = precomputed_data[frame_idx]
    lam = data['lam']
    
    # Colori scelti per coerenza
    color_inf = '#1f77b4'  # Blu
    color_kl = '#ff7f0e'   # Arancione
    color_bound = '#2ca02c' # Verde
    
    # Plot delle tre curve
    ax.plot(alphas, data['inf'], 'o-', color=color_inf, label=r'Influence Density $\bar{\Delta}/d$', markersize=4)
    ax.plot(alphas, data['kl'], 's--', color=color_kl, label=r'KL Divergence $\mathcal{D}$', markersize=4)
    ax.plot(alphas, data['bound'], '^:', color=color_bound, label=r'Lower Bound $C(\alpha) \frac{\bar{\Delta}}{d}$', markersize=4)
    
    # Shading dell'area del teorema "Sandwich"
    ax.fill_between(alphas, data['bound'], data['inf'], color='gray', alpha=0.15)
    
    # Formattazione assi (deve essere riapplicata ad ogni frame a causa del clear())
    ax.set_yscale('log')
    # Blocchiamo l'asse Y aggiungendo un piccolo margine (10%) sopra e sotto
    ax.set_ylim(global_min * 0.8, global_max * 1.2) 
    
    ax.set_xlabel(r'Sample Complexity $\alpha = n/d$')
    ax.set_ylabel('Value (Log Scale)')
    ax.set_title(f'Rigorous Sandwich Bound Evolution\nRegularization $\\lambda = {lam:.4f}$')
    ax.legend(loc='upper right')
    ax.grid(True, which="both", linestyle='--', alpha=0.4)

# Creazione dell'animazione
ani = FuncAnimation(fig, update, frames=len(reg_list), interval=150) # interval = millisecondi tra i frame

# Salvataggio
output_path = "/home/ceci/Thesis/Plots/sandwich_evolution.gif"
ani.save(output_path, writer=PillowWriter(fps=6))

print(f"GIF salvata con successo in: {output_path}")