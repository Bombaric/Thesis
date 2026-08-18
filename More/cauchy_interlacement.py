import numpy as np
import matplotlib.pyplot as plt

def marchenko_pastur_pdf(x, gamma, sigma=1.0):
    """Theoretical MP Density."""
    lambda_plus = sigma**2 * (1 + np.sqrt(gamma))**2
    lambda_minus = sigma**2 * (1 - np.sqrt(gamma))**2
    
    # Calculate PDF only within support
    with np.errstate(invalid='ignore'):
        pdf = (1/(2 * np.pi * sigma**2 * gamma * x)) * \
              np.sqrt(np.maximum(0, (lambda_plus - x) * (x - lambda_minus)))
    return np.nan_to_num(pdf)

# --- Parameters ---
n = 1000        # Samples
d = 400       # Dimensions
gamma = d/n     # Aspect ratio
n_trials = 50   # Number of matrices to simulate
sigma = 1.0     # Variance of X

all_eig_Sn = []
all_eig_Sn_minus = []

# --- 1. Thermodynamic Limit Collection ---
for _ in range(n_trials):
    X = np.random.normal(0, sigma, (n, d))
    
    # S_n
    Sn = (X.T @ X) / n 
    all_eig_Sn.extend(np.linalg.eigvalsh(Sn))
    
    # S_{n-1} (leave one out)
    # Using n-1 here for the local scaling to keep densities aligned
    X_minus = X[1:, :]
    Sn_minus = (X_minus.T @ X_minus) / (n-1) 
    all_eig_Sn_minus.extend(np.linalg.eigvalsh(Sn_minus))

# --- 2. Plotting the Bulk Distributions ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x_range = np.linspace(0.1, 3.5, 1000)
theoretical_pdf = marchenko_pastur_pdf(x_range, gamma, sigma)

# Plot for Sn
axes[0].hist(all_eig_Sn, bins=80, density=True, alpha=0.5, color='tomato', label='Empirical $S_n$')
axes[0].plot(x_range, theoretical_pdf, 'k-', lw=2, label='Theoretical MP')
axes[0].set_title(f"Bulk Distribution $S_n$ ($\gamma={gamma}$)")
axes[0].legend()

# Plot for Sn-1
# Note: gamma for Sn-1 is d/(n-1), practically identical for large n
axes[1].hist(all_eig_Sn_minus, bins=80, density=True, alpha=0.5, color='royalblue', label='Empirical $S_{n-1}$')
axes[1].plot(x_range, theoretical_pdf, 'k-', lw=2, label='Theoretical MP')
axes[1].set_title(f"Bulk Distribution $S_{n-1}$ ($\gamma \\approx {gamma}$)")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/ceci/Thesis/Plots/cauchy_interlacement_bulk.png", dpi=300)

# --- 3. Zooming in on Interlacement (Single Instance) ---
# Use smaller d for visual clarity
d_small, n_small = 20, 50
X_single = np.random.normal(0, 1, (n_small, d_small))
e_Sn = np.sort(np.linalg.eigvalsh((X_single.T @ X_single) / n_small))
e_Sn_minus = np.sort(np.linalg.eigvalsh((X_single[1:, :].T @ X_single[1:, :]) / n_small))

plt.figure(figsize=(12, 3))
plt.hlines(1, e_Sn_minus.min(), e_Sn.max(), colors='gray', alpha=0.3)
plt.scatter(e_Sn_minus, np.ones_like(e_Sn_minus), color='blue', marker='|', s=500, label='$\mu_i$ ($S_{n-1}$)')
plt.scatter(e_Sn, np.ones_like(e_Sn), color='red', marker='o', s=50, label='$\lambda_i$ ($S_n$)', facecolors='none')
plt.title("Deterministic Interlacement Snapshot (Zoomed)")
plt.yticks([])
plt.legend()
plt.savefig("/home/ceci/Thesis/Plots/cauchy_interlacement.png", dpi=300)