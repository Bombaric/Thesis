import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def analytical_q_factor(alpha, lam):
    """
    Computes the convergence overlap q using the theoretical Marchenko-Pastur expectation.
    """
    gamma = 1.0 / alpha
    lam_minus = (1.0 - np.sqrt(gamma))**2
    lam_plus = (1.0 + np.sqrt(gamma))**2
    
    def integrand(x):
        radicand = np.clip((lam_plus - x) * (x - lam_minus), 0, None)
        density = np.sqrt(radicand) / (2 * np.pi * gamma * x)
        return np.sqrt(x + lam) * density

    bulk_integral, _ = integrate.quad(integrand, lam_minus, lam_plus, limit=200)
    
    # Handle the topological point mass in the underparameterized regime
    point_mass = (1.0 - 1.0 / gamma) * np.sqrt(lam) if gamma > 1.0 else 0.0
        
    E_sqrt = bulk_integral + point_mass
    return (1.0 / (1.0 + lam)) * (E_sqrt**2)

# --- Configuration ---
d = 400
n_values = np.linspace(10, 3000, 60).astype(int)
alphas = n_values / d
lam = 0.01  # Fixed regularization



inf_vals = []
q_vals = []
inv_q_vals = []
variances = []
various = []

print("Running empirical matrix inversions. This may take a moment...")

for i, n in enumerate(n_values):
    # 1. Empirical Influence Function (Matrix inversion)
    X = np.random.randn(n, d)
    # Using np.dot to prevent syntax errors on older Python versions
    Sigma_n = np.dot(X.T, X) / n + lam * np.eye(d)
    
    V = np.linalg.solve(Sigma_n, X.T)
    h_vec = np.sum(X * V.T, axis=1) / n
    r = n / (n - 1)
    
    term_det = 0.5 * (np.log(1 - h_vec) + d * np.log(r))
    term_quad = 0.5 * (h_vec * (n * h_vec - 1)) / (1 - h_vec)
    
    # THE FIX: np.mean() squashes the array of size 'n' into a single scalar metric
    emp_inf = np.mean(term_det + term_quad) / d
    
    # 2. Theoretical Convergence Overlap
    q = analytical_q_factor(alphas[i], lam)

    variance = 1/alphas[i] * 1 / (1 + lam)**2  # Variance of nu in the underparameterized regime
    
    inf_vals.append(emp_inf)
   # q_vals.append(q)
    #inv_q_vals.append(q)
   # variances.append(1.0-0.25 * variance)
    various.append(0.5 *variance)


# Now they are perfectly homogeneous lists of scalars
inf_vals = np.array(inf_vals)
q_vals = np.array(q_vals)
inv_q_vals = np.array(inv_q_vals)
variances = np.array(variances)
various = np.array(various)

# --- Plotting ---
fig = plt.figure(figsize=(12, 6))

color_inf = '#d62728'  # Red
color_q = '#1f77b4'    # Blue
color_inv_q = '#ff7f0e' # Orange
color_var = '#2ca02c'  # Green


# 1. Left Plot: Metrics vs Sample Complexity
plt.plot(alphas, inf_vals, 'o-', color=color_inf, markersize=4, label=r'influence function $\Delta/d$')
#plt.plot(alphas, q_vals, '--', color=color_q, linewidth=2, label=f'q')
#plt.plot(alphas, variances, '-.', color=color_inv_q, linewidth=2, label=r'$ 1 - 1/4 var(\nu)$')
plt.plot(alphas, various, ':', color=color_var, linewidth=2, label=r'$1/2 var(\nu)$')
#plt.yscale('log')
plt.xlabel(r'Sample Complexity $\alpha = n/d$')
plt.ylabel('Value')
plt.title(r'Influence vs Var(\nu) ($\sigma^2=0.01$)')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.4)


plt.tight_layout()
plt.savefig('/home/ceci/Thesis/Plots/influence_vs_variance_nu_nonlog.png', dpi=300)
print("Plot generated successfully!")