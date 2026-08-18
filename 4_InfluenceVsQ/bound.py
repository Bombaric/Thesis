import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# ── integrali MP ─────────────────────────────────────────────────────────────
def mp_moment(alpha, lam, func, limit=600):
    gc = 1.0 / alpha
    am = (1 - np.sqrt(gc))**2
    ap = (1 + np.sqrt(gc))**2
    pm = max(0.0, 1 - alpha)
    dens = lambda x: alpha * np.sqrt(max((ap-x)*(x-am), 0)) / (2*np.pi*x)
    bulk, _ = integrate.quad(
        lambda x: func((x+lam)/(1+lam)) * dens(x),
        am, ap, limit=limit, points=[am, ap]
    )
    return bulk + pm * func(lam/(1+lam))

def q_of(a, s):
    return mp_moment(a, s, lambda nu: np.sqrt(nu))**2

def inf_of(a, s):
    return 0.5 * (mp_moment(a, s, lambda nu: 1/nu) - 1)

def V_of(a, s):
    # Var(nu) esatta dalla distribuzione MP
    return mp_moment(a, s, lambda nu: (nu-1)**2)

# ── griglia ───────────────────────────────────────────────────────────────────
alphas = np.geomspace(0.01, 500, 500)
s = 0.01

q_arr   = np.array([q_of(a, s)   for a in alphas])
inf_arr = np.array([inf_of(a, s) for a in alphas])
V_arr   = np.array([V_of(a, s)   for a in alphas])

# ── plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 6))
plt.plot(alphas, q_arr,   label=r"$q$",   color="C0")
plt.plot(alphas, 1/(1+inf_arr), label=r"$1/(1+2\mathcal{I})$", color="C1")
plt.xscale("log")
plt.xlabel(r"$\alpha = n/d$")
plt.ylabel("Value") 
plt.grid(True, which="both", alpha=0.25)
plt.title(r"$q$ and $\mathcal{I}$ vs $\alpha$ for $\sigma^2 = 0.01$")
plt.legend()
plt.savefig("/home/ceci/Thesis/Plots/bound.png", dpi=300, bbox_inches="tight")
