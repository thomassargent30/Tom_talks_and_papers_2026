"""
Generate figures for the Morris (1996) Beamer slides.
Outputs PDF files for inclusion in LaTeX.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Use LaTeX-style fonts
rc('text', usetex=True)
rc('font', family='serif', size=12)

# ── Helper functions ──────────────────────────────────────────────

def posterior_mean(a, b, s, t):
    return (a + s) / (a + b + t)

def perpetuity_value(a, b, s, t, beta):
    return (beta / (1 - beta)) * posterior_mean(a, b, s, t)

def price_learning_two_agents(prior1, prior2, beta=0.75, T=200):
    a1, b1 = prior1
    a2, b2 = prior2
    price_array = np.zeros((T+1, T+1))

    for s in range(T+1):
        perp1 = perpetuity_value(a1, b1, s, T, beta)
        perp2 = perpetuity_value(a2, b2, s, T, beta)
        price_array[s, T] = max(perp1, perp2)

    for t in range(T-1, -1, -1):
        for s in range(t, -1, -1):
            mu1 = posterior_mean(a1, b1, s, t)
            mu2 = posterior_mean(a2, b2, s, t)
            cont1 = mu1 * (1.0 + price_array[s+1, t+1]) \
                    + (1.0 - mu1) * price_array[s, t+1]
            cont2 = mu2 * (1.0 + price_array[s+1, t+1]) \
                    + (1.0 - mu2) * price_array[s, t+1]
            price_array[s, t] = beta * max(cont1, cont2)

    return price_array

def normalized_price_two_agents(prior1, prior2, r, T=250):
    beta = 1.0 / (1.0 + r)
    price_array = price_learning_two_agents(prior1, prior2, beta=beta, T=T)
    return r * price_array

# ── Figure 1: Normalized price vs. interest rate ─────────────────

priors = ((1,1), (0.5,0.5))
r_grid = np.linspace(1e-3, 5.0, 200)
p00 = np.array([normalized_price_two_agents(
                priors[0], priors[1], r, T=300)[0,0]
                for r in r_grid])

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(r_grid, p00, lw=2.2, color='C0')
ax.axhline(0.5, color='C1', linestyle='--', lw=1.5,
           label=r'Fundamental $= 0.5$')
ax.set_xlabel(r'Interest rate $r$', fontsize=13)
ax.set_ylabel(r'$p^*(0,0,r)$', fontsize=13)
ax.legend(fontsize=11, loc='upper right')
ax.set_title(r'Normalized price at $(s,t)=(0,0)$', fontsize=13)
fig.tight_layout()
fig.savefig('morris_fig1_price_vs_r.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved morris_fig1_price_vs_r.pdf")

# ── Figure 2: Normalized price vs. time ──────────────────────────

r = 0.05
T = 60
p_mat = normalized_price_two_agents(priors[0], priors[1], r, T=T)
t_vals = np.arange(0, 54, 2)
s_vals = t_vals // 2
y = np.array([p_mat[s, t] for s, t in zip(s_vals, t_vals)])

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(t_vals, y, lw=2.2, color='C0')
ax.axhline(0.5, color='C1', linestyle='--', lw=1.5,
           label=r'Fundamental $= 0.5$')
ax.set_xlabel(r'$t$', fontsize=13)
ax.set_ylabel(r'$p^*(t/2,\, t,\, 0.05)$', fontsize=13)
ax.legend(fontsize=11, loc='upper right')
ax.set_title(r'Normalized price along symmetric path $s=t/2$', fontsize=13)
fig.tight_layout()
fig.savefig('morris_fig2_price_vs_t.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved morris_fig2_price_vs_t.pdf")

# ── Figure 3: Beta prior densities ───────────────────────────────

from scipy.stats import beta as beta_dist

theta = np.linspace(0.001, 0.999, 500)

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for (a, b), label, color in [
    ((1, 1), r'Beta$(1,1)$  (uniform)', 'C0'),
    ((0.5, 0.5), r'Beta$(1/2,1/2)$  (Jeffreys)', 'C1'),
    ((2, 1), r'Beta$(2,1)$', 'C2'),
    ((1, 2), r'Beta$(1,2)$', 'C3'),
]:
    ax.plot(theta, beta_dist.pdf(theta, a, b), lw=2, label=label, color=color)

ax.set_xlabel(r'$\theta$', fontsize=13)
ax.set_ylabel('Density', fontsize=13)
ax.set_title('Prior densities over dividend probability', fontsize=13)
ax.legend(fontsize=10, loc='upper center')
ax.set_ylim(0, 3.5)
fig.tight_layout()
fig.savefig('morris_fig3_beta_priors.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved morris_fig3_beta_priors.pdf")

# ── Figure 4: Posterior means crossing ────────────────────────────

t_range = np.arange(0, 40)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

# Panel (a): Rate-dominant case – no crossing
a1, b1, a2, b2 = 2, 1, 1, 2
for frac, ls, lbl in [(0.3, '-', r'$s=0.3t$'),
                       (0.5, '--', r'$s=0.5t$'),
                       (0.7, ':', r'$s=0.7t$')]:
    s_arr = np.round(frac * t_range).astype(int)
    mu1 = np.array([posterior_mean(a1, b1, s, t) for s, t in zip(s_arr, t_range)])
    mu2 = np.array([posterior_mean(a2, b2, s, t) for s, t in zip(s_arr, t_range)])
    axes[0].plot(t_range, mu1 - mu2, lw=2, ls=ls, label=lbl)

axes[0].axhline(0, color='gray', lw=0.8)
axes[0].set_xlabel(r'$t$', fontsize=12)
axes[0].set_ylabel(r'$\mu_1 - \mu_2$', fontsize=12)
axes[0].set_title(r'(a) Rate dominance: Beta$(2,1)$ vs Beta$(1,2)$', fontsize=11)
axes[0].legend(fontsize=9)

# Panel (b): Crossing case
a1, b1, a2, b2 = 1, 1, 0.5, 0.5
for frac, ls, lbl in [(0.3, '-', r'$s=0.3t$'),
                       (0.5, '--', r'$s=0.5t$'),
                       (0.7, ':', r'$s=0.7t$')]:
    s_arr = np.round(frac * t_range).astype(int)
    mu1 = np.array([posterior_mean(a1, b1, s, t) for s, t in zip(s_arr, t_range)])
    mu2 = np.array([posterior_mean(a2, b2, s, t) for s, t in zip(s_arr, t_range)])
    axes[1].plot(t_range, mu1 - mu2, lw=2, ls=ls, label=lbl)

axes[1].axhline(0, color='gray', lw=0.8)
axes[1].set_xlabel(r'$t$', fontsize=12)
axes[1].set_ylabel(r'$\mu_1 - \mu_2$', fontsize=12)
axes[1].set_title(r'(b) Crossing: Beta$(1,1)$ vs Beta$(1/2,1/2)$', fontsize=11)
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig('morris_fig4_posterior_crossing.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved morris_fig4_posterior_crossing.pdf")

# ── Figure 5: Speculative premium heat map ────────────────────────

beta_val = 0.75
T_heat = 30
a1, b1 = 1, 1
a2, b2 = 0.5, 0.5
price_heat = price_learning_two_agents((a1, b1), (a2, b2),
                                        beta=beta_val, T=T_heat)

premium = np.full((T_heat+1, T_heat+1), np.nan)
for t in range(T_heat+1):
    for s in range(t+1):
        mu_star = max(posterior_mean(a1, b1, s, t),
                      posterior_mean(a2, b2, s, t))
        perp_star = (beta_val / (1 - beta_val)) * mu_star
        premium[s, t] = price_heat[s, t] - perp_star

fig, ax = plt.subplots(figsize=(6, 4.5))
# Only show lower-triangular region
masked = np.ma.masked_invalid(premium)
im = ax.pcolormesh(masked, cmap='YlOrRd', shading='auto')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Speculative premium', fontsize=11)
ax.set_xlabel(r'$t$ (periods)', fontsize=12)
ax.set_ylabel(r'$s$ (successes)', fontsize=12)
ax.set_title('Speculative premium over most-optimistic fundamental', fontsize=11)
fig.tight_layout()
fig.savefig('morris_fig5_premium_heat.pdf', bbox_inches='tight')
plt.close(fig)
print("Saved morris_fig5_premium_heat.pdf")

print("\nAll figures generated successfully.")
