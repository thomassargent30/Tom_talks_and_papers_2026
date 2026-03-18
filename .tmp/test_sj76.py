import numpy as np

def compute_covariances(alpha, lam, sigma_u2, sigma_v2, sigma_eu):
    phi = 1.0 / (1.0 + alpha * (1.0 - lam))
    c0 = ((phi*(1-lam) + 1)**2 * sigma_u2
          + sigma_v2
          + phi**2 * (1-lam)**2 * sigma_u2
          - 2*(phi*(1-lam)+1)*phi*(1-lam)*sigma_eu)
    c1 = -(phi*(1-lam) + 1) * sigma_u2
    g0 = (phi*(phi*(1-lam)+1) + phi)*sigma_u2 \
         + phi**2*(1-lam)*sigma_v2 \
         - (phi**2*(1-lam) + phi*(phi*(1-lam)+1))*sigma_eu
    g1 = (-phi * (lam + alpha*(1-lam)) / (1 + alpha*(1-lam))
          * (phi*(1-lam)+1) * sigma_u2
          + phi**2*lam*(1-lam)*sigma_v2
          + (phi*(lam + alpha*(1-lam))/(1+alpha*(1-lam))
             * (phi*(1-lam)+1)
             + phi**2*lam*(1-lam)) * sigma_eu)
    return c0, c1, g0, g1


def spectral_factor(c0, c1):
    if abs(c1) < 1e-14:
        return np.sqrt(c0), 0.0
    ratio = c0 / c1
    disc = ratio**2 - 4.0
    if disc < 0:
        raise ValueError("Spectral density not non-negative definite.")
    roots = [(ratio + np.sqrt(disc)) / 2.0, (ratio - np.sqrt(disc)) / 2.0]
    b = min(roots, key=abs)
    b0_sq = c1 / b
    b0 = np.sqrt(abs(b0_sq))
    return b0, b


def wiener_kolmogorov_projection(c0, c1, g0, g1):
    b0, b = spectral_factor(c0, c1)
    b0_sq = b0**2
    A = g0 - g1 * b
    B = g1
    return A, B, b0, b


def compute_h_coefficients(c0, c1, g0, g1):
    A, B, b0, b = wiener_kolmogorov_projection(c0, c1, g0, g1)
    b0_sq = b0**2
    h0 = (b0_sq - A) / b0_sq
    h1 = (b0_sq * b - B) / b0_sq
    h2 = b
    return h0, h1, h2


def jacobs_population_params(alpha, lam, sigma_u2, sigma_v2, sigma_eu):
    c0, c1, g0, g1 = compute_covariances(alpha, lam, sigma_u2, sigma_v2, sigma_eu)
    h0, h1, h2 = compute_h_coefficients(c0, c1, g0, g1)
    y1 = 1.0
    y0 = h0 / (1.0 - h2)
    return y0, y1, h0, h1, h2


# Test over a grid
alphas = np.linspace(-1.5, -0.1, 8)
lambdas = np.linspace(0.1, 0.9, 8)
sigma_u2, sigma_v2, sigma_eu = 1.0, 0.5, 0.1

results = []
errors = []
for a in alphas:
    for l in lambdas:
        try:
            y0, y1, h0, h1, h2 = jacobs_population_params(a, l, sigma_u2, sigma_v2, sigma_eu)
            results.append({'alpha': a, 'lambda': l, 'y0': y0, 'y1': y1,
                            'h0': h0, 'h1': h1, 'h2': h2})
        except Exception as e:
            errors.append(f"alpha={a:.2f}, lam={l:.2f}: {e}")

print(f"Successful: {len(results)}/64")
if errors:
    print("Errors:", errors)
y1_vals = [r['y1'] for r in results]
print(f"Range of y1: [{min(y1_vals):.8f}, {max(y1_vals):.8f}]")
print(f"Max deviation from 1: {max(abs(v - 1) for v in y1_vals):.2e}")

# Check |b| < 1 (fundamental factorization) for all cases
b_vals = [r['h2'] for r in results]
print(f"\nRange of |b|=h2: [{min(abs(v) for v in b_vals):.4f}, {max(abs(v) for v in b_vals):.4f}]")
print("All |b| < 1:", all(abs(v) < 1 for v in b_vals))

# Also check spectral density of M is positive
print("\nChecking spectral density positivity:")
for r in results[:5]:
    a, l = r['alpha'], r['lambda']
    c0, c1, g0, g1 = compute_covariances(a, l, sigma_u2, sigma_v2, sigma_eu)
    omega_test = np.linspace(0, np.pi, 200)
    Cw = c1 * np.exp(1j*omega_test) + c0 + c1 * np.exp(-1j*omega_test)
    min_val = np.min(np.real(Cw))
    print(f"  alpha={a:.2f} lam={l:.2f}: min C(e^iw) = {min_val:.4f}")
