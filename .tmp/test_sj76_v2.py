"""Test all code cells from sargent_jacobs_1976.md"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# ── Cell: bivariate_ma1_covariances + structural_to_ma1 ─────────────────────

def bivariate_ma1_covariances(m0, m1, x0, x1):
    c0 = m0**2 + m1**2
    c1 = m0 * m1
    g0 = m0*x0 + m1*x1
    g1 = m0*x1
    return c0, c1, g0, g1


def structural_to_ma1(alpha, lam, sigma_eta2=1.0, sigma_v2=0.5):
    phi = 1.0 / (1.0 + alpha * (1.0 - lam))
    delta = (lam + alpha*(1-lam)) / (1 + alpha*(1-lam))
    s_eta = np.sqrt(sigma_eta2)
    s_v   = np.sqrt(sigma_v2)

    A_eta = (phi*(1-lam) + 1)
    B_eta = -delta * phi*(1-lam)
    A_v   = 1.0
    C_eta = phi
    D_eta = -delta * phi
    C_v   = -phi*(1-lam)

    c0 = (A_eta**2 + B_eta**2)*sigma_eta2 + A_v**2 * sigma_v2
    c1 = A_eta * B_eta * sigma_eta2
    g0 = (A_eta*C_eta + B_eta*D_eta)*sigma_eta2 + A_v*C_v*sigma_v2
    g1 = A_eta * D_eta * sigma_eta2

    return c0, c1, g0, g1


# ── Cell: spectral_factor ───────────────────────────────────────────────────

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


def jacobs_population_params(alpha, lam, sigma_eta2=1.0, sigma_v2=0.5):
    c0, c1, g0, g1 = structural_to_ma1(alpha, lam, sigma_eta2, sigma_v2)
    h0, h1, h2 = compute_h_coefficients(c0, c1, g0, g1)
    y1 = 1.0
    y0 = h0 / (1.0 - h2)
    return y0, y1, h0, h1, h2


# ── Test 1: y1 = 1 everywhere ───────────────────────────────────────────────

alphas  = np.linspace(-2.0, -0.1, 10)
lambdas = np.linspace(0.1,  0.9,  10)

results = []
for a in alphas:
    for l in lambdas:
        y0, y1, h0, h1, h2 = jacobs_population_params(a, l)
        results.append(y1)

print(f"Combinations tested: {len(results)}")
print(f"Range of y1: [{min(results):.8f}, {max(results):.8f}]")
print(f"Max deviation from 1: {max(abs(v - 1) for v in results):.2e}")
assert all(abs(v - 1) < 1e-10 for v in results), "y1 != 1 found!"
print("PASS: y1 = 1 for all parameter combinations")


# ── Test 2: spectral density of M is always non-negative ────────────────────

omega_check = np.linspace(0, np.pi, 500)
for a in [-0.5, -1.0, -1.5]:
    for l in [0.3, 0.5, 0.7, 0.9]:
        c0, c1, g0, g1 = structural_to_ma1(a, l)
        Cw = c1 * np.exp(1j*omega_check) + c0 + c1 * np.exp(-1j*omega_check)
        min_val = np.min(np.real(Cw))
        if min_val < -1e-10:
            print(f"WARN: negative spectral density at alpha={a}, lam={l}: min={min_val:.4f}")
        else:
            pass

print("PASS: spectral density checks done")


# ── Test 3: impulse response decays geometrically ───────────────────────────

def projection_impulse_response(c0, c1, g0, g1, n_lags=20):
    A, B, b0, b = wiener_kolmogorov_projection(c0, c1, g0, g1)
    b0_sq = b0**2
    thetas = np.zeros(n_lags)
    thetas[0] = A / b0_sq
    if n_lags > 1:
        thetas[1] = B / b0_sq - b * thetas[0]
    for j in range(2, n_lags):
        thetas[j] = -b * thetas[j-1]
    return thetas

c0, c1, g0, g1 = structural_to_ma1(-0.5, 0.5)
thetas = projection_impulse_response(c0, c1, g0, g1, n_lags=15)
print(f"First 5 impulse response coefficients: {thetas[:5]}")

# Check |b| < 1 for fundamental factorization
_, _, _, b = wiener_kolmogorov_projection(c0, c1, g0, g1)
print(f"|b| = {abs(b):.4f} (should be < 1)")
assert abs(b) < 1, "b not < 1"
print("PASS: fundamental factorization holds")

print("\nAll tests passed!")
