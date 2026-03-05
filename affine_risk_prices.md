---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(affine_risk_prices)=
```{raw} html
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Affine Models of Asset Prices

## Overview

This lecture describes a class of **affine** or **exponential quadratic** models of the
stochastic discount factor that have become widely used in empirical finance.

These models are presented in chapter 15 of {cite}`Ljungqvist2012`.

The models discussed here take a different approach from the time-separable CRRA
stochastic discount factor of Hansen and Singleton {cite}`HansenSingleton1983`.

The CRRA stochastic discount factor is

$$
m_{t+1} = \exp\left(-r_t - \frac{1}{2}\sigma_c^2 \gamma^2 - \gamma\sigma_c\varepsilon_{t+1}\right)
$$

where $r_t = \rho + \gamma\mu - \frac{1}{2}\sigma_c^2\gamma^2$.

This model asserts that exposure to the random part of aggregate consumption growth,
$\sigma_c\varepsilon_{t+1}$, is the **only** priced risk — the sole source of discrepancies
among expected returns across assets.

Empirical difficulties with this specification (the equity premium puzzle, the
risk-free rate puzzle, and the Hansen-Jagannathan bounds) motivate the alternative approach
described in this lecture.

The **affine model** maintains $\mathbb{E}(m_{t+1}R_{j,t+1}) = 1$ but **divorces** the
stochastic discount factor from consumption risk.  Instead, it

* specifies an analytically tractable stochastic process for $m_{t+1}$, and
* uses overidentifying restrictions from $\mathbb{E}(m_{t+1}R_{j,t+1}) = 1$ applied to $N$
  assets to let the data reveal risks and their prices.

Key applications we study include:

1. **Pricing risky assets** — how risk prices and exposures determine excess returns.
1. **Affine term structure models** — bond yields as affine functions of a state vector
   (Ang and Piazzesi {cite}`AngPiazzesi2003`).
1. **Risk-neutral probabilities** — a change-of-measure representation of the pricing equation.
1. **Distorted beliefs** — reinterpreting risk price estimates when agents hold systematically
   biased forecasts (Piazzesi, Salomao, and Schneider {cite}`PiazzesiSalomaoSchneider2015`).

We start with some standard imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
```

## The Model

### State dynamics and short rate

The model has two components.

**Component 1** is a vector autoregression that describes the state of the economy
and the evolution of the short rate:

```{math}
:label: eq_var

z_{t+1} = \mu + \phi z_t + C\varepsilon_{t+1}
```

```{math}
:label: eq_shortrate

r_t = \delta_0 + \delta_1' z_t
```

Here

* $\phi$ is a stable $m \times m$ matrix,
* $C$ is an $m \times m$ matrix,
* $\varepsilon_{t+1} \sim \mathcal{N}(0, I)$ is an i.i.d. $m \times 1$ random vector,
* $z_t$ is an $m \times 1$ state vector.

Equation {eq}`eq_shortrate` says that the **short rate** $r_t$ — the net yield on a
one-period risk-free claim — is an affine function of the state $z_t$.

**Component 2** is a vector of **risk prices** $\lambda_t$ and an associated stochastic
discount factor $m_{t+1}$:

```{math}
:label: eq_riskprices

\lambda_t = \lambda_0 + \lambda_z z_t
```

```{math}
:label: eq_sdf

\log(m_{t+1}) = -r_t - \frac{1}{2}\lambda_t'\lambda_t - \lambda_t'\varepsilon_{t+1}
```

Here $\lambda_0$ is $m \times 1$ and $\lambda_z$ is $m \times m$.

The entries of $\lambda_t$ that multiply corresponding entries of the risks
$\varepsilon_{t+1}$ are called **risk prices** because they determine how exposures
to each risk component affect expected returns (as we show below).

Because $\lambda_t$ is affine in $z_t$, the stochastic discount factor $m_{t+1}$ is
**exponential quadratic** in the state $z_t$.

### Properties of the SDF

Since $\lambda_t'\varepsilon_{t+1}$ is conditionally normal, it follows that

$$
\mathbb{E}_t(m_{t+1}) = \exp(-r_t)
$$

and   

$$
\text{std}_t(m_{t+1}) \approx |\lambda_t|.
$$

The first equation confirms that $r_t$ is the net yield on a risk-free one-period bond.
That is why $r_t$ is called **the short rate** in the exponential quadratic literature.

The second equation says that the conditional standard deviation of the SDF
is approximately the magnitude of the vector of risk prices — a measure of overall
**market price of risk**.

## Pricing Risky Assets

### Lognormal returns

Consider a risky asset $j$ whose gross return has a lognormal conditional distribution:

```{math}
:label: eq_return

R_{j,t+1} = \exp\left(\nu_t(j) - \frac{1}{2}\alpha_t(j)'\alpha_t(j) + \alpha_t(j)'\varepsilon_{t+1}\right)
```

where the **exposure vector**

```{math}
:label: eq_exposure

\alpha_t(j) = \alpha_0(j) + \alpha_z(j)\, z_t
```

Here $\alpha_0(j)$ is $m \times 1$ and $\alpha_z(j)$ is $m \times m$.

The components of $\alpha_t(j)$ express the **exposures** of $\log R_{j,t+1}$ to
corresponding components of the risk vector $\varepsilon_{t+1}$.

The specification {eq}`eq_return` implies $\mathbb{E}_t R_{j,t+1} = \exp(\nu_t(j))$,
so $\nu_t(j)$ is the expected net log return.

### Expected excess returns

Applying the pricing equation $\mathbb{E}_t(m_{t+1}R_{j,t+1}) = 1$ together with the
formula for the mean of a lognormal random variable gives

```{math}
:label: eq_excess

\nu_t(j) = r_t + \alpha_t(j)'\lambda_t
```

This is a central result.  It says:

> The expected net return on asset $j$ equals the short rate plus the inner product
> of the asset's exposure vector $\alpha_t(j)$ with the risk price vector $\lambda_t$.

Each component of $\lambda_t$ prices the corresponding component of $\varepsilon_{t+1}$.
An asset that loads heavily on a risk component with a large risk price earns a
correspondingly high expected return.

## Affine Term Structure of Yields

One of the most important applications is the **affine term structure model** studied
by Ang and Piazzesi {cite}`AngPiazzesi2003`.

### Bond prices

Let $p_t(n)$ be the price at time $t$ of a risk-free pure discount bond maturing at
$t + n$ (paying one unit of consumption).  The one-period gross return on holding
an $(n+1)$-period bond from $t$ to $t+1$ is

$$
R_{t+1} = \frac{p_{t+1}(n)}{p_t(n+1)}
$$

The pricing equation $\mathbb{E}_t(m_{t+1}R_{t+1}) = 1$ implies

```{math}
:label: eq_bondrecur

p_t(n+1) = \mathbb{E}_t\bigl(m_{t+1}\,p_{t+1}(n)\bigr)
```

with the initial condition

$$
p_t(1) = \mathbb{E}_t(m_{t+1}) = \exp(-r_t) = \exp(-\delta_0 - \delta_1'z_t).
$$

### Exponential affine prices

The recursion {eq}`eq_bondrecur` has an **exponential affine** solution:

```{math}
:label: eq_bondprice

p_t(n) = \exp\!\bigl(\bar A_n + \bar B_n' z_t\bigr)
```

where the scalar $\bar A_n$ and the $m \times 1$ vector $\bar B_n$ satisfy the
**Riccati difference equations**

```{math}
:label: eq_riccati_A

\bar A_{n+1} = \bar A_n + \bar B_n'(\mu - C\lambda_0) + \frac{1}{2}\bar B_n' CC'\bar B_n - \delta_0
```

```{math}
:label: eq_riccati_B

\bar B_{n+1}' = \bar B_n'(\phi - C\lambda_z) - \delta_1'
```

with initial conditions $\bar A_1 = -\delta_0$ and $\bar B_1 = -\delta_1$.

### Yields

The **yield to maturity** on an $n$-period bond is

$$
y_t(n) = -\frac{\log p_t(n)}{n}
$$

Substituting {eq}`eq_bondprice` gives

```{math}
:label: eq_yield

y_t(n) = A_n + B_n' z_t
```

where $A_n = -\bar A_n / n$ and $B_n = -\bar B_n / n$.

**Yields are affine functions of the state vector $z_t$.**  This is the defining
property of affine term structure models.

## Python Implementation

We now implement the affine term structure model and compute bond prices, yields,
and risk premiums numerically.

```{code-cell} ipython3
class AffineTermStructure:
    """
    Implements the affine term structure model of
    Ang and Piazzesi (2003).

    State dynamics:
        z_{t+1} = μ + φ z_t + C ε_{t+1},   ε_{t+1} ~ N(0, I)
    Short rate:
        r_t = δ_0 + δ_1' z_t
    Risk prices:
        λ_t = λ_0 + λ_z z_t
    Log SDF:
        log(m_{t+1}) = -r_t - (1/2) λ_t' λ_t - λ_t' ε_{t+1}

    Parameters
    ----------
    mu : array (m,)
    phi : array (m, m)
    C : array (m, m)
    delta0 : float
    delta1 : array (m,)
    lambda0 : array (m,)
    lambdaz : array (m, m)
    """

    def __init__(self, mu, phi, C, delta0, delta1, lambda0, lambdaz):
        self.mu = np.asarray(mu, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.delta0 = float(delta0)
        self.delta1 = np.asarray(delta1, dtype=float)
        self.lambda0 = np.asarray(lambda0, dtype=float)
        self.lambdaz = np.asarray(lambdaz, dtype=float)
        self.m = len(self.mu)
        # Risk-adjusted drift: φ - C λ_z  and  μ - C λ_0
        self.phi_rn = self.phi - self.C @ self.lambdaz
        self.mu_rn = self.mu - self.C @ self.lambda0

    def bond_coefficients(self, n_max):
        """
        Compute (Ā_n, B̄_n) for n = 1, ..., n_max via forward recursion.

        Returns
        -------
        A_bar : array (n_max + 1,)   A_bar[n] = Ā_n
        B_bar : array (n_max + 1, m) B_bar[n] = B̄_n
        """
        A_bar = np.zeros(n_max + 1)
        B_bar = np.zeros((n_max + 1, self.m))

        # Initial conditions: Ā_1 = -δ_0, B̄_1 = -δ_1
        A_bar[1] = -self.delta0
        B_bar[1] = -self.delta1

        CC = self.C @ self.C.T

        for n in range(1, n_max):
            Bn = B_bar[n]
            A_bar[n + 1] = (A_bar[n]
                            + Bn @ (self.mu - self.C @ self.lambda0)
                            + 0.5 * Bn @ CC @ Bn
                            - self.delta0)
            B_bar[n + 1] = self.phi_rn.T @ Bn - self.delta1

        return A_bar, B_bar

    def yields(self, z, n_max):
        """
        Compute the yield curve y_t(n) = A_n + B_n' z_t for n=1,...,n_max.

        Parameters
        ----------
        z : array (m,)  current state vector
        n_max : int     maximum maturity

        Returns
        -------
        y : array (n_max,)  yields y(1), ..., y(n_max)
        """
        A_bar, B_bar = self.bond_coefficients(n_max)
        ns = np.arange(1, n_max + 1)
        y = np.array([(-A_bar[n] - B_bar[n] @ z) / n for n in ns])
        return y

    def bond_prices(self, z, n_max):
        """
        Compute bond prices p_t(n) = exp(Ā_n + B̄_n' z_t) for n=1,...,n_max.

        Parameters
        ----------
        z : array (m,)
        n_max : int

        Returns
        -------
        p : array (n_max,)
        """
        A_bar, B_bar = self.bond_coefficients(n_max)
        p = np.array([np.exp(A_bar[n] + B_bar[n] @ z) for n in range(1, n_max + 1)])
        return p

    def simulate(self, z0, T, rng=None):
        """
        Simulate the state process for T periods.

        Parameters
        ----------
        z0 : array (m,)  initial state
        T : int          number of periods
        rng : np.random.Generator (optional)

        Returns
        -------
        Z : array (T+1, m)  simulated states including z0
        """
        if rng is None:
            rng = np.random.default_rng(42)
        Z = np.zeros((T + 1, self.m))
        Z[0] = z0
        for t in range(T):
            eps = rng.standard_normal(self.m)
            Z[t + 1] = self.mu + self.phi @ Z[t] + self.C @ eps
        return Z

    def short_rate(self, z):
        """Short rate r_t = δ_0 + δ_1' z_t."""
        return self.delta0 + self.delta1 @ z

    def risk_prices(self, z):
        """Risk price vector λ_t = λ_0 + λ_z z_t."""
        return self.lambda0 + self.lambdaz @ z

    def expected_excess_return(self, z, alpha0, alphaz):
        """
        Expected excess return ν_t(j) - r_t = α_t(j)' λ_t.

        Parameters
        ----------
        z : array (m,)
        alpha0 : array (m,)
        alphaz : array (m, m)
        """
        alpha_t = alpha0 + alphaz @ z
        lambda_t = self.risk_prices(z)
        return alpha_t @ lambda_t
```

### A one-factor Gaussian example

To build intuition, we start with a single-factor ($m=1$) Gaussian model.

```{code-cell} ipython3
# ── One-factor Gaussian model ──────────────────────────────────────────────────
# State z_t follows an AR(1): z_{t+1} = μ(1 - φ) + φ z_t + σ ε_{t+1}
# Short rate: r_t = δ_0 + δ_1 z_t
# Risk price:  λ_t = λ_0 + λ_z z_t  (usually λ_z < 0)

mu = np.array([0.01])        # unconditional mean of z
phi = np.array([[0.95]])     # persistence
C = np.array([[0.01]])       # shock volatility (σ = 0.01)
delta0 = 0.005               # constant in short rate
delta1 = np.array([1.0])     # loading on z in short rate
lambda0 = np.array([0.5])    # constant risk price
lambdaz = np.array([[-20.0]]) # state-dependent risk price (negative: higher z → lower λ)

model_1f = AffineTermStructure(mu, phi, C, delta0, delta1, lambda0, lambdaz)
```

### Yield curve shapes

We compute yield curves across a range of short-rate states $z_t$.

```{code-cell} ipython3
maturities = np.arange(1, 121)   # 1 to 120 periods (e.g., months)
n_max = 120

# Three states: low, medium, high short rate
z_low  = np.array([-0.02])
z_mid  = np.array([0.01])
z_high = np.array([0.04])

fig, ax = plt.subplots(figsize=(8, 5))

for z, label, color in [(z_low, "Low state", "steelblue"),
                         (z_mid, "Median state", "seagreen"),
                         (z_high, "High state", "firebrick")]:
    y = model_1f.yields(z, n_max) * 400   # annualise (×400 for quarterly data)
    ax.plot(maturities, y, color=color, lw=2, label=label)

ax.set_xlabel("Maturity (periods)", fontsize=13)
ax.set_ylabel("Yield (annualised, bps × 4)", fontsize=13)
ax.set_title("Affine Term Structure — One-Factor Model", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
```

The model generates upward-sloping, flat, and inverted yield curves as the short
rate moves across states — a key qualitative feature of observed bond markets.

### Short rate dynamics

```{code-cell} ipython3
T = 200
Z = model_1f.simulate(z_mid, T)
short_rates = [model_1f.short_rate(Z[t]) * 400 for t in range(T + 1)]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(short_rates, color="steelblue", lw=1.5)
ax.axhline(model_1f.short_rate(mu) * 400, color="red", ls="--", lw=1.2,
           label="Unconditional mean")
ax.set_xlabel("Period", fontsize=13)
ax.set_ylabel("Short rate (annualised)", fontsize=13)
ax.set_title("Simulated Short Rate — One-Factor Model", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
```

### A two-factor model

To match richer yield-curve dynamics, practitioners routinely use $m \geq 2$
factors.  We now introduce a two-factor specification in which the factors
can be interpreted as a **level** component and a **slope** component.

```{code-cell} ipython3
# ── Two-factor model ────────────────────────────────────────────────────────────
# z = [level, slope]'
mu2    = np.array([0.01,  0.0])
phi2   = np.array([[0.97, -0.05],   # level is very persistent
                   [0.00,  0.92]])  # slope is somewhat persistent
C2     = np.array([[0.008, 0.000],
                   [0.000, 0.012]])
delta0_2  = 0.003
delta1_2  = np.array([1.0, 0.5])           # both factors affect short rate
lambda0_2 = np.array([0.4,  0.2])
lambdaz_2 = np.array([[-15.0,  0.0],
                      [  0.0, -8.0]])

model_2f = AffineTermStructure(mu2, phi2, C2,
                                delta0_2, delta1_2,
                                lambda0_2, lambdaz_2)
```

```{code-cell} ipython3
n_max = 120
maturities = np.arange(1, n_max + 1)

# Different (level, slope) combinations
states = {
    "Normal (flat)":        np.array([0.005,  0.000]),
    "Steep (low short)":    np.array([-0.01,  0.020]),
    "Flat/inverted (high)": np.array([0.025, -0.015]),
}

fig, ax = plt.subplots(figsize=(8, 5))
for label, z in states.items():
    y = model_2f.yields(z, n_max) * 400
    ax.plot(maturities, y, lw=2, label=label)

ax.set_xlabel("Maturity (periods)", fontsize=13)
ax.set_ylabel("Yield (annualised bps × 4)", fontsize=13)
ax.set_title("Affine Term Structure — Two-Factor Model", fontsize=14)
ax.legend()
plt.tight_layout()
plt.show()
```

## Risk Premiums

A key object in the affine term structure model is the **term premium** — the extra
expected return on a long-term bond relative to rolling over short-term bonds.

For an $(n+1)$-period bond held for one period, the excess log return is
approximately

$$
\mathbb{E}_t\left[\log R_{t+1}^{(n+1)}\right] - r_t \;=\; -\bar B_n' C \lambda_t
$$

That is, the term premium equals (minus) the product of the bond's exposure to
the shocks $(-\bar B_n'C)$ with the risk prices $\lambda_t$.

```{code-cell} ipython3
def term_premiums(model, z, n_max):
    """
    Compute approximate term premiums for maturities 1 to n_max.

    Term premium for holding an (n+1)-period bond ≈ -B̄_n' C λ_t.
    """
    A_bar, B_bar = model.bond_coefficients(n_max + 1)
    lambda_t = model.risk_prices(z)
    tp = np.array([-B_bar[n] @ model.C @ lambda_t
                   for n in range(1, n_max + 1)])
    return tp

maturities_tp = np.arange(1, 121)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

z_states = {
    r"Low $z$ (low short rate)": np.array([-0.01, 0.01]),
    r"High $z$ (high short rate)": np.array([0.03, -0.01]),
}

for ax, (label, z) in zip(axes, z_states.items()):
    tp = term_premiums(model_2f, z, 120) * 400
    ax.plot(maturities_tp, tp, color="purple", lw=2)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Maturity (periods)", fontsize=12)
    ax.set_ylabel("Term premium (annualised)", fontsize=12)
    ax.set_title(f"Term Premiums — {label}", fontsize=12)

plt.tight_layout()
plt.show()
```

## Risk-Neutral Probabilities

The stochastic discount factor {eq}`eq_sdf` defines a **change of measure** from the
physical measure $P$ to the **risk-neutral measure** $Q$.

Define the likelihood ratio

```{math}
:label: eq_RN_ratio

\frac{\xi^Q_{t+1}}{\xi^Q_t} = \exp\!\left(-\frac{1}{2}\lambda_t'\lambda_t - \lambda_t'\varepsilon_{t+1}\right)
```

Then

$$
m_{t+1} = \frac{\xi^Q_{t+1}}{\xi^Q_t}\exp(-r_t)
$$

and the pricing equation $\mathbb{E}^P_t(m_{t+1}R_{j,t+1}) = 1$ becomes

```{math}
:label: eq_Qpricing

\mathbb{E}^Q_t R_{j,t+1} = \exp(r_t)
```

**Under the risk-neutral measure, expected returns on all assets equal the risk-free return.**

### The risk-neutral VAR

Multiplying the physical conditional distribution of $z_{t+1}$ by the likelihood
ratio {eq}`eq_RN_ratio` gives the **risk-neutral conditional distribution**

$$
z_{t+1} \mid z_t \;\overset{Q}{\sim}\; \mathcal{N}\!\bigl(\mu - C\lambda_0 + (\phi - C\lambda_z)z_t,\; CC'\bigr)
$$

In other words, under $Q$ the state vector follows

$$
z_{t+1} = (\mu - C\lambda_0) + (\phi - C\lambda_z)\,z_t + C\varepsilon^Q_{t+1}
$$

where $\varepsilon^Q_{t+1} \sim \mathcal{N}(0, I)$ under $Q$.

The risk-neutral drift adjustments $-C\lambda_0$ (constant) and $-C\lambda_z$ (state-dependent)
encode exactly how the asset pricing formula $\mathbb{E}^P_t m_{t+1}R_{j,t+1}=1$ adjusts
expected returns for exposure to the risks $\varepsilon_{t+1}$.

### Verification via risk-neutral pricing

Bond prices can be computed by discounting at $r_t$ under $Q$:

$$
p_t(n) = \mathbb{E}^Q_t\! \left[\exp\!\left(-\sum_{s=0}^{n-1}r_{t+s}\right)\right]
$$

We can verify that this agrees with {eq}`eq_bondprice` by iterating the affine
recursion under the risk-neutral VAR.  Below we confirm this numerically.

```{code-cell} ipython3
def bond_price_monte_carlo_Q(model, z0, n, n_sims=50_000, rng=None):
    """
    Estimate p_t(n) by Monte Carlo under the risk-neutral measure Q.
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    m_dim = len(z0)
    Z = np.tile(z0, (n_sims, 1))     # (n_sims, m)
    disc = np.zeros(n_sims)           # cumulative discount

    phi_Q = model.phi_rn              # φ - C λ_z
    mu_Q  = model.mu_rn               # μ - C λ_0
    C_mat = model.C

    for _ in range(n):
        r_t = model.delta0 + Z @ model.delta1   # (n_sims,)
        disc += r_t
        eps = rng.standard_normal((n_sims, m_dim))
        Z = mu_Q + Z @ phi_Q.T + eps @ C_mat.T

    return np.mean(np.exp(-disc))

# Compare analytical and Monte Carlo bond prices
z_test = np.array([0.01, 0.005])
n_max_test = 40
p_analytic = model_2f.bond_prices(z_test, n_max_test)

rng = np.random.default_rng(2024)
maturities_check = [4, 12, 24, 40]
mc_prices = [bond_price_monte_carlo_Q(model_2f, z_test, n, n_sims=80_000, rng=rng)
             for n in maturities_check]

print(f"{'Maturity':>10}  {'Analytic':>12}  {'Monte Carlo':>12}  {'Error (bps)':>12}")
print("-" * 52)
for n, mc in zip(maturities_check, mc_prices):
    analytic = p_analytic[n - 1]
    error_bp = abs(analytic - mc) / analytic * 10_000
    print(f"{n:>10}  {analytic:>12.6f}  {mc:>12.6f}  {error_bp:>12.2f}")
```

The analytical and Monte Carlo bond prices agree closely, validating the
Riccati recursion {eq}`eq_riccati_A`–{eq}`eq_riccati_B`.

## Distorted Beliefs

Piazzesi, Salomao, and Schneider {cite}`PiazzesiSalomaoSchneider2015` assemble survey
evidence suggesting that economic experts' forecasts are **systematically biased**
relative to the physical measure.

### The subjective measure

Let $\hat z_{t+1}$ be one-period-ahead expert forecasts.  Regressing these on $z_t$:

$$
\hat z_{t+1} = \hat\mu + \hat\phi\, z_t + e_{t+1}
$$

yields estimates $\hat\mu, \hat\phi$ that differ from the physical parameters $\mu, \phi$.

To formalise the distortion, let $\kappa_t = \kappa_0 + \kappa_z z_t$ and define
the likelihood ratio

```{math}
:label: eq_Srat

\frac{\xi^S_{t+1}}{\xi^S_t}
= \exp\!\left(-\frac{1}{2}\kappa_t'\kappa_t - \kappa_t'\varepsilon_{t+1}\right)
```

Multiplying the physical conditional distribution of $z_{t+1}$ by this likelihood
ratio gives the **subjective (S) conditional distribution**

$$
z_{t+1} \mid z_t \;\overset{S}{\sim}\;
\mathcal{N}\!\bigl(\mu - C\kappa_0 + (\phi - C\kappa_z)\,z_t,\; CC'\bigr)
$$

Comparing with the regression implies

$$
\hat\mu = \mu - C\kappa_0, \qquad \hat\phi = \phi - C\kappa_z
$$

Piazzesi et al. find that experts behave as if the level and slope of the yield
curve are **more persistent** than under the physical measure: $\hat\phi$ has
larger eigenvalues than $\phi$.

### Pricing under distorted beliefs

A representative agent with subjective beliefs $S$ and risk prices $\lambda^\star_t$
satisfies

$$
\mathbb{E}^S_t\bigl(m^\star_{t+1} R_{j,t+1}\bigr) = 1
$$

Expanding this in terms of the physical measure $P$, one finds that the
**rational-expectations econometrician** who imposes $P$ will estimate risk prices

$$
\hat\lambda_t = \lambda^\star_t + \kappa_t
$$

That is, the econometrician's estimate conflates true risk prices $\lambda^\star_t$
and belief distortions $\kappa_t$.  Part of what looks like a high price of risk
is actually a systematic forecast bias.

### Numerical illustration

```{code-cell} ipython3
# Physical parameters (two-factor model from above)
phi_P  = np.array([[0.97, -0.05], [0.00, 0.92]])
mu_P   = np.array([0.01, 0.0])

# Subjective parameters: more persistent level and slope
phi_S  = np.array([[0.985, -0.04], [0.00, 0.96]])
mu_S   = np.array([0.009, 0.0])

# Distortion parameters: κ_z such that  (φ - C κ_z) = φ_S
#   ⟹  C κ_z = φ_P - φ_S
C2_mat = model_2f.C
kappa_z = np.linalg.solve(C2_mat, phi_P - phi_S)
kappa_0 = np.linalg.solve(C2_mat, mu_P - mu_S)
print("Distortion κ_0:", kappa_0.round(2))
print("Distortion κ_z:\n", kappa_z.round(1))

# Rational-expectations econometrician sees:
#   λ̂_t = λ*_t + κ_t
# Suppose true risk prices are
lambda_star_0 = np.array([0.3,  0.1])
lambda_star_z = np.array([[-8.0, 0.0], [0.0, -4.0]])

# Econometrician attributes these risk prices:
lambda_hat_0 = lambda_star_0 + kappa_0
lambda_hat_z = lambda_star_z + kappa_z
print("\nTrue λ*_0      :", lambda_star_0.round(3))
print("Econometrician's λ̂_0:", lambda_hat_0.round(3))
```

```{code-cell} ipython3
# Compare term premiums under true vs. distorted risk price estimates
model_true = AffineTermStructure(mu2, phi2, C2,
                                  delta0_2, delta1_2,
                                  lambda_star_0, lambda_star_z)

model_econ = AffineTermStructure(mu2, phi2, C2,
                                  delta0_2, delta1_2,
                                  lambda_hat_0, lambda_hat_z)

z_ref = np.array([0.01, 0.005])
maturities_tp = np.arange(1, 121)

tp_true = term_premiums(model_true, z_ref, 120) * 400
tp_econ = term_premiums(model_econ, z_ref, 120) * 400

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(maturities_tp, tp_true, lw=2, color="steelblue",  label=r"True risk prices $\lambda^\star_t$")
ax.plot(maturities_tp, tp_econ, lw=2, color="firebrick", ls="--",
        label=r"RE econometrician's estimate $\hat\lambda_t = \lambda^\star_t + \kappa_t$")
ax.axhline(0, color="black", lw=0.8, ls=":")
ax.set_xlabel("Maturity (periods)", fontsize=13)
ax.set_ylabel("Term premium (annualised)", fontsize=13)
ax.set_title("Term Premiums: True vs. Distorted-Belief Estimates", fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()
```

When expert beliefs are overly persistent ($\hat\phi$ has larger eigenvalues than
$\phi$), the rational-expectations econometrician attributes too much of the
observed risk premium to risk aversion.  Disentangling belief distortions from
genuine risk prices requires additional data — for example, the survey forecasts
used by Piazzesi, Salomao, and Schneider.

## Appendix: Deriving the Bond Price Recursion

We verify the exponential affine form {eq}`eq_bondprice` by induction.

**Claim:** If $p_{t+1}(n) = \exp(\bar A_n + \bar B_n' z_{t+1})$, then
$p_t(n+1) = \exp(\bar A_{n+1} + \bar B_{n+1}' z_t)$ with $\bar A_{n+1}$ and
$\bar B_{n+1}$ given by {eq}`eq_riccati_A`–{eq}`eq_riccati_B`.

**Proof sketch.**  Using the SDF {eq}`eq_sdf` and the VAR {eq}`eq_var`:

$$
\log m_{t+1} + \log p_{t+1}(n)
= -r_t - \tfrac{1}{2}\lambda_t'\lambda_t
  + (\bar A_n + \bar B_n'\mu + \bar B_n'\phi z_t)
  + (-\lambda_t + C'\bar B_n)'\varepsilon_{t+1}
$$

Taking the conditional expectation (and using $\varepsilon_{t+1}\sim\mathcal{N}(0,I)$):

$$
\log p_t(n+1) = -r_t - \tfrac{1}{2}\lambda_t'\lambda_t
  + \bar A_n + \bar B_n'(\mu + \phi z_t)
  + \tfrac{1}{2}(\lambda_t - C'\bar B_n)'(\lambda_t - C'\bar B_n)
$$

Substituting $r_t = \delta_0 + \delta_1'z_t$ and $\lambda_t = \lambda_0 + \lambda_z z_t$,
collecting constant and linear-in-$z_t$ terms, and equating coefficients gives
exactly {eq}`eq_riccati_A`–{eq}`eq_riccati_B`. $\blacksquare$

## Concluding Remarks

The affine model of the stochastic discount factor provides a flexible and tractable
framework for studying asset prices.  Key features are:

1. **Analytical tractability** — Bond prices are exponential affine in $z_t$;
   expected returns decompose cleanly into a short rate plus a risk-price×exposure inner product.
2. **Empirical flexibility** — The free parameters $(\mu, \phi, C, \delta_0, \delta_1, \lambda_0, \lambda_z)$
   can be estimated by maximum likelihood (the {doc}`Kalman filter <kalman>` chapter describes
   the relevant methods) without imposing restrictions from a full general equilibrium model.
3. **Multiple risks** — The vector structure accommodates many sources of risk (monetary
   policy, real activity, volatility, etc.).
4. **Belief distortions** — The framework naturally accommodates non-rational beliefs via
   likelihood-ratio twists of the physical measure, as in
   Piazzesi, Salomao, and Schneider {cite}`PiazzesiSalomaoSchneider2015`.

The model also connects directly to the Hansen–Jagannathan bounds and to robust
control interpretations of the stochastic discount factor described in other
chapters of {cite}`Ljungqvist2012`.
