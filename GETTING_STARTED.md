# ZEOZO — y = m x + c, redefined
*A bounded, entropy aware, zero centric canonical you can try right now.*

![GitHub Stars](https://img.shields.io/github/stars/OMPSHUNYAYA/ZEOZO?style=flat&logo=github)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue?style=flat&logo=creative-commons)

**What ZEOZO achieves**

- ZEOZO redefines the classic line **y = m x + c** into a living, bounded signal.  
- It lifts early signals and alignment.  
- It stays universal, audit friendly, and stable.  
- When drift builds, **ZEOZO rises**.  
- When the system recovers, **ZEOZO falls**.  
- The same math applies to geometry, earthquakes, hurricanes, ECG, cybersecurity, insurance, telecom, snow, and many more domains.  
- In the examples below, you can compare ZEOZO with a straight line and a rolling Shannon entropy view.  
- **ZEOZO** reacts promptly and then stabilizes.  
- **Shannon** stays nearly flat on a clean ramp because it summarizes the distribution without time awareness.

**Why now (ZEOZO-Core vs earlier canonical)**

- **More bounded:** log(1 + energy) keeps extremes controlled.  
- **Zero-centric & scale-free:** median / MAD edge units remove units and scale.  
- **Entropy-aware:** exponential memory (λ) discounts distant past, reacts early.  
- **Stable alignment:** a slow EWMA (μ) provides recovery and persistence.  
- **Universal by default:** no per-city or per-domain tuning knobs.  
- **Audit-friendly:** plain-text math, deterministic, reproducible.  
- **Drop-in:** sensible defaults (λ≈0.10, μ≈0.04) work across many signals.


### Canonical formulas (plain text)

**ZEOZO-Core (current) — bounded, zero-centric, time-aware**
- med = median(x)
- rad = median(|x − med|);  rad = max(rad, ε)
- y_t = (x_t − med) / rad
- E_t = (1 − λ)·E_{t−1} + λ·(y_t)^2
- Z_t = log(1 + E_t)
- A_t = (1 − μ)·A_{t−1} + μ·Z_t
- Δ_t = |Z_t − A_t|

**Zentrube (earlier canonical) — variance with horizon decay**
- Zentrube_t = log( Var(x_0:t) + 1 ) · exp(−λ·t)

**Relationship**
- Both are bounded and time-aware through log(·) and λ.
- ZEOZO adds zero-centric normalization (median, MAD) and online EWMA energy E_t,
  making it more stable and universal across domains.
- A_t provides a slow alignment track; Δ_t highlights misalignment for gating.

**Defaults**
- λ = 0.10,  μ = 0.04,  ε = 1e−6

# zeozo_quickstart.py
# ZEOZO quick start — minimal, self-contained (ASCII only)

# Canonical (ASCII):
#   med = median(x)
#   rad = median(|x - med|); rad = max(rad, eps)
#   y_t = (x_t - med) / rad
#   E_t = (1 - lam)*E_{t-1} + lam*(y_t**2)
#   Z_t = log(1 + E_t)
#   A_t = (1 - mu)*A_{t-1} + mu*Z_t
#   Delta_t = |Z_t - A_t|

# ZEOZO vs Shannon (and earlier Zentrube) — copy-paste and run
# ==============================================================
# What this shows
# - A simple demo on y = m*x + c (a clean ramp).
# - ZEOZO: bounded, time-aware readiness signal (reacts, then stabilizes).
# - Rolling Shannon: distribution-centric; mostly flat on a clean ramp.
# - Earlier Zentrube (prefix variance with exponential time decay) for context.
#
# How to run
#   python zeozo_vs_shannon.py
# If you want the plot:
#   pip install matplotlib
#
# License note
#   © Shunyaya Framework Authors — CC BY-NC 4.0 (non-commercial, with attribution).
#   This script is for research, review, and education.

import math
import numpy as np

# ---------------------------
# ZEOZO core (plain-text math)
# ---------------------------
# Edge-normalize (robust, global for this demo):
#   med = median(x)
#   rad = median(|x - med|); rad = max(rad, eps)
#   y_t = (x_t - med) / rad
#
# Energy + log compression (bounded):
#   E_t = (1 - lam) * E_{t-1} + lam * (y_t^2)
#   Z_t = log(1 + E_t)
#
# Alignment (slow recovery) and misalignment:
#   A_t = (1 - mu) * A_{t-1} + mu * Z_t
#   Delta_t = |Z_t - A_t|
def zeozo_core(series, lam=0.10, mu=0.04, eps=1e-6):
    x = np.asarray(series, float)
    med = float(np.median(x))
    rad = float(np.median(np.abs(x - med)))
    if rad < eps:
        rad = eps

    E = 0.0
    Z = []
    A = []
    for xi in x:
        y = (xi - med) / rad
        E = (1 - lam) * E + lam * (y * y)
        z = math.log(1.0 + E)
        Z.append(z)
        a_prev = A[-1] if A else z
        A.append((1 - mu) * a_prev + mu * z)

    Z = np.array(Z, dtype=float)
    A = np.array(A, dtype=float)
    Delta = np.abs(Z - A)
    return Z, A, Delta

# ---------------------------------------
# Rolling Shannon on edge-normalized data
# ---------------------------------------
# Windowed histogram entropy (nats) on y_t.
# Fixed bin edges across the whole stream to reduce wiggles.
def rolling_shannon_on_edges(series, win=64, bins=20, eps=1e-6):
    x = np.asarray(series, float)
    med = float(np.median(x))
    rad = float(np.median(np.abs(x - med)))
    if rad < eps:
        rad = eps
    y = (x - med) / rad

    H = np.full_like(y, np.nan, dtype=float)

    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
        return H
    edges = np.linspace(y_min, y_max, bins + 1)

    for t in range(len(y)):
        if t + 1 < win:
            continue
        seg = y[t + 1 - win : t + 1]
        counts, _ = np.histogram(seg, bins=edges)
        W = float(np.sum(counts))
        if W <= 0:
            continue
        p = counts / W
        H_t = -np.sum([pi * math.log(pi) for pi in p if pi > 0.0])
        H[t] = H_t
    return H

# --------------------------------------------------------
# Earlier canonical (Zentrube, prefix variance with decay)
# --------------------------------------------------------
# Zentrube_t = log(Var(x_0:t) + 1) * exp(-lambda_time * t)
# Uses population variance via Welford for each prefix.
def zentrube_prefix(series, lambda_time=0.02):
    arr = np.asarray(series, float)
    Zp = []
    mean = 0.0
    M2 = 0.0
    n = 0
    for xi in arr:
        n += 1
        delta = xi - mean
        mean += delta / n
        M2 += delta * (xi - mean)
        var = M2 / n  # population variance
        z = math.log(1.0 + var) * math.exp(-lambda_time * n)
        Zp.append(z)
    return np.array(Zp, dtype=float)

if __name__ == "__main__":
    # Synthetic ramp: y = m*x + c
    n = 200
    t = np.linspace(0.0, 50.0, n)
    m, c = 2.0, 5.0
    line = m * t + c

    # Compute signals
    Z, A, D = zeozo_core(line, lam=0.10, mu=0.04)
    H = rolling_shannon_on_edges(line, win=64, bins=20)
    Zp = zentrube_prefix(line, lambda_time=0.02)

    # Console samples
    print("Line sample (y = m*x + c):", np.round(line[:5], 3).tolist())
    print("ZEOZO Z first 5:", np.round(Z[:5], 4).tolist())
    print("Alignment A first 5:", np.round(A[:5], 4).tolist())
    print("Misalignment Delta first 5:", np.round(D[:5], 4).tolist())

    finite_H = H[~np.isnan(H)]
    if finite_H.size > 0:
        print("Rolling Shannon H (first 5 after warm-up):",
              np.round(finite_H[:5], 4).tolist())
    else:
        print("Rolling Shannon H: not enough samples (increase n or reduce window).")

    print("Earlier Zentrube (prefix) first 5:", np.round(Zp[:5], 4).tolist())

    # How to read
    print("\nHow to read this:")
    print("- ZEOZO rises promptly on the ramp, then stabilizes (bounded, time-aware).")
    print("- Rolling Shannon is nearly flat on a clean ramp after warm-up (distribution-centric).")
    print("- Earlier Zentrube compresses prefix variance and decays with time; useful context,")
    print("  but ZEOZO is our preferred readiness dial for rupture-to-recovery.")

    # Plot (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(t, Z, label="ZEOZO Z_t")
        plt.plot(t, A, label="Alignment A_t")
        plt.plot(t, H, label="Rolling Shannon H_t", alpha=0.85)
        plt.plot(t, Zp, "--", label="Earlier Zentrube (prefix)", alpha=0.85)
        plt.xlabel("t")
        plt.ylabel("value")
        plt.title("ZEOZO vs Shannon (and Earlier Zentrube) on y = m*x + c")
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        print("\n[note] matplotlib not available. To see a plot:")
        print("  pip install matplotlib")
        print("  then re-run this script.")

Reading tip:
- ZEOZO Z rises promptly on ramps, then stabilizes (bounded, time-aware).
- A is a slow alignment track; Δ = |Z − A| highlights misalignment.







