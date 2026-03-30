# 01_power_hurdle_basic.py
import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import NormalDist

import statsmodels.api as sm


# =========================
# Helpers
# =========================

def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _zscore(x):
    x = np.asarray(x)
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)

def _simulate_correlated_normals(n, rho, rng):
    # returns (emp, ment) ~ N(0,1) with corr=rho
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(Sigma)
    z = rng.normal(size=(n, 2))
    x = z @ L.T
    return x[:, 0], x[:, 1]

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _nb2_rng(mu, alpha, rng):
    """
    Draw from NB2 with mean=mu and var=mu + alpha*mu^2
    using Gamma-Poisson mixture.
      lambda ~ Gamma(shape=1/alpha, scale=alpha*mu)
      y ~ Poisson(lambda)
    """
    mu = np.asarray(mu)
    mu = np.clip(mu, 1e-12, None)

    if alpha <= 0:
        # Poisson fallback
        return rng.poisson(mu)

    shape = 1.0 / alpha
    scale = alpha * mu
    lam = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(lam)

def _binom_wilson_ci(k, n, level=0.95):
    """
    Wilson interval for binomial proportion.
    Returns (low, high).
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = float(k) / float(n)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return (low, high)

def _fit_logit_pvalue(df, term):
    # Fit logit: engage ~ EMP + MENT + EMPxMENT
    # Return p-value for requested term; np.nan if fit fails
    try:
        X = df[["EMP", "MENT", "EMPxMENT"]]
        X = sm.add_constant(X, has_constant="add")
        y = df["engage"].astype(int)
        model = sm.Logit(y, X).fit(disp=0)
        return float(model.pvalues.get(term, np.nan))
    except Exception:
        return np.nan

def _fit_nb_pvalue(df, term, alpha_nb):
    # Fit NB (GLM): completed ~ EMP + MENT + EMPxMENT among engagers
    try:
        df2 = df.loc[df["engage"] == 1].copy()
        # if nobody engaged or too few points, fail gracefully
        if df2.shape[0] < 20:
            return np.nan

        X = df2[["EMP", "MENT", "EMPxMENT"]]
        X = sm.add_constant(X, has_constant="add")
        y = df2["completed"].astype(float)

        fam = sm.families.NegativeBinomial(alpha=alpha_nb)
        model = sm.GLM(y, X, family=fam).fit()
        return float(model.pvalues.get(term, np.nan))
    except Exception:
        return np.nan


# =========================
# Data-generating process (DGP)
# =========================

def simulate_hurdle_dataset(
    N,
    rng,
    rho_emp_ment=0.3,

    # Stage 1: Logistic engage model (log-odds)
    b0_logit=-0.2,      # sets baseline engagement rate
    b_emp_logit=0.4,
    b_ment_logit=0.35,
    b_int_logit=0.20,

    # Stage 2: NB completed model among engagers (log mean)
    b0_nb=1.0,          # sets baseline expected completions among engagers: exp(b0_nb)
    b_emp_nb=0.4,
    b_ment_nb=0.35,
    b_int_nb=0.20,
    alpha_nb=0.8        # overdispersion (bigger = more variance)
):
    emp, ment = _simulate_correlated_normals(N, rho_emp_ment, rng)
    emp = _zscore(emp)
    ment = _zscore(ment)
    inter = emp * ment

    # Stage 1: engage probability
    lin1 = b0_logit + b_emp_logit*emp + b_ment_logit*ment + b_int_logit*inter
    p_engage = _sigmoid(lin1)
    engage = rng.binomial(1, p_engage, size=N)

    # Stage 2: completions among engagers
    lin2 = b0_nb + b_emp_nb*emp + b_ment_nb*ment + b_int_nb*inter
    mu = np.exp(lin2)  # expected completions
    completed = np.zeros(N, dtype=int)
    idx = (engage == 1)
    if idx.any():
        completed[idx] = _nb2_rng(mu[idx], alpha=alpha_nb, rng=rng)

    return pd.DataFrame({
        "EMP": emp,
        "MENT": ment,
        "EMPxMENT": inter,
        "engage": engage,
        "completed": completed
    })


# =========================
# Power engine
# =========================

def run_power_hurdle(
    N_values=(60, 80, 100, 120, 150),
    nsim=1000,
    alpha=0.05,
    seed=1,

    rho_emp_ment=0.3,

    # effect sizes you asked about (defaults reflect your numbers)
    # EMP strongest: 0.3–0.4 (set 0.35 default)
    b_emp=0.4,
    b_ment=0.35,
    b_int=0.20,

    # intercepts + dispersion (tune in piloting)
    b0_logit=-0.2,
    b0_nb=1.0,
    alpha_nb=0.8,
    power_ci_level=0.95,

    # whether to assume same betas in both stages (simplest)
    same_betas_both_stages=True,

    out_dir="."
):
    rng = np.random.default_rng(seed)
    rows = []

    for N in N_values:
        # counters: how often each term is significant in each stage
        k_logit = {"EMP": 0, "MENT": 0, "EMPxMENT": 0}
        k_nb    = {"EMP": 0, "MENT": 0, "EMPxMENT": 0}

        engage_rates = []
        mean_completed_all = []
        mean_completed_eng = []

        for _ in range(nsim):
            sim_seed = int(rng.integers(1_000_000_000))
            r = np.random.default_rng(sim_seed)

            if same_betas_both_stages:
                df = simulate_hurdle_dataset(
                    N=N, rng=r,
                    rho_emp_ment=rho_emp_ment,
                    b0_logit=b0_logit,
                    b_emp_logit=b_emp,
                    b_ment_logit=b_ment,
                    b_int_logit=b_int,
                    b0_nb=b0_nb,
                    b_emp_nb=b_emp,
                    b_ment_nb=b_ment,
                    b_int_nb=b_int,
                    alpha_nb=alpha_nb
                )
                b_emp_nb_used = b_emp
                b_ment_nb_used = b_ment
                b_int_nb_used = b_int
            else:
                # If later you want different effects across stages, split them here.
                df = simulate_hurdle_dataset(
                    N=N, rng=r,
                    rho_emp_ment=rho_emp_ment,
                    b0_logit=b0_logit,
                    b_emp_logit=b_emp,
                    b_ment_logit=b_ment,
                    b_int_logit=b_int,
                    b0_nb=b0_nb,
                    b_emp_nb=b_emp,
                    b_ment_nb=b_ment,
                    b_int_nb=b_int,
                    alpha_nb=alpha_nb
                )
                b_emp_nb_used = b_emp
                b_ment_nb_used = b_ment
                b_int_nb_used = b_int

            engage_rates.append(df["engage"].mean())
            mean_completed_all.append(df["completed"].mean())
            if df["engage"].sum() > 0:
                mean_completed_eng.append(df.loc[df["engage"] == 1, "completed"].mean())
            else:
                mean_completed_eng.append(np.nan)

            # stage 1 p-values
            for term in ("EMP", "MENT", "EMPxMENT"):
                p = _fit_logit_pvalue(df, term)
                if np.isfinite(p) and p < alpha:
                    k_logit[term] += 1

            # stage 2 p-values
            for term in ("EMP", "MENT", "EMPxMENT"):
                p = _fit_nb_pvalue(df, term, alpha_nb=alpha_nb)
                if np.isfinite(p) and p < alpha:
                    k_nb[term] += 1

        # summarize
        for term in ("EMP", "MENT", "EMPxMENT"):
            p_logit = k_logit[term] / nsim
            p_nb = k_nb[term] / nsim
            p_logit_low, p_logit_high = _binom_wilson_ci(k_logit[term], nsim, level=power_ci_level)
            p_nb_low, p_nb_high = _binom_wilson_ci(k_nb[term], nsim, level=power_ci_level)
            rows.append({
                "N": int(N),
                "nsim": int(nsim),
                "alpha": float(alpha),

                "term": term,

                "power_logit": p_logit,
                "power_logit_ci_low": p_logit_low,
                "power_logit_ci_high": p_logit_high,

                "power_nb": p_nb,
                "power_nb_ci_low": p_nb_low,
                "power_nb_ci_high": p_nb_high,
                "power_ci_level": float(power_ci_level),

                "mean_engage_rate": float(np.mean(engage_rates)),
                "mean_completed_all": float(np.mean(mean_completed_all)),
                "mean_completed_engagers": float(np.nanmean(mean_completed_eng)),

                "rho_emp_ment": float(rho_emp_ment),
                "b0_logit": float(b0_logit),
                "b0_nb": float(b0_nb),
                "alpha_nb": float(alpha_nb),

                "b_emp": float(b_emp),
                "b_ment": float(b_ment),
                "b_int": float(b_int),
                "same_betas_both_stages": bool(same_betas_both_stages),
            })

    df_out = pd.DataFrame(rows)
    _ensure_dir(out_dir)
    stamp = _timestamp()

    csv_path = os.path.join(out_dir, f"01_power_hurdle_results_{stamp}.csv")
    df_out.to_csv(csv_path, index=False)

    # Write a simple text summary
    txt_path = os.path.join(out_dir, f"01_power_hurdle_summary_{stamp}.txt")
    with open(txt_path, "w") as f:
        f.write("HURDLE POWER SIMULATION SUMMARY\n")
        f.write(f"timestamp: {stamp}\n")
        f.write(f"N_values: {list(N_values)}\n")
        f.write(f"nsim: {nsim}\n")
        f.write(f"alpha: {alpha}\n")
        f.write(f"rho_emp_ment: {rho_emp_ment}\n")
        f.write(f"logit intercept b0_logit: {b0_logit}\n")
        f.write(f"NB intercept b0_nb: {b0_nb}\n")
        f.write(f"NB dispersion alpha_nb: {alpha_nb}\n")
        f.write(f"betas (EMP, MENT, INT): {b_emp}, {b_ment}, {b_int}\n")
        f.write("\nMean engage rate and completions (averaged over sims, varies by N):\n")
        f.write(df_out.groupby("N")[["mean_engage_rate", "mean_completed_all", "mean_completed_engagers"]].mean().to_string())
        f.write("\n\nPower table:\n")
        f.write(df_out.pivot_table(index=["N"], columns=["term"], values=["power_logit", "power_nb"]).to_string())
        f.write("\n")

    # Make power plots: one plot per term with two curves (logit vs NB)
    for term in ("EMP", "MENT", "EMPxMENT"):
        d = df_out[df_out["term"] == term].sort_values("N")
        x = d["N"].to_numpy()
        y1 = d["power_logit"].to_numpy()
        y2 = d["power_nb"].to_numpy()
        y1_lo = d["power_logit_ci_low"].to_numpy()
        y1_hi = d["power_logit_ci_high"].to_numpy()
        y2_lo = d["power_nb_ci_low"].to_numpy()
        y2_hi = d["power_nb_ci_high"].to_numpy()

        plt.figure()
        line1, = plt.plot(x, y1, marker="o", label="Logit: engage")
        line2, = plt.plot(x, y2, marker="o", label="NB: completions | engaged")
        plt.fill_between(x, y1_lo, y1_hi, alpha=0.2, color=line1.get_color(), label=f"Logit {int(power_ci_level*100)}% CI")
        plt.fill_between(x, y2_lo, y2_hi, alpha=0.2, color=line2.get_color(), label=f"NB {int(power_ci_level*100)}% CI")
        plt.axhline(0.80, linestyle="--", label="0.80 target")
        plt.ylim(0, 1.02)
        plt.xlabel("N")
        plt.ylabel("Power")
        plt.title(f"Power curves (term = {term})\n"
                  f"betas: EMP={b_emp}, MENT={b_ment}, INT={b_int} | rho={rho_emp_ment} | nsim={nsim}")
        plt.legend()
        plt.tight_layout()

        png_path = os.path.join(out_dir, f"01_power_hurdle_curve_{term}_{stamp}.png")
        plt.savefig(png_path, dpi=180)
        plt.close()

    print("\n=== DONE ===")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved TXT: {txt_path}")
    print(f"Saved PNGs: 01_power_hurdle_curve_(EMP|MENT|EMPxMENT)_{stamp}.png in {out_dir}\n")

    # print a compact terminal summary
    print(df_out.pivot_table(index=["N"], columns=["term"], values=["power_logit", "power_nb"]).round(3))
    print("\nMean engagement/completions by N:")
    print(df_out.groupby("N")[["mean_engage_rate", "mean_completed_all", "mean_completed_engagers"]].mean().round(3))

    return df_out, csv_path, txt_path


# =========================
# Main (edit these defaults as you like)
# =========================
if __name__ == "__main__":
    # Put outputs in the same folder as this script by default
    here = os.path.dirname(os.path.abspath(__file__))

    # Example settings aligned with what you asked:
    # EMP ~ 0.35, MENT ~ 0.30, EMPxMENT ~ 0.20
    run_power_hurdle(
        N_values=(100, 130, 160, 190, 220, 250),
        nsim=1000,
        alpha=0.05,
        seed=1,

        rho_emp_ment=0.3,

        b_emp=0.4,   # you said 0.3–0.4 likely for EMP
        b_ment=0.35,  # you want to test underpower for MENT at 0.3
        b_int=0.20,   # you want to test underpower for EMPxMENT at 0.2

        # Tune these based on piloting:
        b0_logit=-0.2, # baseline engage probability
        b0_nb=1.0,     # baseline mean completions among engagers ~ exp(1)=2.7
        alpha_nb=0.8,  # dispersion

        same_betas_both_stages=True,
        out_dir=here
    )
