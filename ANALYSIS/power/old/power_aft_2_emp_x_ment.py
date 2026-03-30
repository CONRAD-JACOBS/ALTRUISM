import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter


# ---------- CI helper (Wilson) ----------
def _norm_ppf_approx(p):
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def wilson_ci(k, n, ci_alpha=0.05):
    if n <= 0:
        return (np.nan, np.nan)
    z = _norm_ppf_approx(1 - ci_alpha/2.0)
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    return (center - half, center + half)


# ---------- data generator (2 predictors + interaction) ----------
def simulate_weibull_time_with_admin_censoring_empathy_mentacy(
    N,
    t_cap=900.0,
    shape=1.5,
    scale0=300.0,
    cor_EM=0.30,
    HR_E=0.75,
    HR_M=0.85,
    HR_EM=1.00,
    center_predictors=True,
    seed=1,
):
    """
    Simulate Weibull survival times under a Weibull PH data-generating process (equivalent family to Weibull AFT).
    Predictors:
      Empathy ~ N(0,1)
      Mentacy ~ N(0,1) correlated with Empathy by cor_EM
      EM_int = (Empathy_c * Mentacy_c)

    Effects are specified as hazard ratios per 1 SD:
      HR_E  < 1 => longer persistence
      HR_M  < 1 => longer persistence
      HR_EM < 1 => positive synergistic interaction on persistence (on hazard scale)

    Note: Interactions on the hazard scale are a modeling choice; you can also set HR_EM=1.0 (no interaction).
    """
    rng = np.random.default_rng(seed)

    # correlated normals for Empathy and Mentacy
    Sigma = np.array([[1.0, cor_EM],
                      [cor_EM, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Z = rng.normal(size=(N, 2))
    EM = Z @ L.T
    Empathy = EM[:, 0]
    Mentacy = EM[:, 1]

    if center_predictors:
        Empathy_c = Empathy - Empathy.mean()
        Mentacy_c = Mentacy - Mentacy.mean()
    else:
        Empathy_c = Empathy
        Mentacy_c = Mentacy

    EM_int = Empathy_c * Mentacy_c

    # log-hazard coefficients
    bE = np.log(HR_E)
    bM = np.log(HR_M)
    bEM = np.log(HR_EM)

    eta = bE * Empathy_c + bM * Mentacy_c + bEM * EM_int

    # Weibull PH simulation
    U = rng.uniform(size=N)
    T0 = scale0 * (-np.log(U))**(1.0 / shape)
    T_true = T0 / (np.exp(eta)**(1.0 / shape))

    time = np.minimum(T_true, t_cap)
    event = (T_true <= t_cap).astype(int)

    return pd.DataFrame({
        "time": time,
        "event": event,
        "Empathy": Empathy_c,
        "Mentacy": Mentacy_c,
        "EM_int": EM_int,
    })


# ---------- extract p-value safely (generic) ----------
def _get_pvalue_weibull_aft(aft, covariate):
    """
    For lifelines WeibullAFTFitter, covariate effects typically live under ('lambda_', covariate).
    """
    summ = aft.summary

    if isinstance(summ.index, pd.MultiIndex):
        key = ("lambda_", covariate)
        if key in summ.index:
            return float(summ.loc[key, "p"])

    # fallback
    for idx in summ.index:
        if isinstance(idx, tuple) and len(idx) >= 2:
            if str(idx[0]) == "lambda_" and str(idx[1]) == covariate:
                return float(summ.loc[idx, "p"])

    return np.nan


# ---------- intelligible interpretation ----------
def implied_time_ratio_from_aft(aft, covariate="Empathy", verbose=True):
    summ = aft.summary
    beta_aft = None

    if isinstance(summ.index, pd.MultiIndex):
        key = ("lambda_", covariate)
        if key in summ.index:
            beta_aft = float(summ.loc[key, "coef"])

    if beta_aft is None:
        for idx in summ.index:
            if isinstance(idx, tuple) and len(idx) >= 2:
                if str(idx[0]) == "lambda_" and str(idx[1]) == covariate:
                    beta_aft = float(summ.loc[idx, "coef"])
                    break

    if beta_aft is None:
        raise ValueError("Could not find AFT coefficient for '{}'".format(covariate))

    time_ratio = float(np.exp(beta_aft))

    if verbose:
        pct = (time_ratio - 1.0) * 100
        direction = "longer" if time_ratio > 1 else "shorter"
        print(
            "AFT time ratio for {} (1 SD): {:.3f}\n"
            "→ Expected persistence time is {:.1f}% {} per 1 SD increase in {}."
            .format(covariate, time_ratio, abs(pct), direction, covariate)
        )

    return time_ratio, beta_aft


# ---------- power curve (2 predictors + interaction) ----------
def power_curve_weibull_aft_empathy_mentacy_interaction(
    N_values=(50, 75, 100, 125, 150, 200),
    nsim=100,
    t_cap=900.0,
    shape=1.5,
    scale0=300.0,
    cor_EM=0.30,
    HR_E=0.75,
    HR_M=0.85,
    HR_EM=0.90,
    which="Empathy",          # "Empathy", "Mentacy", or "EM_int"
    alpha_test=0.05,
    ci_alpha=0.05,
    seed=42,
    robust=True,
    center_predictors=True,
):
    """
    For each N:
      - simulate nsim datasets with Empathy, Mentacy, and interaction
      - fit Weibull AFT: time,event ~ Empathy + Mentacy + EM_int
      - compute power for the chosen term
      - Wilson CI + mean event rate

    Returns:
      df_last_sim, nsim, df_power
    """
    rows = []
    df_last_sim = None

    for N in N_values:
        N = int(N)
        rng_N = np.random.default_rng(seed + N * 1_000_003)

        k_sig = 0
        n_fit = 0
        event_rates = []

        for _ in range(nsim):
            df_sim = simulate_weibull_time_with_admin_censoring_empathy_mentacy(
                N=N,
                t_cap=t_cap,
                shape=shape,
                scale0=scale0,
                cor_EM=cor_EM,
                HR_E=HR_E,
                HR_M=HR_M,
                HR_EM=HR_EM,
                center_predictors=center_predictors,
                seed=int(rng_N.integers(1_000_000_000)),
            )
            df_last_sim = df_sim  # keep last for inspection
            event_rates.append(float(df_sim["event"].mean()))

            aft = WeibullAFTFitter()
            try:
                aft.fit(df_sim, duration_col="time", event_col="event", robust=robust)
                p = _get_pvalue_weibull_aft(aft, which)

                if not np.isnan(p):
                    n_fit += 1
                    if p < alpha_test:
                        k_sig += 1
            except Exception:
                pass

        power_hat = (k_sig / n_fit) if n_fit > 0 else np.nan
        lo, hi = wilson_ci(k_sig, n_fit, ci_alpha=ci_alpha) if n_fit > 0 else (np.nan, np.nan)

        rows.append({
            "N": N,
            "power_hat": float(power_hat),
            "power_ci_low_95": float(lo),
            "power_ci_high_95": float(hi),
            "nsim": int(nsim),
            "n_fit": int(n_fit),
            "k_sig": int(k_sig),
            "mean_event_rate": float(np.mean(event_rates)) if event_rates else np.nan,
            "term": which,
            "model": "WeibullAFT",
            "HR_E": float(HR_E),
            "HR_M": float(HR_M),
            "HR_EM": float(HR_EM),
            "cor_EM": float(cor_EM),
            "t_cap": float(t_cap),
            "shape": float(shape),
            "scale0": float(scale0),
            "alpha_test": float(alpha_test),
            "ci_alpha": float(ci_alpha),
        })

    return df_last_sim, int(nsim), which, pd.DataFrame(rows)


# ---------- plot ----------
def plot_power_curve(df_power, nsim, target=0.80, title=None):
    df_power = df_power.sort_values("N")

    x = df_power["N"].to_numpy()
    y = df_power["power_hat"].to_numpy()
    lo = df_power["power_ci_low_95"].to_numpy()
    hi = df_power["power_ci_high_95"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.fill_between(x, lo, hi, alpha=0.2)

    if target is not None:
        plt.axhline(target, linestyle="--")
        plt.text(x.min(), target + 0.01, "Target={:.2f}".format(target), va="bottom")

    plt.ylim(0, 1.02)
    plt.xlabel("N")
    plt.ylabel("Power ({} simulations)".format(nsim))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------- example ----------
if __name__ == "__main__":
    N_values = [200, 250, 300, 350, 400]

    df_sim, nsim, which, df_power = power_curve_weibull_aft_empathy_mentacy_interaction(
        N_values=N_values,
        nsim=100,       # run at 10,000 on mac
        HR_E=0.75,     # stronger effect
        HR_M=0.85,     # weaker effect
        HR_EM=1.00,    # set <1 for synergistic interaction on persistence (hazard reduction)
        cor_EM=0.30,
        t_cap=900.0,
        shape=1.5,
        scale0=300.0,
        which="Mentacy",          # "Empathy" or "Mentacy" or "EM_int"
        alpha_test=0.05,
        ci_alpha=0.05,
        seed=42,
        robust=True,
        center_predictors=True,
    )

    print(df_power)

    # Interpret a fitted model from the last simulated dataset
    aft = WeibullAFTFitter()
    aft.fit(df_sim, duration_col="time", event_col="event", robust=True)
    implied_time_ratio_from_aft(aft, covariate="Empathy")
    implied_time_ratio_from_aft(aft, covariate="Mentacy")
    # interaction coefficient is also interpretable on the log-time scale, but not as a simple 1-SD shift


    insert = ""
    if which == "EM_int":
        insert = "Empathy x Mentacy Belief Confidence"
    elif which == "Empathy":
        insert = "Empathy"
    elif which == "Mentacy":
        insert = "Mentacy Belief Confidence"
    title = f"Weibull AFT Power Curve for {insert}"
    
    plot_power_curve(
    df_power, nsim,
    target=0.80,
    title=title
    )

