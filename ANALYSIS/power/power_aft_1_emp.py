import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter


# ---------- CI helper (Wilson) ----------
def _norm_ppf_approx(p):
    """
    Approximate inverse CDF for standard normal.
    (Acklam approximation) – good for typical CI levels (e.g., 90/95/99%).
    """
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
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (np.nan, np.nan)
    z = _norm_ppf_approx(1 - ci_alpha/2.0)
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    return (center - half, center + half)


# ---------- data generator ----------
def simulate_weibull_time_with_admin_censoring(
    N,
    t_cap=900.0,
    shape=1.5,
    scale0=300.0,
    HR_E=0.75,
    seed=1,
):
    """
    Simulate Weibull survival times with a covariate effect specified on the PH scale via HR_E.
    This is fine for AFT power too because Weibull(PH) <-> Weibull(AFT) are equivalent model families.

    Empathy ~ N(0,1)
    HR_E < 1 => lower hazard => longer times (more persistence)
    """
    rng = np.random.default_rng(seed)

    Empathy = rng.normal(size=N)
    bE = np.log(HR_E)          # Cox/log-hazard coefficient
    eta = bE * Empathy

    # Weibull PH simulation:
    U = rng.uniform(size=N)
    T0 = scale0 * (-np.log(U))**(1.0 / shape)
    T_true = T0 / (np.exp(eta)**(1.0 / shape))

    time = np.minimum(T_true, t_cap)
    event = (T_true <= t_cap).astype(int)

    return pd.DataFrame({"time": time, "event": event, "Empathy": Empathy})


# ---------- extract p-value safely ----------
def _get_empathy_pvalue_weibull_aft(aft):
    """
    lifelines WeibullAFTFitter stores covariate effects under the 'lambda_' parameter (time scale).
    The summary index is typically a MultiIndex like ('lambda_', 'Empathy').
    """
    summ = aft.summary

    # Common case: MultiIndex (param, covariate)
    if isinstance(summ.index, pd.MultiIndex):
        key = ("lambda_", "Empathy")
        if key in summ.index:
            return float(summ.loc[key, "p"])

    # Fallback: search for a row containing Empathy and lambda_
    # (handles rare formatting differences)
    for idx in summ.index:
        if isinstance(idx, tuple) and len(idx) >= 2:
            if str(idx[0]) == "lambda_" and str(idx[1]) == "Empathy":
                return float(summ.loc[idx, "p"])

    return np.nan


# ---------- power curve ----------
def power_curve_weibull_aft_empathy(
    #N_values=(50, 75, 100, 125, 150, 200),
    N_values=(50, 100, 200),
    nsim=100,
    t_cap=900.0,
    shape=1.5,
    scale0=300.0,
    HR_E=0.75,
    alpha_test=0.05,
    ci_alpha=0.05,
    seed=42,
    robust=True,
):
    """
    For each N:
      - simulate nsim datasets
      - fit Weibull AFT: time,event ~ Empathy
      - compute power = P(p_Empathy < alpha_test)
      - Wilson CI + mean event rate
    """
    rows = []

    for N in N_values:
        N = int(N)
        rng_N = np.random.default_rng(seed + N * 1_000_003)

        k_sig = 0
        n_fit = 0
        event_rates = []

        for _ in range(nsim):
            df = simulate_weibull_time_with_admin_censoring(
                N=N,
                t_cap=t_cap,
                shape=shape,
                scale0=scale0,
                HR_E=HR_E,
                seed=int(rng_N.integers(1_000_000_000)),
            )
            event_rates.append(float(df["event"].mean()))

            aft = WeibullAFTFitter()
            try:
                aft.fit(df, duration_col="time", event_col="event", robust=robust)

                p = _get_empathy_pvalue_weibull_aft(aft)
                if not np.isnan(p):
                    n_fit += 1
                    if p < alpha_test:
                        k_sig += 1
            except Exception:
                # rare convergence issues; skip replicate
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
            "term": "Empathy",
            "model": "WeibullAFT",
            "HR_E": float(HR_E),
            "t_cap": float(t_cap),
            "shape": float(shape),
            "scale0": float(scale0),
            "alpha_test": float(alpha_test),
            "ci_alpha": float(ci_alpha),
        })

    return df, nsim, pd.DataFrame(rows)  # df = df_sim

# ---------- intelligible interpretation ----------

def implied_time_ratio_from_aft(aft, covariate="Empathy", verbose=True):
    """
    Extract the AFT coefficient for `covariate` from a fitted WeibullAFTFitter
    and compute the implied time ratio exp(beta).

    Returns
    -------
    time_ratio : float
        Multiplicative change in expected survival time per 1 SD increase
        in the covariate.
    beta_aft : float
        The AFT coefficient on log-time scale.
    """
    summ = aft.summary

    beta_aft = None

    # Standard lifelines layout: MultiIndex ('lambda_', covariate)
    if isinstance(summ.index, pd.MultiIndex):
        key = ("lambda_", covariate)
        if key in summ.index:
            beta_aft = float(summ.loc[key, "coef"])

    # Fallback (defensive)
    if beta_aft is None:
        for idx in summ.index:
            if isinstance(idx, tuple) and len(idx) >= 2:
                if str(idx[0]) == "lambda_" and str(idx[1]) == covariate:
                    beta_aft = float(summ.loc[idx, "coef"])
                    break

    if beta_aft is None:
        raise ValueError(f"Could not find AFT coefficient for '{covariate}'")

    time_ratio = float(np.exp(beta_aft))

    if verbose:
        pct = (time_ratio - 1.0) * 100
        direction = "longer" if time_ratio > 1 else "shorter"
        print(
            f"AFT time ratio for {covariate} (1 SD): {time_ratio:.3f}\n"
            f"→ Expected persistence time is {abs(pct):.1f}% {direction} "
            f"per 1 SD increase in {covariate}."
        )

    return time_ratio, beta_aft
# ---------- plot ----------
def plot_power_curve(df, nsim, target=0.80, title=None):
    df = df.sort_values("N")

    x = df["N"].to_numpy()
    y = df["power_hat"].to_numpy()
    lo = df["power_ci_low_95"].to_numpy()
    hi = df["power_ci_high_95"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.fill_between(x, lo, hi, alpha=0.2)

    if target is not None:
        plt.axhline(target, linestyle="--")
        plt.text(x.min(), target + 0.01, f"Target={target:.2f}", va="bottom")

    plt.ylim(0, 1.02)
    plt.xlabel("N")
    plt.ylabel(f"Power ({nsim} simulations)")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------- example ----------
if __name__ == "__main__":
    N_values = [50, 65, 70, 75, 100, 125, 150]
    df_sim, nsim, df_power = power_curve_weibull_aft_empathy(
                            N_values=N_values,
                            nsim=100,    # raise to 5000+ for tighter Monte Carlo CIs
                            HR_E=0.75, # 21% longer persistence time per SD of empathy. a moderate, theoretically grounded, preregistrable effect.
                            t_cap=900.0, # maximum seconds (15 minutes) Anyone still engaged at 900s is right-censored.
                            shape=1.5, # consider boosting to see what happens: shape = 1.0 → constant hazard (memoryless); shape > 1.0 → increasing hazard (fatigue / boredom)
                            scale0=300.0, # persistence when *z-distributed* empathy is at 0 (average empathy, not no empathy) (5 minutes), perhaps too high
                            alpha_test=0.05,
                            ci_alpha=0.05,
                            seed=42,
                            robust=True,
                        )

    print(df_power)
 
    plot_power_curve(
        df_power, nsim,
        target=0.80,
        title="Weibull AFT power curve (Empathy only)"
    )

    aft = WeibullAFTFitter()
    aft.fit(df_sim, duration_col="time", event_col="event", robust=True)
    time_ratio, beta_aft = implied_time_ratio_from_aft(aft)
