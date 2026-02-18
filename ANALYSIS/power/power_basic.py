import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import math
import matplotlib.pyplot as plt
import time
def simulate_linear_regression_interaction(
    N=250,                          # Sample size
    beta1=0.35,                     # β₁: positive effect of empathy
    beta2=0.30,                     # β₂: positive effect of mentacy confidence
    beta3=0.25,                     # β₃: positive interaction (key hypothesis)
    sigma=1.0,                      # Residual standard deviation
    cor_x1_x2=0.3,                  # Correlation between empathy and mentacy
    seed=42
    ):
    rng = np.random.default_rng(seed)
    
    # Generate correlated standard normals
    Sigma = np.array([[1.0, cor_x1_x2],
                      [cor_x1_x2, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Z = rng.normal(size=(N, 2))
    X = Z @ L.T
    
    empathy = X[:, 0]                # Standardized IV_EMPATHY-like
    mentacy_conf = X[:, 1]           # Standardized continuous mentacy proxy
    
    # Raw interaction
    inter_raw = empathy * mentacy_conf
    
    # Continuous outcome y (higher = more effort/time in tedious acts)
    y = (beta1 * empathy + 
         beta2 * mentacy_conf + 
         beta3 * inter_raw + 
         rng.normal(scale=sigma, size=N))
    
    # DataFrame
    df = pd.DataFrame({
        'empathy': empathy,
        'mentacy_conf': mentacy_conf,
        'y': y
    })
    
    # Ensure numeric (safe even if loaded from real csv)
    df['empathy'] = pd.to_numeric(df['empathy'], errors='coerce')
    df['mentacy_conf'] = pd.to_numeric(df['mentacy_conf'], errors='coerce')
    
    # Center predictors for clean interpretation and reduced collinearity
    df['empathy_c'] = df['empathy'] - df['empathy'].mean()
    df['mentacy_conf_c'] = df['mentacy_conf'] - df['mentacy_conf'].mean()
    df['interaction_c'] = df['empathy_c'] * df['mentacy_conf_c']
    
    return df


def power_analysis_linear_regression_interaction(
    N_values=[100, 125, 150, 200, 250, 300],
    nsim=1000,                      # Increase for more precise estimates
    beta1=1.5, beta2=1.5, beta3=0.15,
    cor_x1_x2=0.25, sigma=1.0,
    alpha=0.05, seed=42
):
    rng = np.random.default_rng(seed)
    power_results = []
    
    for N in N_values:
        pvals_inter = []
        
        for _ in range(nsim):
            # Use varying seed per simulation
            df = simulate_linear_regression_interaction(
                N=N, beta1=beta1, beta2=beta2, beta3=beta3,
                cor_x1_x2=cor_x1_x2, sigma=sigma,
                seed=rng.integers(1_000_000)
            )
            
            # Fit confirmatory model (centering already done in function)
            model = smf.ols('y ~ empathy_c + mentacy_conf_c + interaction_c', data=df).fit()
            pvals_inter.append(model.pvalues['interaction_c'])
        
        power = np.mean(np.array(pvals_inter) < alpha)
        power_results.append({'N': N, 'power_interaction': power})
    
    return pd.DataFrame(power_results)


# # ==================== Example 1: Single simulation and fit ====================
# df_sim = simulate_linear_regression_interaction(N=300, beta3=0.20)

# model_single = smf.ols('y ~ empathy_c + mentacy_conf_c + interaction_c', data=df_sim).fit()

# print("=== Single Simulation Model Summary ===")
# print(model_single.summary())

# print("\nRecovered coefficients vs. True values:")
# recovered = model_single.params.round(3)
# true = pd.Series({'Intercept': 0.0, 'empathy_c': 0.30, 'mentacy_conf_c': 0.25, 'interaction_c': 0.20})
# print(pd.DataFrame({'Recovered': recovered, 'True': true.reindex(recovered.index).fillna(0)}))


# # ==================== Example 2: Power analysis for the interaction ====================
# power_df = power_analysis_linear_regression_interaction(
#     N_values=[150, 200, 250, 300, 350, 400],
#     nsim=1000  # Adjust based on runtime vs. precision needs
# )

# print("\n=== Power for detecting the empathy × mentacy interaction (β₃) ===")
# print(power_df.round(3))


# # ==================== Plot power curve ====================
# plt.figure(figsize=(8, 5))
# plt.plot(power_df['N'], power_df['power_interaction'], marker='o', linestyle='-')
# plt.axhline(0.80, color='red', linestyle='--', label='80% Target')
# plt.axhline(0.90, color='orange', linestyle='--', label='90% Target')
# plt.xlabel('Sample Size (N)')
# plt.ylabel('Power for Interaction Term')
# plt.title('Power Curve: Detecting Positive Empathy × Mentacy Interaction\n'
#           'in Costly Robot-Oriented Altruism (Continuous Outcome)')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # ==================== Plot power curve ====================
# plt.figure(figsize=(8, 5))
# plt.plot(power_df['N'], power_df['power_interaction'], marker='o', linestyle='-')
# plt.axhline(0.80, color='red', linestyle='--', label='80% Target')
# plt.axhline(0.90, color='orange', linestyle='--', label='90% Target')
# plt.xlabel('Sample Size (N)')
# plt.ylabel('Power for Interaction Term')
# plt.title('Power Curve: Detecting Positive Empathy × Mentacy Interaction\n'
#           'in Costly Robot-Oriented Altruism (Continuous Outcome)')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()



def simulate_linear_regression_main_effects(
    N=250,
    beta1=0.30,                 # empathy effect
    beta2=0.30,                 # mentacy effect
    sigma=1.0,                  # residual SD
    cor_x1_x2=0.25,
    seed=42
):
    rng = np.random.default_rng(seed)

    # Correlated standard normals
    Sigma = np.array([[1.0, cor_x1_x2],
                      [cor_x1_x2, 1.0]])
    L = np.linalg.cholesky(Sigma)
    Z = rng.normal(size=(N, 2))
    X = Z @ L.T

    empathy = X[:, 0]
    mentacy_conf = X[:, 1]

    # Outcome with main effects only
    y = (beta1 * empathy +
         beta2 * mentacy_conf +
         rng.normal(scale=sigma, size=N))

    df = pd.DataFrame({
        "empathy": empathy,
        "mentacy_conf": mentacy_conf,
        "y": y
    })

    # Center predictors (optional but fine)
    df["empathy_c"] = df["empathy"] - df["empathy"].mean()
    df["mentacy_conf_c"] = df["mentacy_conf"] - df["mentacy_conf"].mean()

    return df



# import numpy as np
# import pandas as pd
# import statsmodels.formula.api as smf

# def power_analysis_linear_regression_main_effects_stable(
#     N_values=(50, 75, 100, 125, 150, 200),
#     nsim=2000,
#     beta1=0.30, beta2=0.30,
#     cor_x1_x2=0.25, sigma=1.0,
#     alpha=0.05, seed=42
# ):
#     rows = []

#     for N in N_values:
#         # Create an RNG *specific to this N* so results for N don't depend on other Ns
#         rng_N = np.random.default_rng(seed + int(N) * 1_000_003)

#         p1, p2 = [], []

#         for _ in range(nsim):
#             df = simulate_linear_regression_main_effects(
#                 N=N, beta1=beta1, beta2=beta2,
#                 cor_x1_x2=cor_x1_x2, sigma=sigma,
#                 seed=int(rng_N.integers(1_000_000_000))
#             )

#             model = smf.ols("y ~ empathy_c + mentacy_conf_c", data=df).fit()
#             p1.append(model.pvalues["empathy_c"])
#             p2.append(model.pvalues["mentacy_conf_c"])

#         rows.append({
#             "N": int(N),
#             "power_beta1_empathy": float(np.mean(np.array(p1) < alpha)),
#             "power_beta2_mentacy": float(np.mean(np.array(p2) < alpha)),
#         })

#     return pd.DataFrame(rows)



# # ==================== Example 4: Power analysis for the main effects ====================
# power_df = power_analysis_linear_regression_main_effects_stable(
#     N_values=[75, 85, 95, 100],
#     nsim=1000  # Adjust based on runtime vs. precision needs
# )

# print("\n=== Power for detecting empathy and mentacy (β1, β2 only) ===")
# # print(power_df.round(3))
# print(power_df)


#======================

def wilson_ci(k, n, alpha=0.05):
    # Wilson score interval for a binomial proportion
    if n == 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # ~N(0,1) 97.5% quantile for 95% CI
    phat = k / n
    denom = 1 + (z**2)/n
    center = (phat + (z**2)/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat) + (z**2)/(4*n)) / n)) / denom
    return (center - half, center + half)

def power_curve_with_ci_interaction(
    nsim=100, #do up to 100,000
    alpha_test=0.05,
    seed=42,
    which="mentacy",  # "interaction", "empathy", "mentacy"
    # DGP params
    beta1=0.4, beta2=0.3, beta3=0.15,
    cor_x1_x2=0.3, sigma=1.0
):

    N_values = []
    if which == "interaction":
        N_values = [200, 250, 300, 350, 400]
    elif which == "empathy":
        N_values = [50, 75, 100, 125, 150]
    elif which == "mentacy":
        N_values = [50, 75, 100, 125, 150]

    rows = []
    time_start = time.time()
    for N in N_values:
        rng_N = np.random.default_rng(seed + int(N)*1_000_003)

        k = 0
        for _ in range(nsim):
            df = simulate_linear_regression_interaction(
                N=int(N),
                beta1=beta1, beta2=beta2, beta3=beta3,
                cor_x1_x2=cor_x1_x2, sigma=sigma,
                seed=int(rng_N.integers(1_000_000_000))
            )

            model = smf.ols("y ~ empathy_c + mentacy_conf_c + interaction_c", data=df).fit()

            if which == "interaction":
                p = model.pvalues["interaction_c"]
            elif which == "empathy":
                p = model.pvalues["empathy_c"]
            elif which == "mentacy":
                p = model.pvalues["mentacy_conf_c"]
            else:
                raise ValueError("which must be 'interaction', 'empathy', or 'mentacy'")

            if p < alpha_test:
                k += 1

        pwr = k / nsim
        lo, hi = wilson_ci(k, nsim, alpha=0.05)

        rows.append({
            "N": int(N),
            "power_hat": float(pwr),
            "power_ci_low_95": float(lo),
            "power_ci_high_95": float(hi),
            "nsim": int(nsim),
            "k_sig": int(k),
            "term": which
        })
    time_finish = time.time()
    sim_duration = (time_finish - time_start)/60
    print(f"\nSIMULATION DURATION: {sim_duration} minutes")

    return pd.DataFrame(rows), nsim, which, beta1, beta2, beta3, cor_x1_x2


N_values = [100, 150, 200, 250, 300, 350, 400]
power_df, nsim, which, beta1, beta2, beta3, cor_x1_x2 = power_curve_with_ci_interaction()
print(f"\n*** Power curve with CI for detecting {which} \n β_emp = {beta1}, β_ment = {beta2}, β_inter = {beta3}, corr = {cor_x1_x2} ***")
print(power_df)


import matplotlib.pyplot as plt

def plot_power_curve(df, nsim, title=None, show_target=0.80):
    """
    Plot power_hat vs N with a 95% CI band.

    df must have columns:
      - N
      - power_hat
      - power_ci_low_95
      - power_ci_high_95
    """
    df = df.sort_values("N")

    x = df["N"].to_numpy()
    y = df["power_hat"].to_numpy()
    lo = df["power_ci_low_95"].to_numpy()
    hi = df["power_ci_high_95"].to_numpy()

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.fill_between(x, lo, hi, alpha=0.2)

    if show_target is not None:
        plt.axhline(show_target, linestyle="--")
        plt.text(x.min(), show_target + 0.01, "", va="bottom")

    plt.ylim(0, 1.02)
    plt.xlabel("N")
    plt.ylabel(f"Power (simulations = {nsim})")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

insert = ""
if which == "interaction":
    insert = "Empathy x Mentacy Belief Confidence"
elif which == "empathy":
    insert = "Empathy"
elif which == "mentacy":
    insert = "Mentacy Belief Confidence"
title = f"Power Curve for Effect of {insert}"

plot_power_curve(power_df, nsim, title=title)