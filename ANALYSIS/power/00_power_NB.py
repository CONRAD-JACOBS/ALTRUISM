import os
import time
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _zscore(x):
    x = np.asarray(x)
    return (x - x.mean()) / (x.std(ddof=0) + 1e-12)


def _simulate_correlated_normals(n, rho, rng):
    sigma = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(sigma)
    z = rng.normal(size=(n, 2))
    x = z @ L.T
    return x[:, 0], x[:, 1]


def _nb2_rng(mu, alpha, rng):
    """
    Draw from NB2 with mean=mu and var=mu + alpha*mu^2
    using Gamma-Poisson mixture:
      lambda ~ Gamma(shape=1/alpha, scale=alpha*mu)
      y ~ Poisson(lambda)
    """
    mu = np.asarray(mu)
    mu = np.clip(mu, 1e-12, None)

    if alpha <= 0:
        return rng.poisson(mu)

    shape = 1.0 / alpha
    scale = alpha * mu
    lam = rng.gamma(shape=shape, scale=scale)
    return rng.poisson(lam)


def _binom_wilson_ci(k, n, level=0.95):
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


def _fit_nb_pvalue(df, term, alpha_nb):
    try:
        X = df[["EMP", "MENT", "EMPxMENT"]]
        X = sm.add_constant(X, has_constant="add")
        y = df["completed"].astype(float)

        fam = sm.families.NegativeBinomial(alpha=alpha_nb)
        model = sm.GLM(y, X, family=fam).fit()
        return float(model.pvalues.get(term, np.nan))
    except Exception:
        return np.nan


def simulate_nb_dataset(
    N,
    rng,
    rho_emp_ment=0.3,
    b0_nb=1.0,
    b_emp_nb=0.4,
    b_ment_nb=0.35,
    b_int_nb=0.20,
    alpha_nb=0.8,
):
    emp, ment = _simulate_correlated_normals(N, rho_emp_ment, rng)
    emp = _zscore(emp)
    ment = _zscore(ment)
    inter = emp * ment

    lin = b0_nb + b_emp_nb * emp + b_ment_nb * ment + b_int_nb * inter
    mu = np.exp(lin)
    completed = _nb2_rng(mu, alpha=alpha_nb, rng=rng).astype(int)

    return pd.DataFrame(
        {
            "EMP": emp,
            "MENT": ment,
            "EMPxMENT": inter,
            "completed": completed,
            "mu_true": mu,
        }
    )


def run_power_nb(
    N_values=(100, 130, 160, 190, 220, 250),
    nsim=1000,
    alpha=0.05,
    seed=1,
    rho_emp_ment=0.3,
    b_emp=0.4,
    b_ment=0.35,
    b_int=0.20,
    b0_nb=1.0,
    alpha_nb=0.8,
    power_ci_level=0.95,
    out_dir=".",
):
    rng = np.random.default_rng(seed)
    rows = []

    for N in N_values:
        k_nb = {"EMP": 0, "MENT": 0, "EMPxMENT": 0}
        mean_completed = []
        mean_mu_true = []

        for _ in range(nsim):
            sim_seed = int(rng.integers(1_000_000_000))
            r = np.random.default_rng(sim_seed)

            df = simulate_nb_dataset(
                N=N,
                rng=r,
                rho_emp_ment=rho_emp_ment,
                b0_nb=b0_nb,
                b_emp_nb=b_emp,
                b_ment_nb=b_ment,
                b_int_nb=b_int,
                alpha_nb=alpha_nb,
            )

            mean_completed.append(df["completed"].mean())
            mean_mu_true.append(df["mu_true"].mean())

            for term in ("EMP", "MENT", "EMPxMENT"):
                p = _fit_nb_pvalue(df, term, alpha_nb=alpha_nb)
                if np.isfinite(p) and p < alpha:
                    k_nb[term] += 1

        for term in ("EMP", "MENT", "EMPxMENT"):
            p_nb = k_nb[term] / nsim
            p_nb_low, p_nb_high = _binom_wilson_ci(k_nb[term], nsim, level=power_ci_level)
            rows.append(
                {
                    "N": int(N),
                    "nsim": int(nsim),
                    "alpha": float(alpha),
                    "term": term,
                    "power_nb": p_nb,
                    "power_nb_ci_low": p_nb_low,
                    "power_nb_ci_high": p_nb_high,
                    "power_ci_level": float(power_ci_level),
                    "mean_completed": float(np.mean(mean_completed)),
                    "mean_mu_true": float(np.mean(mean_mu_true)),
                    "rho_emp_ment": float(rho_emp_ment),
                    "b0_nb": float(b0_nb),
                    "alpha_nb": float(alpha_nb),
                    "b_emp": float(b_emp),
                    "b_ment": float(b_ment),
                    "b_int": float(b_int),
                }
            )

    df_out = pd.DataFrame(rows)
    _ensure_dir(out_dir)
    stamp = _timestamp()

    csv_path = os.path.join(out_dir, "00_power_nb_results_{}.csv".format(stamp))
    txt_path = os.path.join(out_dir, "00_power_nb_summary_{}.txt".format(stamp))
    df_out.to_csv(csv_path, index=False)

    with open(txt_path, "w") as f:
        f.write("NEGATIVE BINOMIAL POWER SIMULATION SUMMARY\n")
        f.write("timestamp: {}\n".format(stamp))
        f.write("N_values: {}\n".format(list(N_values)))
        f.write("nsim: {}\n".format(nsim))
        f.write("alpha: {}\n".format(alpha))
        f.write("rho_emp_ment: {}\n".format(rho_emp_ment))
        f.write("NB intercept b0_nb: {}\n".format(b0_nb))
        f.write("NB dispersion alpha_nb: {}\n".format(alpha_nb))
        f.write("betas (EMP, MENT, INT): {}, {}, {}\n".format(b_emp, b_ment, b_int))
        f.write("\nMean outcomes by N:\n")
        f.write(df_out.groupby("N")[["mean_completed", "mean_mu_true"]].mean().to_string())
        f.write("\n\nPower table (NB):\n")
        f.write(df_out.pivot_table(index=["N"], columns=["term"], values=["power_nb"]).to_string())
        f.write("\n")

    for term in ("EMP", "MENT", "EMPxMENT"):
        d = df_out[df_out["term"] == term].sort_values("N")
        x = d["N"].to_numpy()
        y = d["power_nb"].to_numpy()
        y_lo = d["power_nb_ci_low"].to_numpy()
        y_hi = d["power_nb_ci_high"].to_numpy()

        plt.figure()
        line, = plt.plot(x, y, marker="o", label="NB: completed")
        plt.fill_between(
            x,
            y_lo,
            y_hi,
            alpha=0.2,
            color=line.get_color(),
            label="NB {}% CI".format(int(power_ci_level * 100)),
        )
        plt.axhline(0.80, linestyle="--", label="0.80 target")
        plt.ylim(0, 1.02)
        plt.xlabel("N")
        plt.ylabel("Power")
        plt.title(
            "NB Power curve (term = {})\n"
            "betas: EMP={}, MENT={}, INT={} | rho={} | nsim={}".format(
                term, b_emp, b_ment, b_int, rho_emp_ment, nsim
            )
        )
        plt.legend()
        plt.tight_layout()
        png_path = os.path.join(out_dir, "00_power_nb_curve_{}_{}.png".format(term, stamp))
        plt.savefig(png_path, dpi=180)
        plt.close()

    print("\n=== DONE ===")
    print("Saved CSV: {}".format(csv_path))
    print("Saved TXT: {}".format(txt_path))
    print("Saved PNGs: 00_power_nb_curve_(EMP|MENT|EMPxMENT)_{}.png in {}\n".format(stamp, out_dir))
    print(df_out.pivot_table(index=["N"], columns=["term"], values=["power_nb"]).round(3))
    print("\nMean outcomes by N:")
    print(df_out.groupby("N")[["mean_completed", "mean_mu_true"]].mean().round(3))

    return df_out, csv_path, txt_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))

    run_power_nb(
        N_values=(50, 70, 90, 110, 130, 150),
        nsim=1000,
        alpha=0.05,
        seed=1,
        rho_emp_ment=0.3,
        b_emp=0.4,
        b_ment=0.35,
        b_int=0.20,
        b0_nb=1.0,
        alpha_nb=0.8,
        power_ci_level=0.95,
        out_dir=here,
    )
