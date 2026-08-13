#!/usr/bin/python3
"""Simulation sensitivity analysis for the preregistered primary NB2 model.

This is prospective, scenario-conditional simulated power, not observed post hoc
power. The default run uses 10,000 simulations per scenario at N=100 and writes
timestamped outputs under 8_sensitivity/simulation_sensitivity.

Examples
--------
python 8_sensitivity.py
python 8_sensitivity.py --nsim 100 --seed 20260807
python 8_sensitivity.py --predictor-mode bootstrap
python 8_sensitivity.py --run-n-sensitivity --n-nsim 2000
"""

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "ANALYSIS"
MPL_DIR = HERE / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DEFAULT_INPUT = HERE / "3_purified.csv"
DEFAULT_OUTPUT_DIR = HERE / "8_sensitivity" / "simulation_sensitivity"

OUTCOME = "captcha_post_completions"
LIKE_COL = "q_post_specific_likeability"
MENT_COL = "q_post_specific_mentism"
PRIMARY_FORMULA = "{} ~ like_c + mentism_c + like_x_mentism".format(OUTCOME)
TERMS = ("LIKE", "MENT", "LIKExMENT")
PARAM_NAMES = {
    "LIKE": "like_c",
    "MENT": "mentism_c",
    "LIKExMENT": "like_x_mentism",
}

# The primary script drops complete cases across its full fitted model suite
# before centering the two preregistered predictors. Repeating that preprocessing
# here makes the observed-model validation a genuine like-for-like check.
PRIMARY_SUITE_COLUMNS = (
    OUTCOME,
    LIKE_COL,
    MENT_COL,
    "q_post_gators_pos",
    "q_post_gators_neg",
    "q_pre_idaq",
    "q_pre_captcha_fun",
    "q_pre_captcha_difficulty",
)

EXPECTED_OBSERVED = {
    "Intercept": 2.8607,
    "like_c": 0.2329,
    "mentism_c": 0.0498,
    "like_x_mentism": -0.0698,
    "alpha": 0.7251,
    "logLik": -377.743,
}

ORIGINAL_STANDARDIZED_BETAS = {
    "LIKE": 0.40,
    "MENT": 0.35,
    "LIKExMENT": 0.20,
}
ORIGINAL_ALPHA = 0.80
ORIGINAL_B0 = 1.00
DEFAULT_N_VALUES = (50, 75, 100, 125, 150, 200, 250)


@dataclass(frozen=True)
class Scenario:
    name: str
    beta_like: float
    beta_ment: float
    beta_int: float
    alpha: float
    intercept: float
    predictor_source: str
    recalibrate_each_design: bool
    scale_note: str

    @property
    def betas(self):
        return {
            "LIKE": self.beta_like,
            "MENT": self.beta_ment,
            "LIKExMENT": self.beta_int,
        }


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _fit_nb2(data, maxiter=200):
    """Fit the same discrete NB2 MLE used by ANALYSIS/5_nb_model.py."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.negativebinomial(
            formula=PRIMARY_FORMULA,
            data=data,
        ).fit(disp=False, maxiter=maxiter)


def _prepare_observed_data(path):
    df = pd.read_csv(path)
    missing = [column for column in PRIMARY_SUITE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Observed dataset is missing required columns: {}".format(", ".join(missing)))

    dat = df[list(PRIMARY_SUITE_COLUMNS)].copy()
    for column in PRIMARY_SUITE_COLUMNS:
        dat[column] = pd.to_numeric(dat[column], errors="coerce")
    dat = dat.dropna(subset=list(PRIMARY_SUITE_COLUMNS)).copy()
    if dat.empty:
        raise ValueError("No complete rows remain after primary-analysis preprocessing.")

    dat["like_c"] = dat[LIKE_COL] - dat[LIKE_COL].mean()
    dat["mentism_c"] = dat[MENT_COL] - dat[MENT_COL].mean()
    dat["like_x_mentism"] = dat["like_c"] * dat["mentism_c"]
    return dat


def _validate_observed_fit(dat, tolerance=0.02, ll_tolerance=0.10, allow_warning=False):
    fit = _fit_nb2(dat)
    required = ["Intercept", "like_c", "mentism_c", "like_x_mentism", "alpha"]
    missing = [name for name in required if name not in fit.params.index]
    if missing:
        raise RuntimeError(
            "Observed NB2 fit did not estimate the expected parameters: {}. "
            "This may indicate that a GLM with fixed dispersion was used.".format(", ".join(missing))
        )

    actual = {name: float(fit.params[name]) for name in required}
    actual["logLik"] = float(fit.llf)
    differences = {}
    for name, expected in EXPECTED_OBSERVED.items():
        allowed = ll_tolerance if name == "logLik" else tolerance
        if not np.isfinite(actual[name]) or abs(actual[name] - expected) > allowed:
            differences[name] = {
                "expected": expected,
                "actual": actual[name],
                "tolerance": allowed,
            }

    converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
    if not converged:
        differences["convergence"] = {"expected": True, "actual": False}

    if differences:
        message = (
            "Observed-data NB2 validation failed; refusing to simulate with a silently different model. "
            "Differences: {}".format(json.dumps(differences, sort_keys=True))
        )
        if allow_warning:
            warnings.warn(message)
        else:
            raise RuntimeError(message)
    return fit, actual


def _nb2_rng(mu, alpha, rng):
    """Gamma-Poisson NB2 draws with Var(Y|X) = mu + alpha * mu**2."""
    mu = np.asarray(mu, dtype=float)
    if np.any(~np.isfinite(mu)) or np.any(mu <= 0):
        raise ValueError("All conditional means must be finite and positive.")
    if alpha <= 0:
        draws = rng.poisson(mu)
    else:
        lam = rng.gamma(shape=1.0 / float(alpha), scale=float(alpha) * mu)
        draws = rng.poisson(lam)
    draws = np.asarray(draws)
    if np.any(draws < 0) or not np.issubdtype(draws.dtype, np.integer):
        raise RuntimeError("NB2 generator produced values that were not non-negative integers.")
    return draws.astype(int, copy=False)


def _calibrate_intercept(like, ment, beta_like, beta_ment, beta_int, target_mean):
    interaction = like * ment
    eta_effect = beta_like * like + beta_ment * ment + beta_int * interaction
    # log-sum-exp form avoids unnecessary overflow while preserving the exact formula.
    eta_max = float(np.max(eta_effect))
    log_mean_exp = eta_max + math.log(float(np.mean(np.exp(eta_effect - eta_max))))
    intercept = math.log(float(target_mean)) - log_mean_exp
    achieved = float(np.mean(np.exp(intercept + eta_effect)))
    if not np.isclose(achieved, target_mean, rtol=1e-10, atol=1e-10):
        raise RuntimeError(
            "Intercept calibration failed: target mean={}, achieved mean={}.".format(target_mean, achieved)
        )
    return float(intercept), eta_effect, achieved


def _wilson_ci(k, n, level=0.95):
    if n <= 0:
        return np.nan, np.nan
    p = float(k) / float(n)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _zscore(x):
    x = np.asarray(x, dtype=float)
    sd = float(np.std(x, ddof=0))
    if sd <= 0:
        raise ValueError("Cannot z-score a constant predictor.")
    return (x - float(np.mean(x))) / sd


def _original_predictors(n, rng, rho=0.30):
    covariance = np.array([[1.0, rho], [rho, 1.0]])
    values = rng.multivariate_normal(mean=[0.0, 0.0], cov=covariance, size=n)
    return _zscore(values[:, 0]), _zscore(values[:, 1])


def _observed_predictors(observed_design, n, rng, mode):
    like = observed_design["like_c"].to_numpy(dtype=float)
    ment = observed_design["mentism_c"].to_numpy(dtype=float)
    if mode == "fixed":
        if n != len(observed_design):
            raise ValueError(
                "Fixed predictor mode requires N={} (the observed design); got N={}. "
                "Use bootstrap mode for other sample sizes.".format(len(observed_design), n)
            )
        return like.copy(), ment.copy()
    if mode == "bootstrap":
        indices = rng.integers(0, len(observed_design), size=n)
        return like[indices], ment[indices]
    raise ValueError("Unknown predictor mode: {}".format(mode))


def _build_scenarios(observed_fit, observed_design, target_mean):
    sd_like = float(observed_design["like_c"].std(ddof=1))
    sd_ment = float(observed_design["mentism_c"].std(ddof=1))
    sd_int_scale = sd_like * sd_ment
    observed_alpha = float(observed_fit.params["alpha"])
    observed_betas = {
        "LIKE": float(observed_fit.params["like_c"]),
        "MENT": float(observed_fit.params["mentism_c"]),
        "LIKExMENT": float(observed_fit.params["like_x_mentism"]),
    }

    original_raw = {
        "LIKE": ORIGINAL_STANDARDIZED_BETAS["LIKE"] / sd_like,
        "MENT": ORIGINAL_STANDARDIZED_BETAS["MENT"] / sd_ment,
        "LIKExMENT": ORIGINAL_STANDARDIZED_BETAS["LIKExMENT"] / sd_int_scale,
    }

    def calibrated(name, betas, scale_note):
        intercept, _, _ = _calibrate_intercept(
            observed_design["like_c"].to_numpy(dtype=float),
            observed_design["mentism_c"].to_numpy(dtype=float),
            betas["LIKE"],
            betas["MENT"],
            betas["LIKExMENT"],
            target_mean,
        )
        return Scenario(
            name=name,
            beta_like=betas["LIKE"],
            beta_ment=betas["MENT"],
            beta_int=betas["LIKExMENT"],
            alpha=observed_alpha,
            intercept=intercept,
            predictor_source="observed_paired",
            recalibrate_each_design=True,
            scale_note=scale_note,
        )

    scenarios = [
        Scenario(
            name="original_power_assumptions",
            beta_like=ORIGINAL_STANDARDIZED_BETAS["LIKE"],
            beta_ment=ORIGINAL_STANDARDIZED_BETAS["MENT"],
            beta_int=ORIGINAL_STANDARDIZED_BETAS["LIKExMENT"],
            alpha=ORIGINAL_ALPHA,
            intercept=ORIGINAL_B0,
            predictor_source="idealized_correlated_normal",
            recalibrate_each_design=False,
            scale_note="standardized idealized predictors",
        ),
        calibrated(
            "original_effects_observed_design",
            original_raw,
            "original standardized effects converted to observed raw-centered scale",
        ),
        calibrated("observed_effects", observed_betas, "observed raw-centered scale"),
        calibrated(
            "observed_effects_75pct",
            {term: 0.75 * beta for term, beta in observed_betas.items()},
            "75% of observed raw-centered slopes",
        ),
        calibrated(
            "observed_effects_50pct",
            {term: 0.50 * beta for term, beta in observed_betas.items()},
            "50% of observed raw-centered slopes",
        ),
    ]
    return scenarios


def _simulate_one_design(scenario, observed_design, n, rng, predictor_mode, target_mean):
    if scenario.predictor_source == "idealized_correlated_normal":
        like, ment = _original_predictors(n, rng)
        intercept = scenario.intercept
        eta_effect = (
            scenario.beta_like * like
            + scenario.beta_ment * ment
            + scenario.beta_int * like * ment
        )
    else:
        like, ment = _observed_predictors(observed_design, n, rng, predictor_mode)
        if scenario.recalibrate_each_design:
            intercept, eta_effect, _ = _calibrate_intercept(
                like,
                ment,
                scenario.beta_like,
                scenario.beta_ment,
                scenario.beta_int,
                target_mean,
            )
        else:
            intercept = scenario.intercept
            eta_effect = (
                scenario.beta_like * like
                + scenario.beta_ment * ment
                + scenario.beta_int * like * ment
            )
    interaction = like * ment
    mu = np.exp(intercept + eta_effect)
    y = _nb2_rng(mu, scenario.alpha, rng)
    return pd.DataFrame(
        {
            OUTCOME: y,
            "like_c": like,
            "mentism_c": ment,
            "like_x_mentism": interaction,
        }
    ), mu, intercept


def _empty_term_store():
    return {
        term: {key: [] for key in ("beta", "se", "z", "p", "ci_low", "ci_high")}
        for term in TERMS
    }


def _summarize_scenario(
    scenario,
    term_store,
    alpha_estimates,
    generated_counts,
    generated_variances,
    generated_expected_means,
    successful_fits,
    failed_fits,
    failure_reasons,
    nsim,
    alpha_test,
    n,
):
    pooled_counts = np.concatenate(generated_counts)
    scenario_common = {
        "scenario": scenario.name,
        "N": int(n),
        "simulation_count": int(nsim),
        "successful_fits": int(successful_fits),
        "failed_fits": int(failed_fits),
        "convergence_pct": 100.0 * successful_fits / nsim,
        "failure_pct": 100.0 * failed_fits / nsim,
        "mean_completed": float(np.mean(pooled_counts)),
        "median_completed": float(np.median(pooled_counts)),
        "mean_variance_completed": float(np.mean(generated_variances)),
        "mean_expected_count": float(np.mean(generated_expected_means)),
        "mean_alpha_est": float(np.mean(alpha_estimates)) if alpha_estimates else np.nan,
        "sd_alpha_est": float(np.std(alpha_estimates, ddof=1)) if len(alpha_estimates) > 1 else np.nan,
        "mean_zero_proportion": float(np.mean(pooled_counts == 0)),
        "generating_alpha": float(scenario.alpha),
        "configured_intercept": float(scenario.intercept),
        "predictor_source": scenario.predictor_source,
        "scale_note": scenario.scale_note,
        "failure_reasons": json.dumps(failure_reasons, sort_keys=True),
    }

    rows = []
    for term in TERMS:
        values = term_store[term]
        beta = np.asarray(values["beta"], dtype=float)
        se = np.asarray(values["se"], dtype=float)
        p = np.asarray(values["p"], dtype=float)
        ci_low = np.asarray(values["ci_low"], dtype=float)
        ci_high = np.asarray(values["ci_high"], dtype=float)
        true_beta = float(scenario.betas[term])
        significant = p < alpha_test
        k = int(np.sum(significant))
        power = float(k / successful_fits) if successful_fits else np.nan
        ci_power_low, ci_power_high = _wilson_ci(k, successful_fits)
        if true_beta > 0:
            correct_sign = beta > 0
        elif true_beta < 0:
            correct_sign = beta < 0
        else:
            correct_sign = np.full(beta.shape, np.nan)

        row = dict(scenario_common)
        row.update(
            {
                "term": term,
                "true_beta": true_beta,
                "power": power,
                "power_ci_low": ci_power_low,
                "power_ci_high": ci_power_high,
                "power_all_simulations": float(k / nsim),
                "mean_beta_hat": float(np.mean(beta)) if beta.size else np.nan,
                "median_beta_hat": float(np.median(beta)) if beta.size else np.nan,
                "empirical_beta_sd": float(np.std(beta, ddof=1)) if beta.size > 1 else np.nan,
                "mean_model_se": float(np.mean(se)) if se.size else np.nan,
                "mean_ci_width": float(np.mean(ci_high - ci_low)) if beta.size else np.nan,
                "coverage_95": float(np.mean((ci_low <= true_beta) & (ci_high >= true_beta))) if beta.size else np.nan,
                "correct_sign_rate": float(np.nanmean(correct_sign)) if beta.size else np.nan,
                "significant_fits": k,
            }
        )
        rows.append(row)
    return rows, scenario_common


def run_scenarios(
    scenarios,
    observed_design,
    n=100,
    nsim=10000,
    alpha_test=0.05,
    seed=20260807,
    predictor_mode="fixed",
    target_mean=17.65,
    maxiter=200,
    progress=True,
    collect_details=True,
):
    rng = np.random.default_rng(seed)
    main_rows = []
    scenario_rows = []
    recovery = {}
    detail_rows = []

    for scenario_index, scenario in enumerate(scenarios, start=1):
        term_store = _empty_term_store()
        alpha_estimates = []
        generated_counts = []
        generated_variances = []
        generated_expected_means = []
        failure_reasons = {}
        successful_fits = 0
        failed_fits = 0
        report_every = max(1, nsim // 20)

        if progress:
            print("\n[{}/{}] {} (N={}, nsim={})".format(scenario_index, len(scenarios), scenario.name, n, nsim), flush=True)

        for simulation in range(1, nsim + 1):
            sim_data, mu, generating_intercept = _simulate_one_design(
                scenario,
                observed_design,
                n,
                rng,
                predictor_mode,
                target_mean,
            )
            y = sim_data[OUTCOME].to_numpy(dtype=int)
            generated_counts.append(y)
            generated_variances.append(float(np.var(y, ddof=1)))
            generated_expected_means.append(float(np.mean(mu)))
            detail = {
                "scenario": scenario.name,
                "simulation": int(simulation),
                "N": int(n),
                "fit_success": False,
                "converged": False,
                "failure_reason": "",
                "generated_mean": float(np.mean(y)),
                "generated_median": float(np.median(y)),
                "generated_variance": float(np.var(y, ddof=1)),
                "generated_zero_proportion": float(np.mean(y == 0)),
                "expected_mean": float(np.mean(mu)),
                "generating_intercept": float(generating_intercept),
                "generating_alpha": float(scenario.alpha),
                "estimated_alpha": np.nan,
            }

            try:
                fit = _fit_nb2(sim_data, maxiter=maxiter)
                converged = bool(getattr(fit, "mle_retvals", {}).get("converged", True))
                required_names = [PARAM_NAMES[term] for term in TERMS] + ["alpha"]
                missing = [name for name in required_names if name not in fit.params.index]
                if missing:
                    raise RuntimeError("missing fitted parameters: {}".format(", ".join(missing)))
                if not converged:
                    raise RuntimeError("optimizer did not converge")
                ci = fit.conf_int(alpha=0.05)
                extracted = {}
                for term in TERMS:
                    name = PARAM_NAMES[term]
                    beta_hat = float(fit.params[name])
                    model_se = float(fit.bse[name])
                    p_value = float(fit.pvalues[name])
                    ci_low = float(ci.loc[name, 0])
                    ci_high = float(ci.loc[name, 1])
                    values = np.asarray([beta_hat, model_se, p_value, ci_low, ci_high])
                    if np.any(~np.isfinite(values)) or model_se <= 0:
                        raise RuntimeError("non-finite coefficient result for {}".format(name))
                    extracted[term] = (beta_hat, model_se, beta_hat / model_se, p_value, ci_low, ci_high)
                alpha_hat = float(fit.params["alpha"])
                if not np.isfinite(alpha_hat) or alpha_hat <= 0:
                    raise RuntimeError("invalid estimated alpha")

                successful_fits += 1
                alpha_estimates.append(alpha_hat)
                detail["fit_success"] = True
                detail["converged"] = True
                detail["estimated_alpha"] = alpha_hat
                for term, result in extracted.items():
                    for key, value in zip(("beta", "se", "z", "p", "ci_low", "ci_high"), result):
                        term_store[term][key].append(value)
                        detail["{}_{}".format(term, key)] = value
            except Exception as exc:
                failed_fits += 1
                reason = "{}: {}".format(type(exc).__name__, str(exc))
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                detail["failure_reason"] = reason

            if collect_details:
                detail_rows.append(detail)

            if progress and (simulation % report_every == 0 or simulation == nsim):
                print("  {:>6}/{} simulations; successful fits={}; failed={}".format(simulation, nsim, successful_fits, failed_fits), flush=True)

        rows, scenario_summary = _summarize_scenario(
            scenario,
            term_store,
            alpha_estimates,
            generated_counts,
            generated_variances,
            generated_expected_means,
            successful_fits,
            failed_fits,
            failure_reasons,
            nsim,
            alpha_test,
            n,
        )
        main_rows.extend(rows)
        scenario_rows.append(scenario_summary)
        recovery[scenario.name] = {
            term: np.asarray(term_store[term]["beta"], dtype=float) for term in TERMS
        }

    return pd.DataFrame(main_rows), pd.DataFrame(scenario_rows), recovery, pd.DataFrame(detail_rows)


def _standardization_scale(term, scenario, sd_like, sd_ment):
    if scenario.predictor_source == "idealized_correlated_normal":
        return 1.0
    if term == "LIKE":
        return sd_like
    if term == "MENT":
        return sd_ment
    return sd_like * sd_ment


def _plot_power(main_table, scenario_order, out_path, nsim, n):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenario_order), dtype=float)
    offsets = {"LIKE": -0.20, "MENT": 0.0, "LIKExMENT": 0.20}
    colors = {"LIKE": "#2f5d8a", "MENT": "#d97924", "LIKExMENT": "#3a8b4b"}
    for term in TERMS:
        d = main_table.set_index(["scenario", "term"])
        y = np.asarray([d.loc[(scenario, term), "power"] for scenario in scenario_order], dtype=float)
        low = np.asarray([d.loc[(scenario, term), "power_ci_low"] for scenario in scenario_order], dtype=float)
        high = np.asarray([d.loc[(scenario, term), "power_ci_high"] for scenario in scenario_order], dtype=float)
        ax.errorbar(
            x + offsets[term],
            y,
            yerr=np.vstack([y - low, high - y]),
            fmt="o-",
            capsize=3,
            linewidth=1.4,
            markersize=6,
            color=colors[term],
            label=term,
        )
    ax.axhline(0.80, color="#555555", linestyle="--", linewidth=1.0, label="80% target")
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_order, rotation=22, ha="right")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Simulated power (p < .05)")
    ax.set_title(
        "Primary NB2 model: scenario-conditional power at N={}\n"
        "{} simulations per scenario; Wilson 95% CIs".format(n, nsim)
    )
    ax.legend(ncol=4, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_coefficient_recovery(recovery, scenarios, sd_like, sd_ment, out_path):
    scenario_order = [scenario.name for scenario in scenarios]
    scenario_map = {scenario.name: scenario for scenario in scenarios}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=False)
    for ax, term in zip(axes, TERMS):
        distributions = []
        true_values = []
        means = []
        for scenario_name in scenario_order:
            scenario = scenario_map[scenario_name]
            scale = _standardization_scale(term, scenario, sd_like, sd_ment)
            values = recovery[scenario_name][term] * scale
            distributions.append(values)
            true_values.append(scenario.betas[term] * scale)
            means.append(float(np.mean(values)) if values.size else np.nan)
        ax.boxplot(distributions, positions=np.arange(len(scenario_order)), widths=0.55, showfliers=False)
        ax.scatter(np.arange(len(scenario_order)), true_values, color="#b24a2a", marker="x", s=60, linewidths=2, label="True beta")
        ax.scatter(np.arange(len(scenario_order)), means, color="#1f1f1f", marker="D", s=30, label="Mean estimate")
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(np.arange(len(scenario_order)))
        ax.set_xticklabels(scenario_order, rotation=35, ha="right", fontsize=8)
        ax.set_title(term)
        if term == "LIKE":
            ax.set_ylabel("Standardized-equivalent coefficient")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("NB2 coefficient recovery by generating scenario")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_n_sensitivity(n_table, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    colors = {
        "observed_effects": "#2f5d8a",
        "observed_effects_75pct": "#d97924",
        "observed_effects_50pct": "#3a8b4b",
    }
    for ax, term in zip(axes, TERMS):
        for scenario, d in n_table[n_table["term"] == term].groupby("scenario"):
            d = d.sort_values("N")
            ax.plot(d["N"], d["power"], marker="o", color=colors.get(scenario), label=scenario)
            ax.fill_between(d["N"], d["power_ci_low"], d["power_ci_high"], color=colors.get(scenario), alpha=0.12)
        ax.axhline(0.80, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("N")
        ax.set_title(term)
    axes[0].set_ylabel("Simulated power")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Optional sample-size sensitivity (paired predictor bootstrap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_summary(
    path,
    args,
    observed_path,
    observed_dat,
    observed_fit,
    observed_actual,
    scenarios,
    main_table,
    scenario_table,
    output_paths,
):
    sd_like = float(observed_dat["like_c"].std(ddof=1))
    sd_ment = float(observed_dat["mentism_c"].std(ddof=1))
    observed_standardized = {
        "LIKE": float(observed_fit.params["like_c"] * sd_like),
        "MENT": float(observed_fit.params["mentism_c"] * sd_ment),
        "LIKExMENT": float(observed_fit.params["like_x_mentism"] * sd_like * sd_ment),
    }
    observed_mean = float(observed_dat[OUTCOME].mean())
    observed_median = float(observed_dat[OUTCOME].median())
    original_outcome = scenario_table.set_index("scenario").loc["original_power_assumptions"]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("SIMULATION-BASED SENSITIVITY ANALYSIS: PRIMARY NB2 MODEL\n")
        handle.write("This is scenario-conditional simulated power, not observed post hoc or achieved power.\n\n")
        handle.write("CONFIGURATION\n")
        handle.write("Observed data: {}\n".format(observed_path))
        handle.write("Model: {}\n".format(PRIMARY_FORMULA))
        handle.write("Fitting method: statsmodels discrete NegativeBinomial NB2 MLE; alpha estimated in every fit\n")
        handle.write("N: {}\nnsim per scenario: {}\nalpha_test: {}\nseed: {}\n".format(args.N, args.nsim, args.alpha_test, args.seed))
        handle.write("Observed-design predictor mode: {}\ntarget mean: {}\n\n".format(args.predictor_mode, args.target_mean))

        handle.write("OBSERVED-DATA VALIDATION\n")
        handle.write("N complete cases: {}\n".format(len(observed_dat)))
        for name in ("Intercept", "like_c", "mentism_c", "like_x_mentism", "alpha", "logLik"):
            handle.write("{}: {:.6f}\n".format(name, observed_actual[name]))
        handle.write("Observed predictor SDs (sample): LIKE={:.7f}, MENT={:.7f}\n".format(sd_like, sd_ment))
        handle.write("Observed completion mean={:.4f}, median={:.4f}, variance={:.4f}\n\n".format(observed_mean, observed_median, observed_dat[OUTCOME].var(ddof=1)))

        handle.write("ORIGINAL VS OBSERVED\n")
        handle.write("Original standardised betas: LIKE=.40, MENT=.35, INT=.20\n")
        handle.write(
            "Observed standardised-equivalent betas: LIKE={:+.3f}, MENT={:+.3f}, INT={:+.3f}\n".format(
                observed_standardized["LIKE"], observed_standardized["MENT"], observed_standardized["LIKExMENT"]
            )
        )
        handle.write("Original alpha=.80\nObserved alpha={:.6f}\n".format(observed_actual["alpha"]))
        handle.write(
            "Original simulated mean/median: {:.3f}/{:.3f}\nObserved mean/median: {:.3f}/{:.3f}\n\n".format(
                original_outcome["mean_completed"], original_outcome["median_completed"], observed_mean, observed_median
            )
        )

        handle.write("GENERATING SCENARIOS\n")
        for scenario in scenarios:
            handle.write(
                "{}: b0={:.6f}, LIKE={:+.6f}, MENT={:+.6f}, INT={:+.6f}, alpha={:.6f}; {}; {}\n".format(
                    scenario.name,
                    scenario.intercept,
                    scenario.beta_like,
                    scenario.beta_ment,
                    scenario.beta_int,
                    scenario.alpha,
                    scenario.predictor_source,
                    scenario.scale_note,
                )
            )

        handle.write("\nPOWER (conditional on successful fits; Wilson 95% CI)\n")
        power_view = main_table.pivot(index="scenario", columns="term", values="power").reindex([s.name for s in scenarios])
        handle.write(power_view.to_string(float_format=lambda value: "{:.3f}".format(value)))
        handle.write("\n\nCOEFFICIENT BIAS (mean estimate - true beta)\n")
        bias_table = main_table.assign(bias=main_table["mean_beta_hat"] - main_table["true_beta"]).pivot(index="scenario", columns="term", values="bias")
        handle.write(bias_table.reindex([s.name for s in scenarios]).to_string(float_format=lambda value: "{:+.4f}".format(value)))
        handle.write("\n\n95% CI COVERAGE\n")
        coverage = main_table.pivot(index="scenario", columns="term", values="coverage_95").reindex([s.name for s in scenarios])
        handle.write(coverage.to_string(float_format=lambda value: "{:.3f}".format(value)))
        handle.write("\n\nCONVERGENCE / FAILURES\n")
        handle.write(
            scenario_table[
                [
                    "scenario",
                    "successful_fits",
                    "failed_fits",
                    "convergence_pct",
                    "failure_pct",
                    "failure_reasons",
                ]
            ].to_string(index=False)
        )
        handle.write("\n\nSIMULATED COUNT DISTRIBUTIONS AND DISPERSION RECOVERY\n")
        handle.write(
            scenario_table[
                [
                    "scenario",
                    "mean_completed",
                    "median_completed",
                    "mean_variance_completed",
                    "mean_zero_proportion",
                    "generating_alpha",
                    "mean_alpha_est",
                    "sd_alpha_est",
                ]
            ].to_string(index=False, float_format=lambda value: "{:.4f}".format(value))
        )
        handle.write("\n\nOUTPUT FILES\n")
        for label, output_path in output_paths.items():
            handle.write("{}: {}\n".format(label, output_path))


def _run_optional_n_sensitivity(args, scenarios, observed_design, output_dir, stamp):
    selected_names = {"observed_effects", "observed_effects_75pct", "observed_effects_50pct"}
    selected = [scenario for scenario in scenarios if scenario.name in selected_names]
    tables = []
    for index, n in enumerate(args.N_values):
        table, _, _, _ = run_scenarios(
            selected,
            observed_design,
            n=n,
            nsim=args.n_nsim,
            alpha_test=args.alpha_test,
            seed=args.seed + 100000 + index,
            predictor_mode="bootstrap",
            target_mean=args.target_mean,
            maxiter=args.maxiter,
            progress=not args.quiet,
            collect_details=False,
        )
        tables.append(table)
    n_table = pd.concat(tables, ignore_index=True)
    csv_path = output_dir / "8_nb2_n_sensitivity_{}.csv".format(stamp)
    plot_path = output_dir / "8_nb2_n_sensitivity_power_curves_{}.png".format(stamp)
    n_table.to_csv(csv_path, index=False)
    _plot_n_sensitivity(n_table, plot_path)
    return csv_path, plot_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--N", type=int, default=100, help="Main sensitivity sample size (default: 100).")
    parser.add_argument("--nsim", type=int, default=10000, help="Simulations per main scenario (default: 10000).")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--target-mean", type=float, default=17.65)
    parser.add_argument("--alpha-test", type=float, default=0.05)
    parser.add_argument("--predictor-mode", choices=("fixed", "bootstrap"), default="fixed")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--validation-tolerance", type=float, default=0.02)
    parser.add_argument("--ll-tolerance", type=float, default=0.10)
    parser.add_argument("--allow-validation-warning", action="store_true")
    parser.add_argument("--run-n-sensitivity", action="store_true")
    parser.add_argument("--N-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--n-nsim", type=int, default=2000, help="Simulations per scenario/N in optional N sensitivity.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.N <= 0 or args.nsim <= 0 or args.n_nsim <= 0:
        parser.error("N and simulation counts must be positive.")
    if args.target_mean <= 0:
        parser.error("--target-mean must be positive.")
    if not 0 < args.alpha_test < 1:
        parser.error("--alpha-test must be between 0 and 1.")
    return args


def main(argv=None):
    args = parse_args(argv)
    observed_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observed_dat = _prepare_observed_data(observed_path)
    if args.N == 100 and len(observed_dat) != 100:
        raise RuntimeError(
            "The requested N=100 fixed-design analysis expected 100 observed complete cases, but found {}.".format(len(observed_dat))
        )
    if args.predictor_mode == "fixed" and args.N != len(observed_dat):
        raise ValueError("--predictor-mode fixed requires --N {}.".format(len(observed_dat)))

    observed_fit, observed_actual = _validate_observed_fit(
        observed_dat,
        tolerance=args.validation_tolerance,
        ll_tolerance=args.ll_tolerance,
        allow_warning=args.allow_validation_warning,
    )
    scenarios = _build_scenarios(observed_fit, observed_dat, args.target_mean)

    # Explicit pre-simulation calibration validation on the fixed observed matrix.
    for scenario in scenarios:
        if scenario.predictor_source == "observed_paired":
            _, _, achieved = _calibrate_intercept(
                observed_dat["like_c"].to_numpy(dtype=float),
                observed_dat["mentism_c"].to_numpy(dtype=float),
                scenario.beta_like,
                scenario.beta_ment,
                scenario.beta_int,
                args.target_mean,
            )
            if not np.isclose(achieved, args.target_mean, atol=1e-10):
                raise RuntimeError("Scenario {} failed expected-mean validation.".format(scenario.name))

    print("Observed NB2 validation passed: N={}, alpha={:.6f}, logLik={:.6f}".format(len(observed_dat), observed_actual["alpha"], observed_actual["logLik"]))
    print("Fitted simulations estimate alpha via discrete NB2 MLE; alpha is not supplied to the fitted model.")

    main_table, scenario_table, recovery, simulation_details = run_scenarios(
        scenarios,
        observed_dat,
        n=args.N,
        nsim=args.nsim,
        alpha_test=args.alpha_test,
        seed=args.seed,
        predictor_mode=args.predictor_mode,
        target_mean=args.target_mean,
        maxiter=args.maxiter,
        progress=not args.quiet,
    )

    stamp = _timestamp()
    main_csv = output_dir / "8_nb2_sensitivity_scenario_by_term_{}.csv".format(stamp)
    scenario_csv = output_dir / "8_nb2_sensitivity_scenario_outcomes_{}.csv".format(stamp)
    details_csv = output_dir / "8_nb2_sensitivity_simulation_details_{}.csv".format(stamp)
    power_plot = output_dir / "8_nb2_sensitivity_power_{}.png".format(stamp)
    recovery_plot = output_dir / "8_nb2_sensitivity_coefficient_recovery_{}.png".format(stamp)
    summary_txt = output_dir / "8_nb2_sensitivity_summary_{}.txt".format(stamp)

    main_table.to_csv(main_csv, index=False)
    scenario_table.to_csv(scenario_csv, index=False)
    simulation_details.to_csv(details_csv, index=False)
    scenario_order = [scenario.name for scenario in scenarios]
    _plot_power(main_table, scenario_order, power_plot, args.nsim, args.N)
    _plot_coefficient_recovery(
        recovery,
        scenarios,
        float(observed_dat["like_c"].std(ddof=1)),
        float(observed_dat["mentism_c"].std(ddof=1)),
        recovery_plot,
    )

    output_paths = {
        "scenario_by_term_csv": main_csv,
        "scenario_outcomes_csv": scenario_csv,
        "simulation_details_csv": details_csv,
        "power_figure": power_plot,
        "coefficient_recovery_figure": recovery_plot,
        "text_summary": summary_txt,
    }
    if args.run_n_sensitivity:
        n_csv, n_plot = _run_optional_n_sensitivity(args, scenarios, observed_dat, output_dir, stamp)
        output_paths["optional_n_sensitivity_csv"] = n_csv
        output_paths["optional_n_sensitivity_figure"] = n_plot

    _write_summary(
        summary_txt,
        args,
        observed_path,
        observed_dat,
        observed_fit,
        observed_actual,
        scenarios,
        main_table,
        scenario_table,
        output_paths,
    )

    print("\nSimulation sensitivity analysis complete.")
    print(main_table.pivot(index="scenario", columns="term", values="power").reindex(scenario_order).round(3))
    for label, path in output_paths.items():
        print("{}: {}".format(label, path))
    return main_table, scenario_table, output_paths


if __name__ == "__main__":
    main()
