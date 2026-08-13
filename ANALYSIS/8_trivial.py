"""Exploratory analysis of perceived reCAPTCHA difficulty and persistence.

This is deliberately separate from the preregistered primary-analysis script.
It fits the same maximum-likelihood NB2 model used in ``5_nb_model.py`` and
writes stable, non-timestamped exploratory outputs to ``ANALYSIS/8_trivial``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "ANALYSIS"
DATA_PATH = ANALYSIS_DIR / "3_purified.csv"
OUTPUT_DIR = ANALYSIS_DIR / "8_trivial"
MPL_DIR = ANALYSIS_DIR / ".mplconfig"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2


OUTCOME = "captcha_post_completions"
LIKE = "q_post_specific_likeability"
MENTISM = "q_post_specific_mentism"
DIFFICULTY = "q_pre_captcha_difficulty"
PRIMARY_TERMS = "like_c + mentism_c + like_x_mentism"

FORMULAS = {
    "primary": f"{OUTCOME} ~ {PRIMARY_TERMS}",
    "primary_plus_linear_difficulty": f"{OUTCOME} ~ {PRIMARY_TERMS} + difficulty_c",
    "primary_plus_high_difficulty": f"{OUTCOME} ~ {PRIMARY_TERMS} + high_difficulty",
    "primary_plus_difficulty_hinge": f"{OUTCOME} ~ {PRIMARY_TERMS} + difficulty_hinge",
    "supplementary_quadratic": (
        f"{OUTCOME} ~ {PRIMARY_TERMS} + difficulty_c + difficulty_c_sq"
    ),
}


def fit_nb(formula: str, data: pd.DataFrame):
    """Match the primary script: discrete NB2 MLE with estimated dispersion."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.negativebinomial(formula=formula, data=data).fit(
            disp=False, maxiter=200
        )


def lr_test(smaller, larger) -> tuple[float, int, float]:
    statistic = 2.0 * (larger.llf - smaller.llf)
    df_difference = int(larger.df_model - smaller.df_model)
    # Small negative values can arise only from numerical optimization noise.
    p_value = float(chi2.sf(max(statistic, 0.0), df_difference))
    return float(statistic), df_difference, p_value


def load_data() -> pd.DataFrame:
    required = [OUTCOME, LIKE, MENTISM, DIFFICULTY]
    source = pd.read_csv(DATA_PATH)
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data = source[required].copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required).copy()
    if data.empty:
        raise ValueError("No complete cases remain for the requested analysis.")
    if (data[OUTCOME] < 0).any() or not np.allclose(data[OUTCOME] % 1, 0):
        raise ValueError(f"{OUTCOME} must contain non-negative integer counts.")
    observed = sorted(data[DIFFICULTY].unique())
    if any(value < 1 or value > 7 or value % 1 for value in observed):
        raise ValueError(f"Unexpected difficulty ratings: {observed}")

    data["like_c"] = data[LIKE] - data[LIKE].mean()
    data["mentism_c"] = data[MENTISM] - data[MENTISM].mean()
    data["like_x_mentism"] = data["like_c"] * data["mentism_c"]
    data["difficulty_c"] = data[DIFFICULTY] - data[DIFFICULTY].mean()
    data["high_difficulty"] = (data[DIFFICULTY] >= 5).astype(int)
    data["difficulty_hinge"] = np.maximum(data[DIFFICULTY] - 4, 0)
    data["difficulty_c_sq"] = data["difficulty_c"] ** 2
    return data


def verify_primary_reproduction(primary_fit) -> dict[str, object]:
    """Compare the refit with the newest saved primary-model artifacts."""
    comparison_files = sorted(
        (ANALYSIS_DIR / "5_nb_models").glob("5_nb_model_primary_comparison_*.csv")
    )
    coefficient_files = sorted(
        (ANALYSIS_DIR / "5_nb_models").glob(
            "5_nb_model_primary_irr_theory_primary_only_*.csv"
        )
    )
    if not comparison_files:
        return {
            "status": "not_checked",
            "detail": "No saved primary comparison artifact was found.",
        }

    comparison_path = comparison_files[-1]
    saved = pd.read_csv(comparison_path)
    saved = saved.loc[saved["model"] == "theory_primary_only"]
    if len(saved) != 1:
        return {
            "status": "not_checked",
            "detail": f"No unique primary row in {comparison_path.name}.",
        }

    saved_row = saved.iloc[0]
    checks = {
        "logLik": (float(primary_fit.llf), float(saved_row["logLik"])),
        "AIC": (float(primary_fit.aic), float(saved_row["AIC"])),
        "BIC": (float(primary_fit.bic), float(saved_row["BIC"])),
        "alpha": (float(primary_fit.params["alpha"]), float(saved_row["alpha_est"])),
    }
    coefficient_source = None
    if coefficient_files:
        coefficient_source = coefficient_files[-1]
        saved_coef = pd.read_csv(coefficient_source).set_index("term")
        for term in ["Intercept", "like_c", "mentism_c", "like_x_mentism"]:
            if term in saved_coef.index:
                checks[f"coef:{term}"] = (
                    float(primary_fit.params[term]),
                    float(saved_coef.loc[term, "coef"]),
                )

    differences = {key: abs(current - previous) for key, (current, previous) in checks.items()}
    max_difference = max(differences.values())
    tolerance = 1e-5
    status = "passed" if max_difference <= tolerance else "failed"
    sources = comparison_path.name
    if coefficient_source is not None:
        sources += f" and {coefficient_source.name}"
    return {
        "status": status,
        "detail": (
            f"Compared with {sources}; maximum absolute difference "
            f"was {max_difference:.3g} (tolerance {tolerance:g})."
        ),
    }


def coefficient_table(fits: dict[str, object]) -> pd.DataFrame:
    rows = []
    for model, fit in fits.items():
        confidence = fit.conf_int()
        for term in fit.params.index:
            if term == "alpha":
                continue
            coefficient = float(fit.params[term])
            rows.append(
                {
                    "model": model,
                    "term": term,
                    "coefficient": coefficient,
                    "SE": float(fit.bse[term]),
                    "z": float(fit.tvalues[term]),
                    "p_value": float(fit.pvalues[term]),
                    "IRR": float(np.exp(coefficient)),
                    "CI_low_coefficient": float(confidence.loc[term, 0]),
                    "CI_high_coefficient": float(confidence.loc[term, 1]),
                    "CI_low_IRR": float(np.exp(confidence.loc[term, 0])),
                    "CI_high_IRR": float(np.exp(confidence.loc[term, 1])),
                }
            )
    return pd.DataFrame(rows)


def model_comparison(fits: dict[str, object]) -> pd.DataFrame:
    primary = fits["primary"]
    rows = []
    for model in [
        "primary",
        "primary_plus_linear_difficulty",
        "primary_plus_high_difficulty",
        "primary_plus_difficulty_hinge",
    ]:
        fit = fits[model]
        if model == "primary":
            lr, df_difference, p_value = np.nan, 0, np.nan
        else:
            lr, df_difference, p_value = lr_test(primary, fit)
        rows.append(
            {
                "model": model,
                "n": int(fit.nobs),
                "df_model": int(fit.df_model),
                "logLik": float(fit.llf),
                "AIC": float(fit.aic),
                "BIC": float(fit.bic),
                "alpha_est": float(fit.params["alpha"]),
                "LR_chi2_vs_primary": lr,
                "df_difference": df_difference,
                "p_LR_vs_primary": p_value,
            }
        )
    return pd.DataFrame(rows)


def summarize_group(values: pd.DataFrame) -> dict[str, float]:
    completions = values[OUTCOME]
    return {
        "n": int(len(values)),
        "mean_completions": float(completions.mean()),
        "median_completions": float(completions.median()),
        "sd_completions": float(completions.std(ddof=1)),
        "min_completions": float(completions.min()),
        "max_completions": float(completions.max()),
        "proportion_gt_10": float((completions > 10).mean()),
        "proportion_gt_20": float((completions > 20).mean()),
    }


def descriptive_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rating in range(1, 7):
        subset = data.loc[data[DIFFICULTY] == rating]
        row = {
            "grouping": "individual_rating",
            "difficulty_group": str(rating),
            "difficulty": rating,
        }
        if subset.empty:
            row.update({key: np.nan for key in summarize_group(data).keys()})
            row["n"] = 0
        else:
            row.update(summarize_group(subset))
        rows.append(row)

    broad_groups = {
        "1-3": data[DIFFICULTY].between(1, 3),
        "4": data[DIFFICULTY] == 4,
        "5-6": data[DIFFICULTY].between(5, 6),
    }
    for label, mask in broad_groups.items():
        row = {
            "grouping": "broad_group",
            "difficulty_group": label,
            "difficulty": np.nan,
        }
        row.update(summarize_group(data.loc[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_table(data: pd.DataFrame, fits: dict[str, object]) -> pd.DataFrame:
    difficulty_mean = float(data[DIFFICULTY].mean())
    ratings = pd.DataFrame({"difficulty": np.arange(1, 7, dtype=float)})
    ratings["like_c"] = 0.0
    ratings["mentism_c"] = 0.0
    ratings["like_x_mentism"] = 0.0
    ratings["difficulty_c"] = ratings["difficulty"] - difficulty_mean
    ratings["high_difficulty"] = (ratings["difficulty"] >= 5).astype(int)
    ratings["difficulty_hinge"] = np.maximum(ratings["difficulty"] - 4, 0)
    ratings["difficulty_c_sq"] = ratings["difficulty_c"] ** 2

    rows = []
    for model in [
        "primary_plus_linear_difficulty",
        "primary_plus_high_difficulty",
        "primary_plus_difficulty_hinge",
        "supplementary_quadratic",
    ]:
        with warnings.catch_warnings():
            # statsmodels currently emits an informational warning while using
            # the model's (correct) log link for discrete-NB prediction.
            warnings.filterwarnings(
                "ignore", message="using default log-link in get_prediction"
            )
            prediction = fits[model].get_prediction(ratings).summary_frame(alpha=0.05)
        for index, rating in enumerate(ratings["difficulty"].astype(int)):
            rows.append(
                {
                    "model": model,
                    "difficulty": rating,
                    "difficulty_group": "5-6" if rating >= 5 else "1-4",
                    "likeability_held_at": float(data[LIKE].mean()),
                    "mentism_held_at": float(data[MENTISM].mean()),
                    "predicted_completions": float(prediction.iloc[index]["predicted"]),
                    "CI_low": float(prediction.iloc[index]["ci_lower"]),
                    "CI_high": float(prediction.iloc[index]["ci_upper"]),
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    data: pd.DataFrame, predictions: pd.DataFrame, output_path: Path, log_scale: bool = False
) -> None:
    rng = np.random.default_rng(20260809)
    jittered = data[DIFFICULTY].to_numpy() + rng.uniform(-0.13, 0.13, len(data))
    hinge = predictions.loc[
        predictions["model"] == "primary_plus_difficulty_hinge"
    ].sort_values("difficulty")

    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    ax.scatter(
        jittered,
        data[OUTCOME],
        s=34,
        alpha=0.52,
        color="#356b8c",
        edgecolors="white",
        linewidths=0.35,
        label="Observed completions",
    )
    ax.plot(
        hinge["difficulty"],
        hinge["predicted_completions"],
        color="#b2472f",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="NB2 hinge-model expectation",
    )
    ax.fill_between(
        hinge["difficulty"].to_numpy(),
        hinge["CI_low"].to_numpy(),
        hinge["CI_high"].to_numpy(),
        color="#b2472f",
        alpha=0.16,
        linewidth=0,
        label="95% confidence interval",
    )
    ax.axvline(4.5, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(range(1, 7))
    ax.set_xlim(0.65, 6.35)
    ax.set_xlabel("Perceived reCAPTCHA difficulty rating")
    ax.set_ylabel("Voluntary reCAPTCHA completions")
    ax.set_title("Perceived difficulty and voluntary persistence")
    if log_scale:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("Voluntary reCAPTCHA completions (symlog scale)")
    else:
        ax.set_ylim(bottom=-2)
    ax.grid(axis="y", color="#d5d5d5", linewidth=0.65, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt_p(value: float) -> str:
    return "p < .001" if value < 0.001 else f"p = {value:.3f}".replace("0.", ".")


def effect_sentence(coefficients: pd.DataFrame, model: str, term: str) -> str:
    row = coefficients.loc[
        (coefficients["model"] == model) & (coefficients["term"] == term)
    ].iloc[0]
    return (
        f"b = {row.coefficient:.3f}, SE = {row.SE:.3f}, z = {row.z:.2f}, "
        f"{fmt_p(row.p_value)}, IRR = {row.IRR:.3f}, 95% CI "
        f"[{row.CI_low_IRR:.3f}, {row.CI_high_IRR:.3f}]"
    )


def write_summary(
    data: pd.DataFrame,
    fits: dict[str, object],
    coefficients: pd.DataFrame,
    comparison: pd.DataFrame,
    descriptives: pd.DataFrame,
    predictions: pd.DataFrame,
    verification: dict[str, object],
) -> None:
    difficulty = data[DIFFICULTY]
    low_lt4 = data.loc[difficulty < 4, OUTCOME]
    high_gt4 = data.loc[difficulty > 4, OUTCOME]
    high = data.loc[difficulty >= 5, OUTCOME]
    rating_four = data.loc[difficulty == 4, OUTCOME]

    threshold_prediction = predictions.loc[
        predictions["model"] == "primary_plus_high_difficulty"
    ]
    predicted_low = threshold_prediction.loc[
        threshold_prediction["difficulty"] == 1, "predicted_completions"
    ].iloc[0]
    predicted_high = threshold_prediction.loc[
        threshold_prediction["difficulty"] == 5, "predicted_completions"
    ].iloc[0]

    linear_fit = fits["primary_plus_linear_difficulty"]
    quadratic_fit = fits["supplementary_quadratic"]
    quadratic_lr, quadratic_df, quadratic_p = lr_test(linear_fit, quadratic_fit)
    hinge_aic = fits["primary_plus_difficulty_hinge"].aic
    linear_aic = linear_fit.aic
    hinge_bic = fits["primary_plus_difficulty_hinge"].bic
    linear_bic = linear_fit.bic
    best_expanded = comparison.loc[comparison["model"] != "primary"].sort_values("AIC").iloc[0]

    text = f"""Exploratory analysis of perceived reCAPTCHA difficulty
=======================================================

Data: {DATA_PATH}
Complete cases used in every model: n = {len(data)}
Difficulty variable: {DIFFICULTY}
Difficulty: M = {difficulty.mean():.2f}, SD = {difficulty.std(ddof=1):.2f}, range = {difficulty.min():.0f}-{difficulty.max():.0f}
Rating counts: {difficulty.value_counts().sort_index().astype(int).to_dict()}
No participant selected rating 7.

Primary-model reproduction check: {verification['status'].upper()}
{verification['detail']}
Refitted primary: logLik = {fits['primary'].llf:.6f}, AIC = {fits['primary'].aic:.3f}, BIC = {fits['primary'].bic:.3f}, alpha = {fits['primary'].params['alpha']:.6f}.

Observed descriptive checks
---------------------------
Difficulty < 4: n = {len(low_lt4)}, mean completions = {low_lt4.mean():.2f}.
Difficulty = 4: n = {len(rating_four)}, mean completions = {rating_four.mean():.2f}.
Difficulty > 4 (ratings 5-6): n = {len(high_gt4)}, mean completions = {high_gt4.mean():.2f}.
Among ratings 5-6, {int((high > 10).sum())} participants completed >10, {int((high > 20).sum())} completed >20, and the maximum was {high.max():.0f}.

1. Is there evidence for a simple linear difficulty association?
----------------------------------------------------------------
{effect_sentence(coefficients, 'primary_plus_linear_difficulty', 'difficulty_c')}.
The LR comparison with the primary model was chi-square(1) = {comparison.loc[comparison.model == 'primary_plus_linear_difficulty', 'LR_chi2_vs_primary'].iloc[0]:.3f}, {fmt_p(comparison.loc[comparison.model == 'primary_plus_linear_difficulty', 'p_LR_vs_primary'].iloc[0])}.

2. Is there evidence specifically that ratings 5-6 suppress persistence?
------------------------------------------------------------------------
{effect_sentence(coefficients, 'primary_plus_high_difficulty', 'high_difficulty')}.
At mean likeability and mentism, predicted completions were {predicted_low:.2f} for ratings 1-4 and {predicted_high:.2f} for ratings 5-6.
The LR comparison with the primary model was chi-square(1) = {comparison.loc[comparison.model == 'primary_plus_high_difficulty', 'LR_chi2_vs_primary'].iloc[0]:.3f}, {fmt_p(comparison.loc[comparison.model == 'primary_plus_high_difficulty', 'p_LR_vs_primary'].iloc[0])}.

3. Does a hinge model fit better than the ordinary linear model?
----------------------------------------------------------------
{effect_sentence(coefficients, 'primary_plus_difficulty_hinge', 'difficulty_hinge')} per additional difficulty point above 4.
The LR comparison with the primary model was chi-square(1) = {comparison.loc[comparison.model == 'primary_plus_difficulty_hinge', 'LR_chi2_vs_primary'].iloc[0]:.3f}, {fmt_p(comparison.loc[comparison.model == 'primary_plus_difficulty_hinge', 'p_LR_vs_primary'].iloc[0])}.
The hinge model had AIC = {hinge_aic:.3f} and BIC = {hinge_bic:.3f}; the linear model had AIC = {linear_aic:.3f} and BIC = {linear_bic:.3f}. These non-nested models are compared descriptively by information criteria, not by an LR test.

4. Does adding difficulty materially improve the primary liking x robomentism model?
-------------------------------------------------------------------------------------
The lowest-AIC expanded model was {best_expanded.model} (AIC = {best_expanded.AIC:.3f}) compared with primary AIC = {fits['primary'].aic:.3f}. See the model-comparison CSV for all LR tests, AIC, BIC, and dispersion estimates.

5. How much substantial persistence nevertheless occurred at ratings 5-6?
-------------------------------------------------------------------------
The upper-end group was small (n = {len(high)}), so its estimates are imprecise and confidence intervals should be emphasized. Nevertheless, {int((high > 10).sum())} of {len(high)} participants exceeded 10 completions, {int((high > 20).sum())} exceeded 20, and one reached {high.max():.0f}. Thus high perceived difficulty can be assessed as a competing explanation without treating it as determinative of low persistence.

Supplementary nonlinear check
-----------------------------
Adding difficulty_c squared to the linear model gave LR chi-square({quadratic_df}) = {quadratic_lr:.3f}, {fmt_p(quadratic_p)} (quadratic-model AIC = {quadratic_fit.aic:.3f}, BIC = {quadratic_fit.bic:.3f}). This is supplementary and is not the principal test.

Interpretive caution
--------------------
This analysis is exploratory. A non-significant coefficient is not evidence that high difficulty has no effect. In particular, only {len(high)} participants supplied information at ratings 5-6, making the upper-end estimates uncertain.
"""
    (OUTPUT_DIR / "difficulty_exploratory_summary.txt").write_text(text, encoding="utf-8")


def main() -> None:
    data = load_data()
    fits = {name: fit_nb(formula, data) for name, formula in FORMULAS.items()}
    failed = [name for name, fit in fits.items() if not bool(fit.mle_retvals.get("converged", False))]
    if failed:
        raise RuntimeError("The following models did not converge: " + ", ".join(failed))

    verification = verify_primary_reproduction(fits["primary"])
    if verification["status"] == "failed":
        raise RuntimeError("Primary reproduction check failed. " + str(verification["detail"]))

    coefficients = coefficient_table(fits)
    comparison = model_comparison(fits)
    descriptives = descriptive_table(data)
    predictions = prediction_table(data, fits)

    comparison.to_csv(OUTPUT_DIR / "difficulty_exploratory_model_comparison.csv", index=False)
    coefficients.to_csv(OUTPUT_DIR / "difficulty_exploratory_coefficients.csv", index=False)
    descriptives.to_csv(OUTPUT_DIR / "difficulty_descriptives.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "difficulty_exploratory_predictions.csv", index=False)
    make_figure(
        data,
        predictions,
        OUTPUT_DIR / "difficulty_exploratory_observed_and_hinge.png",
    )
    make_figure(
        data,
        predictions,
        OUTPUT_DIR / "difficulty_exploratory_observed_and_hinge_log.png",
        log_scale=True,
    )
    write_summary(
        data, fits, coefficients, comparison, descriptives, predictions, verification
    )

    print(f"Completed difficulty analysis with {len(data)} complete cases.")
    print(f"Primary reproduction check: {verification['status']}.")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
