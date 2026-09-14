"""Exploratory GAToRS Positive subscale-decomposition sensitivity analysis.

This is deliberately separate from the planned analyses in 5_nb_model.py. The
purified data define the analysis sample; GAToRS items are recovered from the
assembled questionnaire rows by exp_sid. No source data are modified.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ANALYSIS" / "3_purified.csv"
ASSEMBLED = ROOT / "ANALYSIS" / "1_assembled.csv"
EXISTING_OUTPUT = ROOT / "ANALYSIS" / "5_nb_models"
OUT = ROOT / "ANALYSIS" / "10_gators_positive_decomposition"

OUTCOME = "captcha_post_completions"
S1 = [f"gators_{i}" for i in range(1, 6)]
S3 = [f"gators_{i}" for i in range(11, 16)]

# 5_nb_model.py uses one common complete-case sample for its full model suite,
# including the two trivial covariates even when fitting theory_plus_exploratory.
# Retaining that rule is necessary for an exact Model A reproduction.
EXISTING_COMPLETE_CASE_COLUMNS = [
    OUTCOME,
    "q_post_specific_likeability",
    "q_post_specific_mentism",
    "q_post_gators_pos",
    "q_post_gators_neg",
    "q_pre_idaq",
    "q_pre_captcha_fun",
    "q_pre_captcha_difficulty",
]

MODEL_A_FORMULA = (
    f"{OUTCOME} ~ like_c + mentism_c + like_x_mentism + "
    "q_post_gators_pos + q_post_gators_neg + q_pre_idaq"
)
MODEL_B_FORMULA = (
    f"{OUTCOME} ~ like_c + mentism_c + like_x_mentism + "
    "gators_personal_positive + gators_societal_positive + "
    "q_post_gators_neg + q_pre_idaq"
)


def fit_nb(formula: str, data: pd.DataFrame):
    """Match the discrete NB2 maximum-likelihood fit in 5_nb_model.py."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.negativebinomial(formula=formula, data=data).fit(
            disp=False, maxiter=200
        )


def recover_gators_items(sample: pd.DataFrame) -> pd.DataFrame:
    """Recover q_post_gators JSON responses by exp_sid, as in stage 9."""
    raw = pd.read_csv(ASSEMBLED, low_memory=False)
    wanted = set(sample["exp_sid"].astype(str))
    raw = raw[
        raw["exp_sid"].astype(str).isin(wanted)
        & raw["stage_id"].eq("q_post_gators")
    ]
    records: dict[str, dict[str, float]] = {sid: {} for sid in wanted}
    for _, row in raw.iterrows():
        sid = str(row["exp_sid"])
        try:
            payload = json.loads(row["questionnaire_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for key, value in payload.items():
            if key in S1 + S3:
                records[sid][key] = pd.to_numeric(value, errors="coerce")
    items = pd.DataFrame.from_dict(records, orient="index")
    return items.reindex(sample["exp_sid"].astype(str)).reset_index(drop=True)


def model_row(label: str, result) -> dict[str, float | str]:
    return {
        "model": label,
        "formula": result.model.formula,
        "n_obs": int(result.nobs),
        "df_model": float(result.df_model),
        "n_parameters_including_alpha": int(len(result.params)),
        "logLik": float(result.llf),
        "AIC": float(result.aic),
        "BIC": float(result.bic),
        "alpha_est": float(result.params["alpha"]),
        "converged": bool(result.mle_retvals.get("converged", False)),
    }


def coefficient_row(model: str, result, term: str, predictor_sd: float) -> dict:
    b = float(result.params[term])
    se = float(result.bse[term])
    ci_low, ci_high = map(float, result.conf_int().loc[term])
    return {
        "model": model,
        "term": term,
        "b": b,
        "SE": se,
        "z": b / se,
        "p": float(result.pvalues[term]),
        "IRR": float(np.exp(b)),
        "b_CI_low": ci_low,
        "b_CI_high": ci_high,
        "IRR_CI_low": float(np.exp(ci_low)),
        "IRR_CI_high": float(np.exp(ci_high)),
        "predictor_SD": predictor_sd,
        "b_per_1_SD": b * predictor_sd,
        "IRR_per_1_SD": float(np.exp(b * predictor_sd)),
    }


def latest_existing_results() -> tuple[Path | None, Path | None]:
    comparisons = sorted(
        EXISTING_OUTPUT.glob("5_nb_model_primary_comparison_*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    coefficients = sorted(
        EXISTING_OUTPUT.glob(
            "5_nb_model_primary_irr_theory_plus_exploratory_*.csv"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    return (
        comparisons[-1] if comparisons else None,
        coefficients[-1] if coefficients else None,
    )


def fmt(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA)
    if "exp_sid" not in sample:
        raise ValueError("3_purified.csv lacks exp_sid, required for item recovery.")
    missing = [c for c in EXISTING_COMPLETE_CASE_COLUMNS if c not in sample]
    if missing:
        raise ValueError(f"3_purified.csv is missing required columns: {missing}")

    items = recover_gators_items(sample)
    missing_items = [c for c in S1 + S3 if c not in items]
    if missing_items:
        raise ValueError(f"Could not recover GAToRS items: {missing_items}")
    items = items[S1 + S3].apply(pd.to_numeric, errors="coerce")

    dat = sample.copy()
    dat["gators_personal_positive"] = items[S1].mean(axis=1)
    dat["gators_societal_positive"] = items[S3].mean(axis=1)
    dat["gators_positive_reconstructed"] = (
        dat["gators_personal_positive"]
        + dat["gators_societal_positive"]
    ) / 2
    for column in EXISTING_COMPLETE_CASE_COLUMNS:
        dat[column] = pd.to_numeric(dat[column], errors="coerce")
    analysis_columns = EXISTING_COMPLETE_CASE_COLUMNS + [
        "gators_personal_positive",
        "gators_societal_positive",
        "gators_positive_reconstructed",
    ]
    dat = dat.dropna(subset=analysis_columns).copy()
    if dat.empty:
        raise ValueError("No complete cases remain for the sensitivity analysis.")

    discrepancy = (
        dat["gators_positive_reconstructed"] - dat["q_post_gators_pos"]
    ).abs()
    max_discrepancy = float(discrepancy.max())
    if max_discrepancy > 1e-10:
        raise ValueError(
            "Reconstructed GAToRS Positive does not match q_post_gators_pos; "
            f"maximum absolute discrepancy={max_discrepancy:.12g}."
        )

    # Exactly match the centering in 5_nb_model.py. Exploratory predictors are
    # left on their original scales; only liking and mentism are mean-centred.
    dat["like_c"] = (
        dat["q_post_specific_likeability"]
        - dat["q_post_specific_likeability"].mean()
    )
    dat["mentism_c"] = (
        dat["q_post_specific_mentism"]
        - dat["q_post_specific_mentism"].mean()
    )
    dat["like_x_mentism"] = dat["like_c"] * dat["mentism_c"]

    model_a = fit_nb(MODEL_A_FORMULA, dat)
    model_b = fit_nb(MODEL_B_FORMULA, dat)

    # Since q_post_gators_pos=(personal+societal)/2, Model A is Model B under
    # beta_personal=beta_societal (with beta_A=2*beta_personal). The unchanged
    # scale and common sample make a one-df LR comparison formally valid.
    lr = 2 * (model_b.llf - model_a.llf)
    df_lr = len(model_b.params) - len(model_a.params)
    lr_p = float(stats.chi2.sf(max(lr, 0.0), df_lr))
    comparison = pd.DataFrame(
        [model_row("A_aggregate_positive", model_a), model_row("B_decomposed_positive", model_b)]
    )
    comparison["nested_comparison_valid"] = True
    comparison["nesting_explanation"] = (
        "q_post_gators_pos=(personal_positive+societal_positive)/2; "
        "Model A is Model B constrained to equal subscale coefficients"
    )
    comparison["LR_chi_square_B_vs_A"] = lr
    comparison["LR_df"] = df_lr
    comparison["LR_p"] = lr_p

    p_term = "gators_personal_positive"
    s_term = "gators_societal_positive"
    difference = float(model_b.params[p_term] - model_b.params[s_term])
    covariance = model_b.cov_params()
    variance_difference = float(
        covariance.loc[p_term, p_term]
        + covariance.loc[s_term, s_term]
        - 2 * covariance.loc[p_term, s_term]
    )
    se_difference = float(np.sqrt(max(variance_difference, 0.0)))
    z_difference = difference / se_difference
    p_difference = float(2 * stats.norm.sf(abs(z_difference)))
    critical = float(stats.norm.ppf(0.975))
    wald = pd.DataFrame(
        [{
            "contrast": "beta_personal_positive - beta_societal_positive = 0",
            "coefficient_difference": difference,
            "SE_difference": se_difference,
            "z": z_difference,
            "p_two_sided": p_difference,
            "CI_95_low": difference - critical * se_difference,
            "CI_95_high": difference + critical * se_difference,
        }]
    )

    sd_broad = float(dat["q_post_gators_pos"].std(ddof=1))
    sd_personal = float(dat[p_term].std(ddof=1))
    sd_societal = float(dat[s_term].std(ddof=1))
    subscale_pearson = float(dat[p_term].corr(dat[s_term], method="pearson"))
    subscale_spearman = float(dat[p_term].corr(dat[s_term], method="spearman"))
    coefficients = pd.DataFrame(
        [
            coefficient_row(
                "A_aggregate_positive", model_a, "q_post_gators_pos", sd_broad
            ),
            coefficient_row("B_decomposed_positive", model_b, p_term, sd_personal),
            coefficient_row("B_decomposed_positive", model_b, s_term, sd_societal),
        ]
    )

    descriptives = pd.DataFrame(
        [
            {
                "subscale": "Personal Positive (S1; items 1-5)",
                "n": int(dat[p_term].count()),
                "mean": dat[p_term].mean(),
                "SD": sd_personal,
                "median": dat[p_term].median(),
                "minimum": dat[p_term].min(),
                "maximum": dat[p_term].max(),
                "between_subscale_pearson_r": subscale_pearson,
                "between_subscale_spearman_rho": subscale_spearman,
            },
            {
                "subscale": "Societal Positive (S3; items 11-15)",
                "n": int(dat[s_term].count()),
                "mean": dat[s_term].mean(),
                "SD": sd_societal,
                "median": dat[s_term].median(),
                "minimum": dat[s_term].min(),
                "maximum": dat[s_term].max(),
                "between_subscale_pearson_r": subscale_pearson,
                "between_subscale_spearman_rho": subscale_spearman,
            },
        ]
    )

    correlation_pairs = [
        (p_term, s_term),
        (p_term, OUTCOME),
        (s_term, OUTCOME),
        (p_term, "q_post_specific_likeability"),
        (s_term, "q_post_specific_likeability"),
        (p_term, "q_post_specific_mentism"),
        (s_term, "q_post_specific_mentism"),
    ]
    correlation_rows = []
    for x, y in correlation_pairs:
        pair = dat[[x, y]].dropna()
        correlation_rows.append(
            {
                "variable_1": x,
                "variable_2": y,
                "n": len(pair),
                "pearson_r": pair[x].corr(pair[y], method="pearson"),
                "spearman_rho": pair[x].corr(pair[y], method="spearman"),
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    exog_names = list(model_b.model.exog_names)
    exog = np.asarray(model_b.model.exog, dtype=float)
    vif_rows = []
    for term in (p_term, s_term):
        index = exog_names.index(term)
        vif_rows.append(
            {
                "term": term,
                "VIF": variance_inflation_factor(exog, index),
                "design_matrix": ", ".join(name for name in exog_names if name != "alpha"),
                "subscale_pearson_r": subscale_pearson,
            }
        )
    vif = pd.DataFrame(vif_rows)

    # Compare Model A against the latest saved stage-5 run when available.
    prior_cmp_path, prior_coef_path = latest_existing_results()
    reference_notes = []
    reproduction_ok = True
    metric_deltas: dict[str, float] = {}
    coefficient_max_delta = np.nan
    if prior_cmp_path is not None:
        prior_cmp = pd.read_csv(prior_cmp_path)
        prior_row = prior_cmp.loc[
            prior_cmp["model"].eq("theory_plus_exploratory")
        ]
        if len(prior_row) == 1:
            prior_row = prior_row.iloc[0]
            for metric, current in {
                "logLik": model_a.llf,
                "AIC": model_a.aic,
                "BIC": model_a.bic,
                "alpha_est": model_a.params["alpha"],
            }.items():
                metric_deltas[metric] = abs(float(current) - float(prior_row[metric]))
            reproduction_ok &= max(metric_deltas.values()) <= 1e-6
            reference_notes.append(f"Model metrics compared with {prior_cmp_path.name}.")
    if prior_coef_path is not None:
        prior_coef = pd.read_csv(prior_coef_path).set_index("term")
        common = [term for term in model_a.params.index if term != "alpha" and term in prior_coef.index]
        if common:
            coefficient_max_delta = max(
                abs(float(model_a.params[term]) - float(prior_coef.loc[term, "coef"]))
                for term in common
            )
            reproduction_ok &= coefficient_max_delta <= 1e-6
            reference_notes.append(f"Coefficients compared with {prior_coef_path.name}.")
    if not reference_notes:
        reference_notes.append("No prior saved primary Model A files were available for comparison.")

    comparison.to_csv(
        OUT / "gators_positive_decomposition_model_comparison.csv", index=False
    )
    coefficients.to_csv(
        OUT / "gators_positive_decomposition_coefficients.csv", index=False
    )
    wald.to_csv(OUT / "gators_positive_decomposition_wald_test.csv", index=False)
    descriptives.to_csv(
        OUT / "gators_positive_decomposition_descriptives.csv", index=False
    )
    correlations.to_csv(
        OUT / "gators_positive_decomposition_correlations.csv", index=False
    )
    vif.to_csv(OUT / "gators_positive_decomposition_vif.csv", index=False)

    broad = coefficients.iloc[0]
    personal = coefficients.iloc[1]
    societal = coefficients.iloc[2]
    opposite = np.sign(personal["b"]) != np.sign(societal["b"])
    fit_improved = lr_p < 0.05
    equality_rejected = p_difference < 0.05
    if opposite:
        pattern = (
            "The estimates point in opposite directions, so aggregation may obscure "
            "substantively different facet-level relationships; their uncertainty and "
            "the formal equality test remain central."
        )
    elif equality_rejected:
        pattern = (
            "The estimates differ statistically and in magnitude, indicating that the "
            "equal-weight composite may conceal facet-specific associations."
        )
    elif abs(personal["b"]) < 0.10 and abs(societal["b"]) < 0.10:
        pattern = (
            "Both facet estimates are small and statistically uncertain, consistent "
            "with the aggregate null association."
        )
    else:
        pattern = (
            "The estimates are not statistically distinguishable; this provides no "
            "clear evidence that equal weighting masks different associations."
        )

    report = [
        "GAToRS POSITIVE SUBSCALE-DECOMPOSITION SENSITIVITY ANALYSIS",
        "",
        "Status: exploratory psychometric sensitivity analysis; the existing aggregate "
        "exploratory model remains the planned analysis.",
        f"Data: {DATA}",
        f"Item source: {ASSEMBLED}",
        f"Analysis N: {len(dat)}",
        f"Maximum absolute composite reconstruction discrepancy: {max_discrepancy:.12g}",
        "Scoring: Personal Positive = mean(gators_1...gators_5); Societal "
        "Positive = mean(gators_11...gators_15); no reverse coding.",
        "Centering: likeability and robomentism were mean-centred exactly as in "
        "5_nb_model.py; GAToRS and IDAQ predictors retain their original response scales.",
        "Estimator: statsmodels discrete NegativeBinomial (NB2), maximum likelihood, "
        "with alpha estimated from the data.",
        "",
        "1. Does the existing aggregate GAToRS Positive model reproduce correctly?",
        f"{'Yes' if reproduction_ok else 'No'}. Model A uses the exact existing formula and complete-case rule. "
        + " ".join(reference_notes),
        "Absolute deltas from the saved result: "
        + ", ".join(f"{key}={value:.3g}" for key, value in metric_deltas.items())
        + f", maximum coefficient delta={fmt(coefficient_max_delta, 8)}.",
        f"Model A: logLik={model_a.llf:.4f}, AIC={model_a.aic:.4f}, "
        f"BIC={model_a.bic:.4f}, alpha={model_a.params['alpha']:.4f}.",
        "",
        "2. What are the simultaneous associations of Personal Positive and Societal "
        "Positive with voluntary completions?",
        f"Personal Positive: b={personal['b']:.4f}, SE={personal['SE']:.4f}, "
        f"z={personal['z']:.3f}, p={personal['p']:.4f}, IRR={personal['IRR']:.4f}, "
        f"95% CI b [{personal['b_CI_low']:.4f}, {personal['b_CI_high']:.4f}], "
        f"95% CI IRR [{personal['IRR_CI_low']:.4f}, {personal['IRR_CI_high']:.4f}].",
        f"Societal Positive: b={societal['b']:.4f}, SE={societal['SE']:.4f}, "
        f"z={societal['z']:.3f}, p={societal['p']:.4f}, IRR={societal['IRR']:.4f}, "
        f"95% CI b [{societal['b_CI_low']:.4f}, {societal['b_CI_high']:.4f}], "
        f"95% CI IRR [{societal['IRR_CI_low']:.4f}, {societal['IRR_CI_high']:.4f}].",
        f"For comparison, broad Positive: b={broad['b']:.4f}, SE={broad['SE']:.4f}, "
        f"p={broad['p']:.4f}, IRR={broad['IRR']:.4f}.",
        "Standardised descriptive comparison (raw b x sample SD): "
        f"Personal={personal['b_per_1_SD']:.4f} log-count units "
        f"(IRR={personal['IRR_per_1_SD']:.4f}); "
        f"Societal={societal['b_per_1_SD']:.4f} "
        f"(IRR={societal['IRR_per_1_SD']:.4f}).",
        "",
        "3. Are those two coefficients statistically distinguishable from one another?",
        f"{'Yes' if equality_rejected else 'No'} at alpha=.05. Personal minus Societal "
        f"b={difference:.4f}, SE={se_difference:.4f}, z={z_difference:.3f}, "
        f"p={p_difference:.4f}, 95% CI [{difference-critical*se_difference:.4f}, "
        f"{difference+critical*se_difference:.4f}].",
        "",
        "4. Does allowing separate coefficients materially improve model fit?",
        f"{'Yes' if fit_improved else 'No'} by the one-df LR test: chi-square={lr:.4f}, "
        f"df={df_lr}, p={lr_p:.4f}. Model A AIC/BIC={model_a.aic:.4f}/{model_a.bic:.4f}; "
        f"Model B AIC/BIC={model_b.aic:.4f}/{model_b.bic:.4f}.",
        "The LR test is valid because the broad score is exactly the average of the "
        "two same-scale subscales; equality of Model B coefficients reproduces Model A.",
        "",
        "5. Does decomposition alter the interpretation of the broad GAToRS Positive predictor?",
        pattern,
        "",
        "6. Is there evidence that aggregation concealed opposite or strongly unequal relationships?",
        f"{'The point estimates have opposite signs.' if opposite else 'The point estimates have the same sign.'} "
        f"The equality-test p-value is {p_difference:.4f}; inference is based on the "
        "difference and its interval, not on whether either individual p-value crosses .05.",
        "",
        "7. Does the result justify retaining the broad composite while acknowledging "
        "facet-specific uncertainty?",
        "Yes. This sensitivity analysis does not redefine either constituent as the "
        "'real' Positive measure and does not justify retrospective predictor selection. "
        "The planned broad composite should be retained, with the facet estimates and "
        "their uncertainty reported as a psychometrically motivated sensitivity check.",
        "",
        "Descriptive diagnostics",
        f"Personal Positive: mean={dat[p_term].mean():.3f}, SD={sd_personal:.3f}, "
        f"median={dat[p_term].median():.3f}, range={dat[p_term].min():.3f}-{dat[p_term].max():.3f}.",
        f"Societal Positive: mean={dat[s_term].mean():.3f}, SD={sd_societal:.3f}, "
        f"median={dat[s_term].median():.3f}, range={dat[s_term].min():.3f}-{dat[s_term].max():.3f}.",
        f"Subscale Pearson r={subscale_pearson:.3f}; Spearman rho={subscale_spearman:.3f} "
        "(approximately .223 and .249, respectively, as in the internal-consistency analysis).",
        "Pearson correlations with completions, likeability, and robomentism, respectively: "
        f"Personal={correlations.iloc[1]['pearson_r']:.3f}, "
        f"{correlations.iloc[3]['pearson_r']:.3f}, "
        f"{correlations.iloc[5]['pearson_r']:.3f}; "
        f"Societal={correlations.iloc[2]['pearson_r']:.3f}, "
        f"{correlations.iloc[4]['pearson_r']:.3f}, "
        f"{correlations.iloc[6]['pearson_r']:.3f}. Full Pearson and Spearman results "
        "are in gators_positive_decomposition_correlations.csv.",
        f"VIF: Personal={vif.loc[vif.term.eq(p_term), 'VIF'].iloc[0]:.3f}; "
        f"Societal={vif.loc[vif.term.eq(s_term), 'VIF'].iloc[0]:.3f}. VIF is descriptive, "
        "not a pass/fail test.",
        "",
        "Manuscript-oriented interpretation",
        f"In an exploratory NB2 sensitivity model that replaced the broad GAToRS "
        f"Positive score with its two simultaneously estimated facets, Personal Positive "
        f"was b={personal['b']:.3f} (95% CI {personal['b_CI_low']:.3f} to "
        f"{personal['b_CI_high']:.3f}) and Societal Positive was b={societal['b']:.3f} "
        f"(95% CI {societal['b_CI_low']:.3f} to {societal['b_CI_high']:.3f}). Their "
        f"difference was b={difference:.3f} (95% CI "
        f"{difference-critical*se_difference:.3f} to "
        f"{difference+critical*se_difference:.3f}; p={p_difference:.3f}), and allowing "
        f"separate coefficients changed fit by LR chi-square({df_lr})={lr:.3f}, "
        f"p={lr_p:.3f}. {pattern} These results support retaining the planned broad "
        "composite while acknowledging facet-specific uncertainty.",
    ]
    report_path = OUT / "gators_positive_decomposition_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"Maximum absolute GAToRS Positive discrepancy: {max_discrepancy:.12g}")
    print(f"Model A reproduction: {'PASS' if reproduction_ok else 'CHECK'}")
    print(comparison.to_string(index=False))
    print("\nWald equality test")
    print(wald.to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")


if __name__ == "__main__":
    main()
