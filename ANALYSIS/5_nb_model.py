import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    from scipy.stats import chi2  # optional, for LR-test p-values
except Exception:
    chi2 = None


OUTCOME = "captcha_post_completions"
PRIMARY = [
    "q_post_specific_likeability",
    "q_post_specific_mentacy_belief_scale",
]
EXPLORATORY = [
    "q_post_gators_pos",
    "q_post_gators_neg",
    "q_pre_idaq",
]
TRIVIAL = [
    "q_pre_captcha_fun",
    "q_pre_captcha_difficulty",
]


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _irr_table(result):
    p = result.params
    ci = result.conf_int()
    out = pd.DataFrame(
        {
            "coef": p,
            "IRR": np.exp(p),
            "CI_low_IRR": np.exp(ci[0]),
            "CI_high_IRR": np.exp(ci[1]),
            "p_value": result.pvalues,
        }
    )
    return out


def _fit_nb(formula, df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.negativebinomial(formula=formula, data=df).fit(disp=False, maxiter=200)


def _lr_compare(smaller, larger):
    ll_small = smaller.llf
    ll_large = larger.llf
    lr = 2.0 * (ll_large - ll_small)
    df_diff = int(larger.df_model - smaller.df_model)
    p = np.nan
    if chi2 is not None and df_diff > 0:
        p = chi2.sf(lr, df_diff)
    return {"lr_stat": lr, "df_diff": df_diff, "p_value": p}


def _plot_outcome_distribution(dat, out_path):
    plt.figure(figsize=(7, 4.5))
    plt.hist(dat[OUTCOME], bins=30, color="#2f5d8a", edgecolor="white")
    plt.xlabel("Post-task completions")
    plt.ylabel("Count")
    plt.title("Distribution of post-task completions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_primary_interaction_heatmap(dat, out_path):
    like_bins = pd.qcut(dat["q_post_specific_likeability"], q=5, duplicates="drop")
    ment_bins = pd.cut(dat["q_post_specific_mentacy_belief_scale"], bins=5, include_lowest=True)

    heat = dat.pivot_table(
        index=ment_bins,
        columns=like_bins,
        values=OUTCOME,
        aggfunc="mean",
        observed=False,
    )

    plt.figure(figsize=(8, 5))
    im = plt.imshow(heat.to_numpy(), aspect="auto", cmap="YlGnBu", origin="lower")
    plt.colorbar(im, label="Mean completions")
    plt.xticks(range(len(heat.columns)), [str(c) for c in heat.columns], rotation=30, ha="right")
    plt.yticks(range(len(heat.index)), [str(i) for i in heat.index])
    plt.xlabel("Likeability bins")
    plt.ylabel("Mentacy bins")
    plt.title("Mean completions across likeability x mentacy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_primary_irr(fit, out_path):
    params = fit.params.drop(labels=["Intercept", "alpha"], errors="ignore")
    conf = fit.conf_int().loc[params.index]

    irr = np.exp(params)
    irr_low = np.exp(conf[0])
    irr_high = np.exp(conf[1])
    order = irr.sort_values().index
    y = np.arange(len(order))

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(
        irr.loc[order],
        y,
        xerr=[irr.loc[order] - irr_low.loc[order], irr_high.loc[order] - irr.loc[order]],
        fmt="o",
        color="#b24a2a",
        ecolor="#444444",
        capsize=3,
    )
    plt.axvline(1.0, linestyle="--", color="black", linewidth=1)
    plt.yticks(y, order)
    plt.xlabel("Incident Rate Ratio")
    plt.title("Primary model effect sizes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_nb_model_suite(csv_path, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    need = [OUTCOME] + PRIMARY + EXPLORATORY + TRIVIAL
    df = pd.read_csv(csv_path)

    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))

    dat = df[need].copy()
    for c in need:
        dat[c] = pd.to_numeric(dat[c], errors="coerce")
    dat = dat.dropna().copy()

    if dat.empty:
        raise ValueError("No complete rows remain after numeric coercion + dropna.")

    # Center primary terms so interaction is easier to interpret.
    dat["like_c"] = dat["q_post_specific_likeability"] - dat["q_post_specific_likeability"].mean()
    dat["ment_c"] = dat["q_post_specific_mentacy_belief_scale"] - dat["q_post_specific_mentacy_belief_scale"].mean()
    dat["like_x_ment"] = dat["like_c"] * dat["ment_c"]

    formulas = {
        "theory_primary_only": "{} ~ like_c + ment_c + like_x_ment".format(OUTCOME),
        "theory_plus_exploratory": (
            "{} ~ like_c + ment_c + like_x_ment + q_post_gators_pos + q_post_gators_neg + q_pre_idaq".format(OUTCOME)
        ),
        "trivial_only": "{} ~ q_pre_captcha_fun + q_pre_captcha_difficulty".format(OUTCOME),
        "theory_plus_exploratory_plus_trivial": (
            "{} ~ like_c + ment_c + like_x_ment + q_post_gators_pos + q_post_gators_neg + q_pre_idaq + "
            "q_pre_captcha_fun + q_pre_captcha_difficulty".format(OUTCOME)
        ),
    }

    fits = {}
    errors = {}
    for name, formula in formulas.items():
        try:
            fits[name] = _fit_nb(formula, dat)
        except Exception as e:
            errors[name] = str(e)

    model_rows = []
    for name in formulas:
        if name in fits:
            r = fits[name]
            model_rows.append(
                {
                    "model": name,
                    "n_obs": int(r.nobs),
                    "df_model": float(r.df_model),
                    "logLik": float(r.llf),
                    "AIC": float(r.aic),
                    "BIC": float(getattr(r, "bic", np.nan)),
                    "alpha_est": float(r.params.get("alpha", np.nan)),
                }
            )
        else:
            model_rows.append(
                {
                    "model": name,
                    "n_obs": np.nan,
                    "df_model": np.nan,
                    "logLik": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                    "alpha_est": np.nan,
                }
            )
    model_cmp = pd.DataFrame(model_rows).sort_values("AIC", na_position="last").reset_index(drop=True)

    lr_rows = []
    if "theory_primary_only" in fits and "theory_plus_exploratory" in fits:
        d = _lr_compare(fits["theory_primary_only"], fits["theory_plus_exploratory"])
        d["comparison"] = "theory_primary_only -> theory_plus_exploratory"
        lr_rows.append(d)
    if "theory_plus_exploratory" in fits and "theory_plus_exploratory_plus_trivial" in fits:
        d = _lr_compare(fits["theory_plus_exploratory"], fits["theory_plus_exploratory_plus_trivial"])
        d["comparison"] = "theory_plus_exploratory -> theory_plus_exploratory_plus_trivial"
        lr_rows.append(d)
    lr_df = pd.DataFrame(lr_rows)

    stamp = _timestamp()
    cmp_csv = os.path.join(out_dir, "5_nb_model_comparison_{}.csv".format(stamp))
    model_cmp.to_csv(cmp_csv, index=False)

    irr_paths = []
    for name, r in fits.items():
        t = _irr_table(r)
        p = os.path.join(out_dir, "5_nb_model_irr_{}_{}.csv".format(name, stamp))
        t.to_csv(p)
        irr_paths.append(p)

    plot_paths = []
    outcome_plot = os.path.join(out_dir, "5_nb_model_outcome_distribution_{}.png".format(stamp))
    _plot_outcome_distribution(dat, outcome_plot)
    plot_paths.append(outcome_plot)

    interaction_plot = os.path.join(out_dir, "5_nb_model_like_x_ment_heatmap_{}.png".format(stamp))
    _plot_primary_interaction_heatmap(dat, interaction_plot)
    plot_paths.append(interaction_plot)

    if "theory_primary_only" in fits:
        irr_plot = os.path.join(out_dir, "5_nb_model_primary_irr_{}.png".format(stamp))
        _plot_primary_irr(fits["theory_primary_only"], irr_plot)
        plot_paths.append(irr_plot)

    summary_txt = os.path.join(out_dir, "5_nb_model_summary_{}.txt".format(stamp))
    with open(summary_txt, "w") as f:
        f.write("Negative Binomial Model Suite\n")
        f.write("data: {}\n".format(csv_path))
        f.write("rows_used: {}\n".format(len(dat)))
        f.write("outcome: {}\n\n".format(OUTCOME))
        f.write("Model formulas\n")
        for k, v in formulas.items():
            f.write("- {}: {}\n".format(k, v))
        f.write("\nModel comparison (lower AIC/BIC preferred)\n")
        f.write(model_cmp.to_string(index=False))
        f.write("\n\n")
        if not lr_df.empty:
            f.write("Nested LR comparisons\n")
            f.write(lr_df.to_string(index=False))
            f.write("\n\n")
        if errors:
            f.write("Model fit errors\n")
            for k, v in errors.items():
                f.write("- {}: {}\n".format(k, v))
            f.write("\n")
        for name, r in fits.items():
            f.write("==== {} ====\n".format(name))
            f.write(str(r.summary()))
            f.write("\n\n")
        if len(dat) < 30:
            f.write(
                "WARNING: Very small N. Inference will be unstable and may fail to converge "
                "or overfit with richer models.\n"
            )

    print("\nSaved:")
    print("- {}".format(cmp_csv))
    print("- {}".format(summary_txt))
    for p in irr_paths:
        print("- {}".format(p))
    for p in plot_paths:
        print("- {}".format(p))
    if errors:
        print("\nSome models failed:")
        for k, v in errors.items():
            print("- {}: {}".format(k, v))
    print("\nModel comparison:")
    print(model_cmp.to_string(index=False))
    if not lr_df.empty:
        print("\nNested LR comparisons:")
        print(lr_df.to_string(index=False))

    return {
        "data_used": dat,
        "fits": fits,
        "errors": errors,
        "model_comparison": model_cmp,
        "lr_comparisons": lr_df,
        "comparison_csv": cmp_csv,
        "summary_txt": summary_txt,
        "irr_csvs": irr_paths,
        "plots": plot_paths,
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(here, "5_nb_models")
    default_csv = os.path.join(here, "2_simplified.csv")
    run_nb_model_suite(default_csv, out_dir=output)
