import os
import time
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    from scipy.stats import chi2  # optional, for LR-test p-values
except Exception:
    chi2 = None


OUTCOME = "captcha_post_completions"
PRIMARY = [
    "q_post_specific_empathy",
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
    dat["emp_c"] = dat["q_post_specific_empathy"] - dat["q_post_specific_empathy"].mean()
    dat["ment_c"] = dat["q_post_specific_mentacy_belief_scale"] - dat["q_post_specific_mentacy_belief_scale"].mean()
    dat["emp_x_ment"] = dat["emp_c"] * dat["ment_c"]

    formulas = {
        "theory_primary_only": "{} ~ emp_c + ment_c".format(OUTCOME),
        "theory_plus_exploratory": (
            "{} ~ emp_c + ment_c + emp_x_ment + q_post_gators_pos + q_post_gators_neg + q_pre_idaq".format(OUTCOME)
        ),
        "trivial_only": "{} ~ q_pre_captcha_fun + q_pre_captcha_difficulty".format(OUTCOME),
        "theory_plus_exploratory_plus_trivial": (
            "{} ~ emp_c + ment_c + emp_x_ment + q_post_gators_pos + q_post_gators_neg + q_pre_idaq + "
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
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(here, "2_simplified.csv")
    run_nb_model_suite(default_csv, out_dir=here)
