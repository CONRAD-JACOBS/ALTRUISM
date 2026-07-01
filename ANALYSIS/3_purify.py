#!/usr/bin/python3
import argparse
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

"""
I used a conservative default cutoff of max(0.50, 2/sqrt(n)), which is 0.50 for the current sample. You can adjust it with:
python3 ALTRUISM/ANALYSIS/3_purify.py --threshold 0.75
python3 ALTRUISM/ANALYSIS/3_purify.py --threshold-like 0.75 --threshold-ment 0.6 --threshold-interaction 0.8
Verification run succeeded. With the current data, it excluded participants 51, 76, 78, and 98, leaving 95 rows in 3_purified.csv. The latest summary is [3_purify_summary_20260701_134417.txt (line 1)](/Users/neurorobots/Desktop/repos/ALTRUISM/ANALYSIS/3_purify/3_purify_summary_20260701_134417.txt:1).
"""

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "ANALYSIS"
INFILE = HERE / "2_simplified.csv"
OUTFILE = HERE / "3_purified.csv"
OUTDIR = HERE / "3_purify"

OUTCOME = "captcha_post_completions"
LIKE_COL = "q_post_specific_likeability"
MENT_COL = "q_post_specific_mentacy_belief_scale"
PREDICTORS = ["like_c", "ment_c", "like_x_ment"]
PREDICTOR_LABELS = {
    "like_c": "LIKING",
    "ment_c": "MENT",
    "like_x_ment": "LIKINGxMENT",
}

# Conventional DFBETAS screening is often 2 / sqrt(n). That can be too eager
# for small, noisy psychology samples, so use a more conservative floor.
DEFAULT_THRESHOLD_FLOOR = 0.50


def _timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def _fit_nb(formula, df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.negativebinomial(formula=formula, data=df).fit(disp=False, maxiter=300)


def _prepare_model_data(df):
    need = ["exp_sid", "participant_number", OUTCOME, LIKE_COL, MENT_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))

    dat = df[need].copy()
    for c in [OUTCOME, LIKE_COL, MENT_COL, "participant_number"]:
        dat[c] = pd.to_numeric(dat[c], errors="coerce")
    dat = dat.dropna(subset=[OUTCOME, LIKE_COL, MENT_COL, "participant_number"]).copy()
    if dat.empty:
        raise ValueError("No complete rows remain for the primary NB model.")

    like_mean = dat[LIKE_COL].mean()
    ment_mean = dat[MENT_COL].mean()
    dat["like_c"] = dat[LIKE_COL] - like_mean
    dat["ment_c"] = dat[MENT_COL] - ment_mean
    dat["like_x_ment"] = dat["like_c"] * dat["ment_c"]
    dat["participant_number"] = dat["participant_number"].astype(int)
    return dat, like_mean, ment_mean


def _default_threshold(n_obs):
    return max(DEFAULT_THRESHOLD_FLOOR, 2.0 / np.sqrt(float(n_obs)))


def _thresholds_from_args(args, n_obs):
    default = args.threshold
    if default is None:
        default = _default_threshold(n_obs)
    return {
        "like_c": args.threshold_like if args.threshold_like is not None else default,
        "ment_c": args.threshold_ment if args.threshold_ment is not None else default,
        "like_x_ment": args.threshold_interaction if args.threshold_interaction is not None else default,
    }


def _plot_predictor(influence_df, predictor, threshold, out_path):
    label = PREDICTOR_LABELS[predictor]
    d = influence_df.sort_values("participant_number").copy()
    colors = np.where(d["exclude_flag"], "#b24a2a", "#2f5d8a")

    plt.figure(figsize=(10, 4.8))
    plt.axhline(0.0, color="#333333", linewidth=0.8)
    plt.axhline(threshold, color="#8a1f1f", linestyle="--", linewidth=1.0)
    plt.axhline(-threshold, color="#8a1f1f", linestyle="--", linewidth=1.0)
    plt.scatter(
        d["participant_number"],
        d["dfbetas_{}".format(predictor)],
        c=colors,
        s=36,
        edgecolors="white",
        linewidths=0.6,
    )
    plt.xlabel("Participant number")
    plt.ylabel("Leave-one-out DFBETAS")
    plt.title("{} influence on primary NB beta".format(label))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_combined(influence_df, thresholds, out_path):
    d = influence_df.sort_values("participant_number").copy()
    plt.figure(figsize=(11, 5.5))
    for predictor in PREDICTORS:
        plt.plot(
            d["participant_number"],
            d["dfbetas_{}".format(predictor)].abs(),
            marker="o",
            linewidth=1.2,
            markersize=4,
            label=PREDICTOR_LABELS[predictor],
        )
        plt.axhline(thresholds[predictor], linestyle="--", linewidth=0.9, alpha=0.45)
    plt.xlabel("Participant number")
    plt.ylabel("Absolute leave-one-out DFBETAS")
    plt.title("Primary NB leave-one-out influence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_purify(infile=INFILE, outfile=OUTFILE, outdir=OUTDIR, thresholds=None):
    infile = Path(infile)
    outfile = Path(outfile)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    dat, like_mean, ment_mean = _prepare_model_data(df)
    formula = "{} ~ like_c + ment_c + like_x_ment".format(OUTCOME)
    full_fit = _fit_nb(formula, dat)

    if thresholds is None:
        thresholds = {p: _default_threshold(len(dat)) for p in PREDICTORS}

    full_params = full_fit.params.reindex(PREDICTORS)
    full_se = full_fit.bse.reindex(PREDICTORS).replace(0, np.nan)

    rows = []
    errors = []
    for _, row in dat[["exp_sid", "participant_number"]].drop_duplicates().sort_values("participant_number").iterrows():
        exp_sid = row["exp_sid"]
        participant_number = int(row["participant_number"])
        loo = dat[dat["exp_sid"] != exp_sid].copy()
        try:
            loo_fit = _fit_nb(formula, loo)
            loo_params = loo_fit.params.reindex(PREDICTORS)
            delta = loo_params - full_params
            dfbetas = delta / full_se
            out = {
                "exp_sid": exp_sid,
                "participant_number": participant_number,
                "n_full": int(full_fit.nobs),
                "n_leave_one_out": int(loo_fit.nobs),
            }
            for predictor in PREDICTORS:
                out["full_beta_{}".format(predictor)] = full_params[predictor]
                out["loo_beta_{}".format(predictor)] = loo_params[predictor]
                out["delta_beta_{}".format(predictor)] = delta[predictor]
                out["dfbetas_{}".format(predictor)] = dfbetas[predictor]
            rows.append(out)
        except Exception as exc:
            errors.append({
                "exp_sid": exp_sid,
                "participant_number": participant_number,
                "error": str(exc),
            })

    influence_df = pd.DataFrame(rows)
    if influence_df.empty:
        raise ValueError("No leave-one-out models fit successfully.")

    for predictor in PREDICTORS:
        influence_df["crosses_{}".format(predictor)] = (
            influence_df["dfbetas_{}".format(predictor)].abs() > float(thresholds[predictor])
        )
    cross_cols = ["crosses_{}".format(p) for p in PREDICTORS]
    influence_df["exclude_flag"] = influence_df[cross_cols].any(axis=1)

    excluded = influence_df.loc[influence_df["exclude_flag"], ["exp_sid", "participant_number"]].copy()
    purified = df.loc[~df["exp_sid"].isin(set(excluded["exp_sid"]))].copy()
    purified.to_csv(outfile, index=False)

    stamp = _timestamp()
    influence_csv = outdir / "3_purify_leave_one_out_dfbetas_{}.csv".format(stamp)
    excluded_csv = outdir / "3_purify_excluded_participants_{}.csv".format(stamp)
    summary_txt = outdir / "3_purify_summary_{}.txt".format(stamp)
    influence_df.to_csv(influence_csv, index=False)
    excluded.to_csv(excluded_csv, index=False)

    plot_paths = []
    for predictor in PREDICTORS:
        plot_path = outdir / "3_purify_dfbetas_{}_{}.png".format(predictor, stamp)
        _plot_predictor(influence_df, predictor, thresholds[predictor], plot_path)
        plot_paths.append(plot_path)
    combined_plot = outdir / "3_purify_dfbetas_combined_{}.png".format(stamp)
    _plot_combined(influence_df, thresholds, combined_plot)
    plot_paths.append(combined_plot)

    errors_csv = None
    if errors:
        errors_csv = outdir / "3_purify_fit_errors_{}.csv".format(stamp)
        pd.DataFrame(errors).to_csv(errors_csv, index=False)

    with open(summary_txt, "w") as f:
        f.write("DFBETAS / leave-one-out purification\n")
        f.write("input: {}\n".format(infile))
        f.write("output: {}\n".format(outfile))
        f.write("rows_in_input: {}\n".format(len(df)))
        f.write("rows_used_for_primary_model: {}\n".format(len(dat)))
        f.write("rows_in_purified_output: {}\n".format(len(purified)))
        f.write("formula: {}\n".format(formula))
        f.write("centering_likeability_mean: {}\n".format(like_mean))
        f.write("centering_mentacy_mean: {}\n".format(ment_mean))
        f.write("thresholds: {}\n".format({PREDICTOR_LABELS[k]: float(v) for k, v in thresholds.items()}))
        f.write("excluded_participant_count: {}\n\n".format(len(excluded)))
        f.write("Full model coefficients\n")
        f.write(full_fit.params.to_string())
        f.write("\n\nExcluded participants\n")
        if excluded.empty:
            f.write("None\n")
        else:
            f.write(excluded.to_string(index=False))
            f.write("\n")
        if errors:
            f.write("\nLeave-one-out fit errors\n")
            f.write(pd.DataFrame(errors).to_string(index=False))
            f.write("\n")

    print("Input rows: {}".format(len(df)))
    print("Rows used in primary model: {}".format(len(dat)))
    print("Excluded participants: {}".format(len(excluded)))
    if not excluded.empty:
        print(excluded.to_string(index=False))
    print("Output rows: {}".format(len(purified)))
    print("Wrote {}".format(outfile))
    print("Diagnostics:")
    print("- {}".format(influence_csv))
    print("- {}".format(excluded_csv))
    print("- {}".format(summary_txt))
    for path in plot_paths:
        print("- {}".format(path))
    if errors_csv is not None:
        print("- {}".format(errors_csv))

    return {
        "data_used": dat,
        "full_fit": full_fit,
        "influence": influence_df,
        "excluded": excluded,
        "purified": purified,
        "thresholds": thresholds,
        "influence_csv": influence_csv,
        "excluded_csv": excluded_csv,
        "summary_txt": summary_txt,
        "plots": plot_paths,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run primary NB leave-one-out DFBETAS diagnostics and write 3_purified.csv."
    )
    parser.add_argument("--input", default=os.environ.get("ANALYSIS_INPUT_CSV", str(INFILE)))
    parser.add_argument("--output", default=str(OUTFILE))
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Default absolute DFBETAS cutoff for all predictors. Defaults to max(0.50, 2/sqrt(n)).",
    )
    parser.add_argument("--threshold-like", type=float, default=None)
    parser.add_argument("--threshold-ment", type=float, default=None)
    parser.add_argument("--threshold-interaction", type=float, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    dat, _, _ = _prepare_model_data(df)
    thresholds = _thresholds_from_args(args, len(dat))
    run_purify(args.input, args.output, args.outdir, thresholds=thresholds)


if __name__ == "__main__":
    main()
