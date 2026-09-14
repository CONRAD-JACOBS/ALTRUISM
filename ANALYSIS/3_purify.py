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
The automated pipeline uses the fixed DFBETAS_THRESHOLDS and
COOKS_DISTANCE_THRESHOLD values defined below. Edit those constants to change
the thresholds for future pipeline runs.

Command-line arguments override the configured values for one standalone run:
python3 ALTRUISM/ANALYSIS/3_purify.py --threshold 0.75
python3 ALTRUISM/ANALYSIS/3_purify.py --threshold-like 0.75 --threshold-mentism 0.6 --threshold-interaction 0.8
python3 ALTRUISM/ANALYSIS/3_purify.py --cooks-threshold 0.10

By default, DFBETAS is diagnostic only and does not affect 3_purified.csv.
To make DFBETAS contribute to 3_purified.csv exclusions, set:
DFBETAS_CONTRIBUTES_TO_EXCLUSIONS = True

Cook's Distance is also reported as a comparison diagnostic only.
"""

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "ANALYSIS"
INFILE = HERE / "2_simplified.csv"
OUTFILE = HERE / "3_purified.csv"
SENSITIVITY_OUTFILE = HERE / "3_dfbetas_sensitivity.csv"
SENSITIVITY_EXCLUDED_OUTFILE = HERE / "3_dfbetas_sensitivity_excluded.csv"
SENSITIVITY_PLOT_OUTFILE = HERE / "3_dfbetas_sensitivity_influence.png"
OUTDIR = HERE / "3_purify"

OUTCOME = "captcha_post_completions"
LIKE_COL = "q_post_specific_likeability"
MENT_COL = "q_post_specific_mentism"
PREDICTORS = ["like_c", "mentism_c", "like_x_mentism"]
PREDICTOR_LABELS = {
    "like_c": "Liking",
    "mentism_c": "Robomentism",
    "like_x_mentism": "Liking x Robomentism",
}
COOK_TERMS = ["Intercept"] + PREDICTORS

# Fixed thresholds used when 3_purify.py runs in the automated pipeline.
# Command-line arguments can still override these for a standalone run.
DFBETAS_THRESHOLDS = {
    "like_c": 1.0,
    "mentism_c": 1.0,
    "like_x_mentism": 1.0,
}
COOKS_DISTANCE_THRESHOLD = 1.0

# Put participant_number values here for custom/manual exclusions due to
# individual session, registration, or data-quality problems.
#
# Example:
# CUSTOM_EXCLUDED_PARTICIPANTS = [76, 101]
CUSTOM_EXCLUDED_PARTICIPANTS = [34, 98]

# Keep this False when DFBETAS is being used as a diagnostic/sensitivity
# analysis rather than as an exclusion rule.
DFBETAS_CONTRIBUTES_TO_EXCLUSIONS = False


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
    dat["mentism_c"] = dat[MENT_COL] - ment_mean
    dat["like_x_mentism"] = dat["like_c"] * dat["mentism_c"]
    dat["participant_number"] = dat["participant_number"].astype(int)
    return dat, like_mean, ment_mean


def _configured_thresholds():
    return {predictor: float(DFBETAS_THRESHOLDS[predictor]) for predictor in PREDICTORS}


def _thresholds_from_args(args):
    configured = _configured_thresholds()
    if args.threshold is not None:
        configured = {predictor: float(args.threshold) for predictor in PREDICTORS}
    return {
        "like_c": args.threshold_like if args.threshold_like is not None else configured["like_c"],
        "mentism_c": args.threshold_mentism if args.threshold_mentism is not None else configured["mentism_c"],
        "like_x_mentism": args.threshold_interaction if args.threshold_interaction is not None else configured["like_x_mentism"],
    }


def _default_cooks_threshold(n_obs):
    return float(COOKS_DISTANCE_THRESHOLD)


def _plot_predictor(influence_df, predictor, threshold, out_path):
    label = PREDICTOR_LABELS[predictor]
    d = influence_df.sort_values("participant_number").copy()
    colors = np.where(d["dfbetas_exclude_flag"], "#b24a2a", "#2f5d8a")

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
    plt.xlabel("Participant Number")
    plt.ylabel("Leave-one-out DFBETAS")
    plt.title("{} Onfluence on Primary NB beta".format(label))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_cooks_distance(influence_df, threshold, out_path):
    d = influence_df.sort_values("participant_number").copy()
    colors = np.where(d["cooks_crosses_threshold"], "#6f2d8f", "#2f5d8a")

    plt.figure(figsize=(10, 4.8))
    plt.axhline(threshold, color="#6f2d8f", linestyle="--", linewidth=1.0)
    plt.scatter(
        d["participant_number"],
        d["cooks_distance"],
        c=colors,
        s=36,
        edgecolors="white",
        linewidths=0.6,
    )
    plt.xlabel("Participant number")
    plt.ylabel("Leave-one-out Cook-style distance")
    plt.title("Primary NB Cook-style influence")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_dfbetas_vs_cooks(influence_df, cooks_threshold, out_path):
    d = influence_df.copy()
    x = d["max_abs_dfbetas"]
    y = d["cooks_distance"]
    colors = np.where(
        d["dfbetas_exclude_flag"] & d["cooks_crosses_threshold"],
        "#7a1f1f",
        np.where(d["dfbetas_exclude_flag"], "#b24a2a", np.where(d["cooks_crosses_threshold"], "#6f2d8f", "#2f5d8a")),
    )

    plt.figure(figsize=(7, 5.2))
    plt.axvline(d.attrs.get("dfbetas_reference_threshold", np.nan), color="#b24a2a", linestyle="--", linewidth=1.0)
    plt.axhline(cooks_threshold, color="#6f2d8f", linestyle="--", linewidth=1.0)
    plt.scatter(x, y, c=colors, s=42, edgecolors="white", linewidths=0.6)
    for _, row in d.loc[d["dfbetas_exclude_flag"] | d["cooks_crosses_threshold"]].iterrows():
        plt.annotate(
            str(int(row["participant_number"])),
            (row["max_abs_dfbetas"], row["cooks_distance"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    plt.xlabel("Maximum Absolute DFBETAS across Primary Predictors")
    plt.ylabel("Cook-style Distance")
    plt.title("DFBETAS vs Cook-style Influence")
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
    plt.xlabel("Participant Number")
    plt.ylabel("Absolute Leave-one-out DFBETAS")
    plt.title("Primary Negative Binomial Leave-one-out Influence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_sensitivity_overview(influence_df, thresholds, out_path):
    """Plot each participant's largest absolute DFBETAS across primary terms."""
    d = influence_df.sort_values("participant_number").copy()
    value_columns = ["dfbetas_{}".format(predictor) for predictor in PREDICTORS]
    d["max_abs_dfbetas"] = d[value_columns].abs().max(axis=1)
    flagged = d["dfbetas_exclude_flag"].astype(bool)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(
        d["participant_number"].to_numpy(),
        d["max_abs_dfbetas"].to_numpy(),
        color=np.where(flagged.to_numpy(), "#b24a2a", "#9aafc2"),
        width=0.72,
        alpha=0.9,
        zorder=2,
    )
    ax.scatter(
        d.loc[~flagged, "participant_number"],
        d.loc[~flagged, "max_abs_dfbetas"],
        color="#2f5d8a",
        s=25,
        label="Below cutoff",
        zorder=3,
    )
    ax.scatter(
        d.loc[flagged, "participant_number"],
        d.loc[flagged, "max_abs_dfbetas"],
        color="#b24a2a",
        edgecolors="white",
        linewidths=0.7,
        s=65,
        label="Sensitivity exclusion",
        zorder=4,
    )

    threshold_values = np.asarray([float(thresholds[p]) for p in PREDICTORS])
    if np.allclose(threshold_values, threshold_values[0]):
        ax.axhline(
            threshold_values[0],
            color="#8a1f1f",
            linestyle="--",
            linewidth=1.2,
            label="DFBETAS cutoff ({:.2f})".format(threshold_values[0]),
        )

    for _, row in d.loc[flagged].iterrows():
        ax.annotate(
            "P{}".format(int(row["participant_number"])),
            (row["participant_number"], row["max_abs_dfbetas"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#7a1f1f",
        )

    ax.set_xlabel("Participant excluded")
    ax.set_ylabel("Maximum absolute DFBETAS")
    ax.set_title("DFBETAS sensitivity: influence of excluding each participant")
    ax.set_ylim(0, d["max_abs_dfbetas"].max() * 1.13)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _cook_distance_from_refit(full_fit, loo_params):
    available_terms = [term for term in COOK_TERMS if term in full_fit.params.index and term in loo_params.index]
    if not available_terms:
        return np.nan

    full_params = full_fit.params.reindex(available_terms)
    loo_params = loo_params.reindex(available_terms)
    delta = (loo_params - full_params).to_numpy(dtype=float)

    cov = full_fit.cov_params().reindex(index=available_terms, columns=available_terms).to_numpy(dtype=float)
    cov_inv = np.linalg.pinv(cov)
    return float(delta.T.dot(cov_inv).dot(delta) / float(len(available_terms)))


def run_purify(
    infile=INFILE,
    outfile=OUTFILE,
    outdir=OUTDIR,
    thresholds=None,
    cooks_threshold=None,
    sensitivity_outfile=SENSITIVITY_OUTFILE,
    sensitivity_excluded_outfile=SENSITIVITY_EXCLUDED_OUTFILE,
    sensitivity_plot_outfile=SENSITIVITY_PLOT_OUTFILE,
):
    infile = Path(infile)
    outfile = Path(outfile)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    df["participant_number"] = pd.to_numeric(df["participant_number"], errors="coerce")
    custom_excluded_participants = {int(p) for p in CUSTOM_EXCLUDED_PARTICIPANTS}
    custom_mask = df["participant_number"].isin(custom_excluded_participants)
    custom_excluded = df.loc[custom_mask, ["exp_sid", "participant_number"]].copy()
    custom_excluded["participant_number"] = custom_excluded["participant_number"].astype(int)
    custom_excluded["dfbetas_exclude_flag"] = False
    custom_excluded["custom_exclude_flag"] = True
    custom_excluded["exclusion_reason"] = "custom"
    analysis_df = df.loc[~custom_mask].copy()

    dat, like_mean, ment_mean = _prepare_model_data(analysis_df)
    formula = "{} ~ like_c + mentism_c + like_x_mentism".format(OUTCOME)
    full_fit = _fit_nb(formula, dat)

    if thresholds is None:
        thresholds = _configured_thresholds()
    if cooks_threshold is None:
        cooks_threshold = _default_cooks_threshold(len(dat))

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
            cooks_distance = _cook_distance_from_refit(full_fit, loo_fit.params)
            out = {
                "exp_sid": exp_sid,
                "participant_number": participant_number,
                "n_full": int(full_fit.nobs),
                "n_leave_one_out": int(loo_fit.nobs),
                "cooks_distance": cooks_distance,
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
    influence_df["dfbetas_exclude_flag"] = influence_df[cross_cols].any(axis=1)
    influence_df["custom_exclude_flag"] = False
    influence_df["dfbetas_used_for_exclusion"] = bool(DFBETAS_CONTRIBUTES_TO_EXCLUSIONS)
    influence_df["exclude_flag"] = (
        influence_df["dfbetas_exclude_flag"] if DFBETAS_CONTRIBUTES_TO_EXCLUSIONS else False
    )
    influence_df["exclusion_reason"] = np.select(
        [
            influence_df["exclude_flag"] & influence_df["dfbetas_exclude_flag"],
        ],
        ["dfbetas"],
        default="",
    )
    influence_df["max_abs_dfbetas"] = influence_df[["dfbetas_{}".format(p) for p in PREDICTORS]].abs().max(axis=1)
    influence_df["cooks_threshold"] = float(cooks_threshold)
    influence_df["cooks_crosses_threshold"] = influence_df["cooks_distance"] > float(cooks_threshold)
    influence_df.attrs["dfbetas_reference_threshold"] = max(float(v) for v in thresholds.values())

    excluded = influence_df.loc[
        influence_df["exclude_flag"],
        ["exp_sid", "participant_number", "dfbetas_exclude_flag", "custom_exclude_flag", "exclusion_reason"],
    ].copy()
    excluded = pd.concat([custom_excluded, excluded], ignore_index=True)
    dfbetas_excluded = influence_df.loc[
        influence_df["dfbetas_exclude_flag"],
        ["exp_sid", "participant_number", "dfbetas_exclude_flag", "custom_exclude_flag", "exclusion_reason"],
    ].copy()
    dfbetas_excluded["exclusion_reason"] = "dfbetas_sensitivity"
    cooks_targeted = influence_df.loc[
        influence_df["cooks_crosses_threshold"],
        ["exp_sid", "participant_number", "cooks_distance", "max_abs_dfbetas", "exclude_flag"],
    ].copy()
    comparison_cols = [
        "exp_sid",
        "participant_number",
        "exclude_flag",
        "dfbetas_exclude_flag",
        "custom_exclude_flag",
        "dfbetas_used_for_exclusion",
        "cooks_crosses_threshold",
        "max_abs_dfbetas",
        "cooks_distance",
    ]
    comparison_df = influence_df[comparison_cols].copy()
    comparison_df["targeted_by"] = np.select(
        [
            comparison_df["dfbetas_exclude_flag"] & comparison_df["cooks_crosses_threshold"],
            comparison_df["dfbetas_exclude_flag"] & ~comparison_df["cooks_crosses_threshold"],
            ~comparison_df["dfbetas_exclude_flag"] & comparison_df["cooks_crosses_threshold"],
        ],
        ["both", "dfbetas_only", "cooks_only"],
        default="neither",
    )
    purified = df.loc[~df["exp_sid"].isin(set(excluded["exp_sid"]))].copy()
    purified.to_csv(outfile, index=False)
    dfbetas_sensitivity = analysis_df.loc[
        ~analysis_df["exp_sid"].isin(set(dfbetas_excluded["exp_sid"]))
    ].copy()
    sensitivity_outfile = Path(sensitivity_outfile)
    sensitivity_excluded_outfile = Path(sensitivity_excluded_outfile)
    sensitivity_plot_outfile = Path(sensitivity_plot_outfile)
    dfbetas_sensitivity.to_csv(sensitivity_outfile, index=False)
    dfbetas_excluded.to_csv(sensitivity_excluded_outfile, index=False)
    _plot_sensitivity_overview(influence_df, thresholds, sensitivity_plot_outfile)

    stamp = _timestamp()
    influence_csv = outdir / "3_purify_leave_one_out_dfbetas_{}.csv".format(stamp)
    cooks_csv = outdir / "3_purify_cooks_distance_{}.csv".format(stamp)
    comparison_csv = outdir / "3_purify_dfbetas_vs_cooks_{}.csv".format(stamp)
    excluded_csv = outdir / "3_purify_excluded_participants_{}.csv".format(stamp)
    summary_txt = outdir / "3_purify_summary_{}.txt".format(stamp)
    influence_df.to_csv(influence_csv, index=False)
    cooks_targeted.to_csv(cooks_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    excluded.to_csv(excluded_csv, index=False)

    plot_paths = []
    for predictor in PREDICTORS:
        plot_path = outdir / "3_purify_dfbetas_{}_{}.png".format(predictor, stamp)
        _plot_predictor(influence_df, predictor, thresholds[predictor], plot_path)
        plot_paths.append(plot_path)
    combined_plot = outdir / "3_purify_dfbetas_combined_{}.png".format(stamp)
    _plot_combined(influence_df, thresholds, combined_plot)
    plot_paths.append(combined_plot)
    cooks_plot = outdir / "3_purify_cooks_distance_{}.png".format(stamp)
    _plot_cooks_distance(influence_df, cooks_threshold, cooks_plot)
    plot_paths.append(cooks_plot)
    comparison_plot = outdir / "3_purify_dfbetas_vs_cooks_{}.png".format(stamp)
    _plot_dfbetas_vs_cooks(influence_df, cooks_threshold, comparison_plot)
    plot_paths.append(comparison_plot)

    errors_csv = None
    if errors:
        errors_csv = outdir / "3_purify_fit_errors_{}.csv".format(stamp)
        pd.DataFrame(errors).to_csv(errors_csv, index=False)

    with open(summary_txt, "w") as f:
        f.write("DFBETAS / leave-one-out purification\n")
        f.write("input: {}\n".format(infile))
        f.write("output: {}\n".format(outfile))
        f.write("rows_in_input: {}\n".format(len(df)))
        f.write("rows_after_custom_exclusions: {}\n".format(len(analysis_df)))
        f.write("rows_used_for_dfbetas_model: {}\n".format(len(dat)))
        f.write("rows_in_purified_output: {}\n".format(len(purified)))
        f.write("dfbetas_sensitivity_output: {}\n".format(sensitivity_outfile))
        f.write("rows_in_dfbetas_sensitivity_output: {}\n".format(len(dfbetas_sensitivity)))
        f.write("formula: {}\n".format(formula))
        f.write("centering_likeability_mean: {}\n".format(like_mean))
        f.write("centering_mentism_mean: {}\n".format(ment_mean))
        f.write("thresholds: {}\n".format({PREDICTOR_LABELS[k]: float(v) for k, v in thresholds.items()}))
        f.write("dfbetas_contributes_to_exclusions: {}\n".format(bool(DFBETAS_CONTRIBUTES_TO_EXCLUSIONS)))
        f.write("cooks_distance_threshold: {}\n".format(float(cooks_threshold)))
        f.write("cooks_distance_terms: {}\n".format(", ".join(COOK_TERMS)))
        f.write("custom_excluded_participants: {}\n".format(sorted(custom_excluded_participants)))
        f.write("excluded_participant_count: {}\n\n".format(len(excluded)))
        f.write("dfbetas_sensitivity_excluded_participant_count: {}\n".format(len(dfbetas_excluded)))
        if dfbetas_excluded.empty:
            f.write("DFBETAS sensitivity excluded participants: None\n\n")
        else:
            f.write("DFBETAS sensitivity excluded participants\n")
            f.write(dfbetas_excluded.to_string(index=False))
            f.write("\n\n")
        f.write("Full model coefficients\n")
        f.write(full_fit.params.to_string())
        f.write("\n\nExcluded participants\n")
        if excluded.empty:
            f.write("None\n")
        else:
            f.write(excluded.to_string(index=False))
            f.write("\n")
        f.write("\nCook's Distance comparison\n")
        f.write("Cook's Distance is diagnostic only here; it does not affect 3_purified.csv.\n")
        f.write("Cook-targeted participant count: {}\n".format(len(cooks_targeted)))
        if cooks_targeted.empty:
            f.write("Cook-targeted participants: None\n")
        else:
            f.write("Cook-targeted participants\n")
            f.write(cooks_targeted.to_string(index=False))
            f.write("\n")
        f.write("\nDFBETAS vs Cook targeting counts\n")
        f.write(comparison_df["targeted_by"].value_counts().reindex(["both", "dfbetas_only", "cooks_only", "neither"], fill_value=0).to_string())
        f.write("\n")
        if errors:
            f.write("\nLeave-one-out fit errors\n")
            f.write(pd.DataFrame(errors).to_string(index=False))
            f.write("\n")

    print("Input rows: {}".format(len(df)))
    print("Rows after custom exclusions: {}".format(len(analysis_df)))
    print("Rows used in DFBETAS model: {}".format(len(dat)))
    print("DFBETAS contributes to exclusions: {}".format(bool(DFBETAS_CONTRIBUTES_TO_EXCLUSIONS)))
    print("Excluded participants: {}".format(len(excluded)))
    if not excluded.empty:
        print(excluded.to_string(index=False))
    print("Cook-targeted participants (diagnostic only): {}".format(len(cooks_targeted)))
    if not cooks_targeted.empty:
        print(cooks_targeted.to_string(index=False))
    print("DFBETAS vs Cook targeting counts:")
    print(comparison_df["targeted_by"].value_counts().reindex(["both", "dfbetas_only", "cooks_only", "neither"], fill_value=0).to_string())
    print("Output rows: {}".format(len(purified)))
    print("Wrote {}".format(outfile))
    print("DFBETAS sensitivity rows: {}".format(len(dfbetas_sensitivity)))
    print("Wrote {}".format(sensitivity_outfile))
    print("Wrote {}".format(sensitivity_excluded_outfile))
    print("Wrote {}".format(sensitivity_plot_outfile))
    print("Diagnostics:")
    print("- {}".format(influence_csv))
    print("- {}".format(cooks_csv))
    print("- {}".format(comparison_csv))
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
        "cooks_targeted": cooks_targeted,
        "comparison": comparison_df,
        "excluded": excluded,
        "dfbetas_excluded": dfbetas_excluded,
        "purified": purified,
        "dfbetas_sensitivity": dfbetas_sensitivity,
        "thresholds": thresholds,
        "influence_csv": influence_csv,
        "cooks_csv": cooks_csv,
        "comparison_csv": comparison_csv,
        "excluded_csv": excluded_csv,
        "sensitivity_outfile": sensitivity_outfile,
        "sensitivity_excluded_outfile": sensitivity_excluded_outfile,
        "sensitivity_plot_outfile": sensitivity_plot_outfile,
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
    parser.add_argument("--sensitivity-output", default=str(SENSITIVITY_OUTFILE))
    parser.add_argument("--sensitivity-excluded-output", default=str(SENSITIVITY_EXCLUDED_OUTFILE))
    parser.add_argument("--sensitivity-plot-output", default=str(SENSITIVITY_PLOT_OUTFILE))
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the configured absolute DFBETAS cutoff for all predictors.",
    )
    parser.add_argument("--threshold-like", type=float, default=None)
    parser.add_argument(
        "--threshold-mentism", "--threshold-ment",
        dest="threshold_mentism", type=float, default=None,
        help="Absolute DFBETAS cutoff for mentism (legacy alias: --threshold-ment).",
    )
    parser.add_argument("--threshold-interaction", type=float, default=None)
    parser.add_argument(
        "--cooks-threshold",
        type=float,
        default=None,
        help="Override the configured Cook-style distance cutoff. Does not affect exclusions.",
    )
    args = parser.parse_args()

    thresholds = _thresholds_from_args(args)

    run_purify(
        args.input,
        args.output,
        args.outdir,
        thresholds=thresholds,
        cooks_threshold=args.cooks_threshold,
        sensitivity_outfile=args.sensitivity_output,
        sensitivity_excluded_outfile=args.sensitivity_excluded_output,
        sensitivity_plot_outfile=args.sensitivity_plot_output,
    )


if __name__ == "__main__":
    main()
