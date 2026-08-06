import os
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = ROOT / "ANALYSIS" / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

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
PARTICIPANT_ID = "participant_number"
PRIMARY = [
    "q_post_specific_likeability",
    "q_post_specific_mentism",
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


def _safe_label(label):
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(label)).strip("_")


def _irr_table(result):
    # alpha is the NB2 dispersion estimate, not a regression coefficient or IRR.
    p = result.params.drop(labels=["alpha"], errors="ignore")
    ci = result.conf_int()
    ci = ci.loc[p.index]
    out = pd.DataFrame(
        {
            "coef": p,
            "IRR": np.exp(p),
            "CI_low_IRR": np.exp(ci[0]),
            "CI_high_IRR": np.exp(ci[1]),
            "p_value": result.pvalues.loc[p.index],
        }
    )
    out.index.name = "term"
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
    ment_bins = pd.cut(dat["q_post_specific_mentism"], bins=5, include_lowest=True)

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
    plt.ylabel("Mentism bins")
    plt.title("Mean completions across likeability x mentism")
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


def _nb2_deviance_residuals(observed, predicted, alpha):
    """Return signed NB2 deviance residuals, or NaNs if alpha is invalid."""
    y = np.asarray(observed, dtype=float)
    mu = np.asarray(predicted, dtype=float)
    if not np.isfinite(alpha) or alpha <= 0 or np.any(mu <= 0):
        return np.full(len(y), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_term = np.where(y == 0, 0.0, y * np.log(y / mu))
        size = 1.0 / alpha
        deviance = 2.0 * (
            y_term - (y + size) * np.log((y + size) / (mu + size))
        )
    # Tiny negative values can occur because of floating-point rounding.
    return np.sign(y - mu) * np.sqrt(np.maximum(deviance, 0.0))


def _plot_nb_diagnostics(diagnostics, out_dir, dataset_label, stamp):
    prefix = "5_nb_model_{}_diagnostics_".format(dataset_label)
    paths = []

    observed_path = os.path.join(out_dir, prefix + "observed_vs_predicted_{}.png".format(stamp))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(
        diagnostics["predicted_count"], diagnostics["observed_count"],
        s=34, alpha=0.58, color="#2f5d8a", edgecolors="white", linewidths=0.4,
    )
    upper = max(diagnostics["predicted_count"].max(), diagnostics["observed_count"].max())
    ax.plot([0, upper], [0, upper], linestyle="--", color="#555555", linewidth=1)
    ax.set_xlabel("Predicted CAPTCHA completions (expected count)")
    ax.set_ylabel("Observed CAPTCHA completions")
    ax.set_title("Observed vs predicted CAPTCHA completions")
    plt.tight_layout()
    plt.savefig(observed_path, dpi=300)
    plt.close(fig)
    paths.append(observed_path)

    pearson_path = os.path.join(out_dir, prefix + "pearson_residuals_vs_fitted_{}.png".format(stamp))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(
        diagnostics["predicted_count"], diagnostics["pearson_residual"],
        s=34, alpha=0.58, color="#2f5d8a", edgecolors="white", linewidths=0.4,
    )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.axhline(2, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.axhline(-2, color="#777777", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Predicted CAPTCHA completions (expected count)")
    ax.set_ylabel("Pearson residual")
    ax.set_title("Pearson residuals vs predicted CAPTCHA completions")
    plt.tight_layout()
    plt.savefig(pearson_path, dpi=300)
    plt.close(fig)
    paths.append(pearson_path)

    quadrant_order = [
        "lower_liking__low_mentism", "high_liking__low_mentism",
        "lower_liking__higher_mentism", "high_liking__higher_mentism",
    ]
    quadrant_path = os.path.join(out_dir, prefix + "pearson_residuals_by_quadrant_{}.png".format(stamp))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    groups = [diagnostics.loc[diagnostics["quadrant"] == q, "pearson_residual"].dropna() for q in quadrant_order]
    ax.boxplot(
        groups, tick_labels=[q.replace("__", "\n").replace("_", " ") for q in quadrant_order],
        patch_artist=True,
        boxprops={"facecolor": "#b7c7d8", "edgecolor": "#4a5968"},
        medianprops={"color": "#7a3426", "linewidth": 1.4},
        whiskerprops={"color": "#4a5968"}, capprops={"color": "#4a5968"},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.35, "markerfacecolor": "#2f5d8a"},
    )
    rng = np.random.default_rng(20260713)
    for position, values in enumerate(groups, start=1):
        ax.scatter(
            rng.normal(position, 0.055, len(values)), values,
            s=18, alpha=0.38, color="#2f5d8a", edgecolors="none",
        )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Pearson residual")
    ax.set_xlabel("Liking x mentism quadrant (centered-variable split at 0)")
    ax.set_title("Pearson residuals by liking x mentism quadrant")
    plt.tight_layout()
    plt.savefig(quadrant_path, dpi=300)
    plt.close(fig)
    paths.append(quadrant_path)
    return paths


def _write_nb_diagnostics(dat, fit, formula, out_dir, dataset_label, stamp):
    """Save participant-level diagnostics for the primary NB2 theory model."""
    predicted = np.asarray(fit.predict(dat), dtype=float)
    observed = dat[OUTCOME].to_numpy(dtype=float)
    alpha = float(fit.params.get("alpha", np.nan))
    raw = observed - predicted

    # Pearson residuals standardize raw errors by the model-implied NB2
    # variance: Var(Y|X) = mu + alpha * mu^2.
    variance = predicted + alpha * predicted ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        pearson = raw / np.sqrt(variance)

    diagnostics = pd.DataFrame({
        "participant_id": dat[PARTICIPANT_ID].to_numpy(),
        "observed_count": observed,
        "predicted_count": predicted,
        "raw_residual": raw,
        "pearson_residual": pearson,
        "deviance_residual": _nb2_deviance_residuals(observed, predicted, alpha),
        "liking": dat[PRIMARY[0]].to_numpy(),
        "mentism": dat[PRIMARY[1]].to_numpy(),
        "interaction": dat["like_x_mentism"].to_numpy(),
    })
    diagnostics["quadrant"] = np.select(
        [
            (dat["like_c"] < 0) & (dat["mentism_c"] < 0),
            (dat["like_c"] >= 0) & (dat["mentism_c"] < 0),
            (dat["like_c"] < 0) & (dat["mentism_c"] >= 0),
        ],
        [
            "lower_liking__low_mentism", "high_liking__low_mentism",
            "lower_liking__higher_mentism",
        ],
        default="high_liking__higher_mentism",
    )

    prefix = "5_nb_model_{}_diagnostics_".format(dataset_label)
    values_path = os.path.join(out_dir, prefix + "values_{}.csv".format(stamp))
    diagnostics.to_csv(values_path, index=False)

    invalid = ~np.isfinite(predicted) | (predicted < 0) | ~np.isfinite(variance) | (variance <= 0)
    problematic_path = None
    if invalid.any():
        problematic_path = os.path.join(out_dir, prefix + "problematic_predictions_{}.csv".format(stamp))
        diagnostics.loc[invalid].to_csv(problematic_path, index=False)
        print("WARNING: {} rows have invalid predictions or NB variance; saved to {}".format(invalid.sum(), problematic_path))

    valid = diagnostics.loc[~invalid].copy()
    plot_paths = []
    if valid.empty:
        print("WARNING: No valid fitted values remain, so diagnostic plots were skipped.")
    else:
        plot_paths = _plot_nb_diagnostics(valid, out_dir, dataset_label, stamp)
    top_pearson = diagnostics.assign(abs_residual=diagnostics["pearson_residual"].abs()).nlargest(10, "abs_residual")
    top_raw = diagnostics.assign(abs_residual=diagnostics["raw_residual"].abs()).nlargest(10, "abs_residual")
    concise = (
        "Primary NB diagnostics\n"
        "model formula: {}\nrows used: {}\nAIC: {:.6f}\nBIC: {:.6f}\n"
        "log likelihood: {:.6f}\nalpha estimate: {:.6f}\n"
        "mean observed count: {:.6f}\nmean predicted count: {:.6f}\n"
        "Pearson residual mean: {:.6f}\nPearson residual SD: {:.6f}\n"
        "quadrant split: 0 on mean-centered like_c and mentism_c\n"
    ).format(
        formula, len(diagnostics), fit.aic, getattr(fit, "bic", np.nan), fit.llf, alpha,
        diagnostics["observed_count"].mean(), diagnostics["predicted_count"].mean(),
        diagnostics["pearson_residual"].mean(), diagnostics["pearson_residual"].std(ddof=1),
    )
    summary_path = os.path.join(out_dir, prefix + "summary_{}.txt".format(stamp))
    with open(summary_path, "w") as f:
        f.write(concise)
        f.write("\nCoefficient table\n{}\n".format(fit.summary2().tables[1].to_string()))
        f.write("\nTop 10 participants by absolute Pearson residual\n")
        f.write(top_pearson.drop(columns="abs_residual").to_string(index=False))
        f.write("\n\nTop 10 participants by absolute raw residual\n")
        f.write(top_raw.drop(columns="abs_residual").to_string(index=False))
        f.write("\n")
    print("\n" + concise)
    print("Coefficient table:")
    print(fit.summary2().tables[1].to_string())
    print("Top 10 participants by absolute Pearson residual:")
    print(top_pearson[["participant_id", "pearson_residual", "observed_count", "predicted_count"]].to_string(index=False))
    print("\nTop 10 participants by absolute raw residual:")
    print(top_raw[["participant_id", "raw_residual", "observed_count", "predicted_count"]].to_string(index=False))
    paths = [values_path, summary_path] + plot_paths
    if problematic_path:
        paths.append(problematic_path)
    return paths


def _primary_prediction_grid(dat, fit, focal, moderator, moderator_values):
    focal_values = np.linspace(dat[focal].min(), dat[focal].max(), 80)
    rows = []
    like_mean = dat["q_post_specific_likeability"].mean()
    mentism_mean = dat["q_post_specific_mentism"].mean()

    for label, moderator_value in moderator_values:
        for focal_value in focal_values:
            like = focal_value if focal == "q_post_specific_likeability" else moderator_value
            mentism = focal_value if focal == "q_post_specific_mentism" else moderator_value
            like_c = like - like_mean
            mentism_c = mentism - mentism_mean
            rows.append({
                "focal": focal,
                "moderator": moderator,
                "moderator_level": label,
                "moderator_value": moderator_value,
                "q_post_specific_likeability": like,
                "q_post_specific_mentism": mentism,
                "like_c": like_c,
                "mentism_c": mentism_c,
                "like_x_mentism": like_c * mentism_c,
            })

    grid = pd.DataFrame(rows)
    grid["predicted_completions"] = fit.predict(grid)
    return grid


def _plot_prediction_lines(grid, x_col, x_label, title, out_path):
    plt.figure(figsize=(7.2, 4.8))
    colors = {
        "low": "#2f5d8a",
        "medium": "#555555",
        "high": "#b24a2a",
    }
    for level, d in grid.groupby("moderator_level", sort=False):
        plt.plot(
            d[x_col],
            d["predicted_completions"],
            label="{} {}".format(level.title(), grid["moderator"].iloc[0].replace("q_post_specific_", "").replace("_belief_scale", "")),
            linewidth=2,
            color=colors.get(level, None),
        )
    plt.xlabel(x_label)
    plt.ylabel("Predicted post-task completions")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_raw_interaction_scatter(dat, out_path):
    rng = np.random.default_rng(20260708)
    x = dat["q_post_specific_likeability"].to_numpy(dtype=float)
    y = dat["q_post_specific_mentism"].to_numpy(dtype=float)
    z = dat[OUTCOME].to_numpy(dtype=float)
    x_jitter = x + rng.normal(0, 0.045, size=len(dat))
    y_jitter = y + rng.normal(0, 0.10, size=len(dat))

    like_q75 = dat["q_post_specific_likeability"].quantile(0.75)
    mentism_q25 = dat["q_post_specific_mentism"].quantile(0.25)

    plt.figure(figsize=(7.2, 5.2))
    sc = plt.scatter(
        x_jitter,
        y_jitter,
        c=z,
        s=38 + np.sqrt(np.maximum(z, 0)) * 8,
        cmap="viridis",
        alpha=0.78,
        edgecolors="white",
        linewidths=0.5,
    )
    plt.axvline(like_q75, color="#b24a2a", linestyle="--", linewidth=1)
    plt.axhline(mentism_q25, color="#b24a2a", linestyle="--", linewidth=1)
    plt.fill_between(
        [like_q75, dat["q_post_specific_likeability"].max() + 0.25],
        dat["q_post_specific_mentism"].min() - 0.5,
        mentism_q25,
        color="#b24a2a",
        alpha=0.08,
    )
    plt.colorbar(sc, label="Post-task completions")
    plt.xlabel("Liking")
    plt.ylabel("Mentism")
    plt.title("Raw interaction scatter: completions across liking x mentism")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _binned_interaction_summary(dat):
    like_q75 = dat["q_post_specific_likeability"].quantile(0.75)
    mentism_q25 = dat["q_post_specific_mentism"].quantile(0.25)
    out = dat.copy()
    out["liking_zone"] = np.where(
        out["q_post_specific_likeability"] >= like_q75,
        "high_liking",
        "lower_liking",
    )
    out["mentism_zone"] = np.where(
        out["q_post_specific_mentism"] <= mentism_q25,
        "low_mentism",
        "higher_mentism",
    )
    out["quadrant"] = out["liking_zone"] + "__" + out["mentism_zone"]
    summary = (
        out.groupby("quadrant", observed=False)[OUTCOME]
        .agg(n="size", mean_completions="mean", median_completions="median", sd_completions="std", max_completions="max")
        .reset_index()
        .sort_values("mean_completions", ascending=False)
    )
    return out, summary


def _plot_quadrant_means(summary, out_path):
    d = summary.sort_values("mean_completions", ascending=True)
    colors = np.where(d["quadrant"].eq("high_liking__low_mentism"), "#b24a2a", "#2f5d8a")
    labels = [q.replace("__", "\n").replace("_", " ") for q in d["quadrant"]]

    plt.figure(figsize=(7.4, 4.8))
    plt.barh(labels, d["mean_completions"], color=colors)
    for i, (_, row) in enumerate(d.iterrows()):
        plt.text(row["mean_completions"], i, "  n={}".format(int(row["n"])), va="center", fontsize=9)
    plt.xlabel("Mean post-task completions")
    plt.title("Binned means by liking/mentism quadrant")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _write_interaction_plots(dat, fit, out_dir, dataset_label, stamp):
    plot_paths = []
    csv_paths = []

    mentism_levels = [
        ("low", dat["q_post_specific_mentism"].quantile(0.25)),
        ("medium", dat["q_post_specific_mentism"].quantile(0.50)),
        ("high", dat["q_post_specific_mentism"].quantile(0.75)),
    ]
    like_levels = [
        ("low", dat["q_post_specific_likeability"].quantile(0.25)),
        ("medium", dat["q_post_specific_likeability"].quantile(0.50)),
        ("high", dat["q_post_specific_likeability"].quantile(0.75)),
    ]

    liking_grid = _primary_prediction_grid(
        dat,
        fit,
        "q_post_specific_likeability",
        "q_post_specific_mentism",
        mentism_levels,
    )
    liking_grid_path = os.path.join(out_dir, "5_nb_model_{}_predicted_liking_by_mentism_{}.csv".format(dataset_label, stamp))
    liking_grid.to_csv(liking_grid_path, index=False)
    csv_paths.append(liking_grid_path)
    liking_plot = os.path.join(out_dir, "5_nb_model_{}_predicted_liking_by_mentism_{}.png".format(dataset_label, stamp))
    _plot_prediction_lines(
        liking_grid,
        "q_post_specific_likeability",
        "Liking",
        "Predicted completions by liking at low / medium / high mentism",
        liking_plot,
    )
    plot_paths.append(liking_plot)

    mentism_grid = _primary_prediction_grid(
        dat,
        fit,
        "q_post_specific_mentism",
        "q_post_specific_likeability",
        like_levels,
    )
    mentism_grid_path = os.path.join(out_dir, "5_nb_model_{}_predicted_mentism_by_liking_{}.csv".format(dataset_label, stamp))
    mentism_grid.to_csv(mentism_grid_path, index=False)
    csv_paths.append(mentism_grid_path)
    mentism_plot = os.path.join(out_dir, "5_nb_model_{}_predicted_mentism_by_liking_{}.png".format(dataset_label, stamp))
    _plot_prediction_lines(
        mentism_grid,
        "q_post_specific_mentism",
        "Mentism",
        "Predicted completions by mentism at low / medium / high liking",
        mentism_plot,
    )
    plot_paths.append(mentism_plot)

    scatter_plot = os.path.join(out_dir, "5_nb_model_{}_raw_liking_mentism_scatter_{}.png".format(dataset_label, stamp))
    _plot_raw_interaction_scatter(dat, scatter_plot)
    plot_paths.append(scatter_plot)

    quadrant_data, quadrant_summary = _binned_interaction_summary(dat)
    quadrant_data_path = os.path.join(out_dir, "5_nb_model_{}_quadrant_assignments_{}.csv".format(dataset_label, stamp))
    quadrant_summary_path = os.path.join(out_dir, "5_nb_model_{}_quadrant_summary_{}.csv".format(dataset_label, stamp))
    quadrant_data.to_csv(quadrant_data_path, index=False)
    quadrant_summary.to_csv(quadrant_summary_path, index=False)
    csv_paths.extend([quadrant_data_path, quadrant_summary_path])

    quadrant_plot = os.path.join(out_dir, "5_nb_model_{}_quadrant_means_{}.png".format(dataset_label, stamp))
    _plot_quadrant_means(quadrant_summary, quadrant_plot)
    plot_paths.append(quadrant_plot)

    return plot_paths, csv_paths, quadrant_summary


def run_nb_model_suite(csv_path, out_dir=None, dataset_label="primary"):
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(csv_path))
    os.makedirs(out_dir, exist_ok=True)
    dataset_label = _safe_label(dataset_label) or "primary"

    need = [OUTCOME] + PRIMARY + EXPLORATORY + TRIVIAL
    df = pd.read_csv(csv_path)

    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))

    if PARTICIPANT_ID not in df.columns:
        df[PARTICIPANT_ID] = np.arange(1, len(df) + 1)
        print("Participant ID column '{}' not found; using 1-based source row IDs.".format(PARTICIPANT_ID))
    elif df[PARTICIPANT_ID].isna().any():
        missing_id = df[PARTICIPANT_ID].isna()
        df.loc[missing_id, PARTICIPANT_ID] = "row_" + (df.index[missing_id] + 1).astype(str)
        print("Filled {} missing participant IDs with source row IDs.".format(missing_id.sum()))
    dat = df[[PARTICIPANT_ID] + need].copy()
    for c in need:
        dat[c] = pd.to_numeric(dat[c], errors="coerce")
    # Drop rows only for variables used across the fitted model suite. Participant
    # IDs are labels and therefore never determine complete-case inclusion.
    dat = dat.dropna(subset=need).copy()

    if dat.empty:
        raise ValueError("No complete rows remain after numeric coercion + dropna.")

    # Center primary terms so interaction is easier to interpret.
    dat["like_c"] = dat["q_post_specific_likeability"] - dat["q_post_specific_likeability"].mean()
    dat["mentism_c"] = dat["q_post_specific_mentism"] - dat["q_post_specific_mentism"].mean()
    dat["like_x_mentism"] = dat["like_c"] * dat["mentism_c"]

    formulas = {
        "intercept_only": "{} ~ 1".format(OUTCOME),
        "theory_primary_only": "{} ~ like_c + mentism_c + like_x_mentism".format(OUTCOME),
        "theory_plus_exploratory": (
            "{} ~ like_c + mentism_c + like_x_mentism + q_post_gators_pos + q_post_gators_neg + q_pre_idaq".format(OUTCOME)
        ),
        "trivial_only": "{} ~ q_pre_captcha_fun + q_pre_captcha_difficulty".format(OUTCOME),
        "theory_plus_exploratory_plus_trivial": (
            "{} ~ like_c + mentism_c + like_x_mentism + q_post_gators_pos + q_post_gators_neg + q_pre_idaq + "
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
                    "dataset": dataset_label,
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
                    "dataset": dataset_label,
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
    if "intercept_only" in fits and "theory_primary_only" in fits:
        d = _lr_compare(fits["intercept_only"], fits["theory_primary_only"])
        d["dataset"] = dataset_label
        d["comparison"] = "intercept_only -> theory_primary_only"
        lr_rows.append(d)
    if "theory_primary_only" in fits and "theory_plus_exploratory" in fits:
        d = _lr_compare(fits["theory_primary_only"], fits["theory_plus_exploratory"])
        d["dataset"] = dataset_label
        d["comparison"] = "theory_primary_only -> theory_plus_exploratory"
        lr_rows.append(d)
    if "theory_plus_exploratory" in fits and "theory_plus_exploratory_plus_trivial" in fits:
        d = _lr_compare(fits["theory_plus_exploratory"], fits["theory_plus_exploratory_plus_trivial"])
        d["dataset"] = dataset_label
        d["comparison"] = "theory_plus_exploratory -> theory_plus_exploratory_plus_trivial"
        lr_rows.append(d)
    lr_df = pd.DataFrame(lr_rows)

    stamp = _timestamp()
    cmp_csv = os.path.join(out_dir, "5_nb_model_{}_comparison_{}.csv".format(dataset_label, stamp))
    model_cmp.to_csv(cmp_csv, index=False)

    irr_paths = []
    for name, r in fits.items():
        t = _irr_table(r)
        t.insert(0, "dataset", dataset_label)
        p = os.path.join(out_dir, "5_nb_model_{}_irr_{}_{}.csv".format(dataset_label, name, stamp))
        t.reset_index().to_csv(p, index=False)
        irr_paths.append(p)

    plot_paths = []
    outcome_plot = os.path.join(out_dir, "5_nb_model_{}_outcome_distribution_{}.png".format(dataset_label, stamp))
    _plot_outcome_distribution(dat, outcome_plot)
    plot_paths.append(outcome_plot)

    interaction_plot = os.path.join(out_dir, "5_nb_model_{}_like_x_mentism_heatmap_{}.png".format(dataset_label, stamp))
    _plot_primary_interaction_heatmap(dat, interaction_plot)
    plot_paths.append(interaction_plot)

    if "theory_primary_only" in fits:
        irr_plot = os.path.join(out_dir, "5_nb_model_{}_primary_irr_{}.png".format(dataset_label, stamp))
        _plot_primary_irr(fits["theory_primary_only"], irr_plot)
        plot_paths.append(irr_plot)
        interaction_plot_paths, interaction_csv_paths, quadrant_summary = _write_interaction_plots(
            dat,
            fits["theory_primary_only"],
            out_dir,
            dataset_label,
            stamp,
        )
        plot_paths.extend(interaction_plot_paths)
        extra_csv_paths = interaction_csv_paths
        diagnostic_paths = _write_nb_diagnostics(
            dat,
            fits["theory_primary_only"],
            formulas["theory_primary_only"],
            out_dir,
            dataset_label,
            stamp,
        )
    else:
        quadrant_summary = pd.DataFrame()
        extra_csv_paths = []
        diagnostic_paths = []

    summary_txt = os.path.join(out_dir, "5_nb_model_{}_summary_{}.txt".format(dataset_label, stamp))
    with open(summary_txt, "w") as f:
        f.write("Negative Binomial Model Suite\n")
        f.write("dataset: {}\n".format(dataset_label))
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
        if not quadrant_summary.empty:
            f.write("Liking x mentism quadrant means\n")
            f.write(quadrant_summary.to_string(index=False))
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

    print("\nSaved for dataset '{}':".format(dataset_label))
    print("- {}".format(cmp_csv))
    print("- {}".format(summary_txt))
    for p in irr_paths:
        print("- {}".format(p))
    for p in extra_csv_paths:
        print("- {}".format(p))
    for p in plot_paths:
        print("- {}".format(p))
    for p in diagnostic_paths:
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
        "dataset_label": dataset_label,
        "data_used": dat,
        "fits": fits,
        "errors": errors,
        "model_comparison": model_cmp,
        "lr_comparisons": lr_df,
        "comparison_csv": cmp_csv,
        "summary_txt": summary_txt,
        "irr_csvs": irr_paths,
        "extra_csvs": extra_csv_paths,
        "plots": plot_paths,
        "diagnostics": diagnostic_paths,
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(here, "5_nb_models")
    default_csv = os.environ.get("ANALYSIS_INPUT_CSV", os.path.join(here, "3_purified.csv"))
    results = [run_nb_model_suite(default_csv, out_dir=output, dataset_label="primary")]

    sensitivity_csv = os.environ.get(
        "ANALYSIS_SENSITIVITY_INPUT_CSV",
        os.path.join(here, "3_dfbetas_sensitivity.csv"),
    )
    if os.path.exists(sensitivity_csv):
        results.append(
            run_nb_model_suite(
                sensitivity_csv,
                out_dir=output,
                dataset_label="dfbetas_sensitivity",
            )
        )
    else:
        print("\nNo DFBETAS sensitivity dataset found at: {}".format(sensitivity_csv))

    if len(results) > 1:
        stamp = _timestamp()
        combined_cmp = pd.concat(
            [r["model_comparison"] for r in results],
            ignore_index=True,
        )
        combined_path = os.path.join(output, "5_nb_model_combined_comparison_{}.csv".format(stamp))
        combined_cmp.to_csv(combined_path, index=False)
        print("\nCombined model comparison:")
        print(combined_cmp.to_string(index=False))
        print("- {}".format(combined_path))
