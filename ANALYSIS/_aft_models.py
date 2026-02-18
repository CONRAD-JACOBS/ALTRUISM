#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

from lifelines import LogNormalAFTFitter

ROOT = Path("/home/linuxlite/MEGA/MEGAsync/mega_JOBS___partial/uq/0--PROJECT/EXPERIMENTS/ALTRUISM")
INFILE = ROOT / "ANALYSIS" / "simulated_simplified.csv"   # <- change if needed
OUTDIR = ROOT / "ANALYSIS" / "aft_models"
OUTDIR.mkdir(parents=True, exist_ok=True)

DURATION_COL = "captcha_post_total_time"
EVENT_COL = "event_observed"  # we’ll create this as 1 for all rows


def main():
    df = pd.read_csv(INFILE, dtype=str)

    # ---- coerce numeric columns ----
    num_cols = [
        "participant_number", "age",
        "captcha_pre_attempts", "captcha_pre_completions", "captcha_pre_goal",
        "captcha_pre_total_time", "captcha_pre_mean_rt",
        "captcha_post_attempts", "captcha_post_completions", "captcha_post_goal",
        "captcha_post_total_time", "captcha_post_mean_rt",
        "q_pre_total_time", "q_pre_fun", "q_pre_difficulty",
        "q_pre_idaq_total_time", "q_pre_idaq",
        "q_post_gators_total_time", "q_post_gators_pos", "q_post_gators_neg",
        "q_post_specific_total_time", "q_post_specific_mentacy_belief_scale",
        "q_post_specific_empathy",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- minimal cleaning ----
    # Duration must be positive for AFT. If you can ever get 0, decide whether to:
    #   (a) treat as missing, or (b) add a tiny epsilon.
    df[DURATION_COL] = pd.to_numeric(df[DURATION_COL], errors="coerce")
    df = df[df[DURATION_COL].notna()].copy()
    df = df[df[DURATION_COL] > 0].copy()

    # everyone is "observed" (no censoring) in your current setup
    df[EVENT_COL] = 1

    # ---- define constructs ----
    # empathy in your model = q_post_specific_empathy (1-7 composite mean)
    df["empathy"] = df["q_post_specific_empathy"]

    # mentacy in your model = q_post_specific_mentacy_belief_scale (-6..6)
    df["mentacy"] = df["q_post_specific_mentacy_belief_scale"]

    # interaction
    df["emp_x_ment"] = df["empathy"] * df["mentacy"]

    # “negatively signed” difficulty: higher values -> easier/less difficult
    df["difficulty_signed"] = -df["q_pre_difficulty"]

    # “negatively signed” GATORS_neg: higher values -> less negative attitudes
    df["gators_neg_signed"] = -df["q_post_gators_neg"]

    # anthropomorphization: your IDAQ composite (q_pre_idaq)
    df["anthro"] = df["q_pre_idaq"]

    # ---- model specs ----
    # Note: AFT models in lifelines automatically model log(duration) on RHS,
    # so you pass duration_col, not log(duration).
    models = [
        ("M1_primary",
         [EVENT_COL, DURATION_COL, "empathy", "mentacy", "emp_x_ment"]),
        ("M2_add_task_appraisal",
         [EVENT_COL, DURATION_COL, "empathy", "mentacy", "emp_x_ment", "q_pre_fun", "difficulty_signed"]),
        ("M3_exploratory_traits",
         [EVENT_COL, DURATION_COL, "empathy", "mentacy", "emp_x_ment", "anthro", "q_post_gators_pos", "gators_neg_signed"]),
    ]

    results = []

    for name, cols in models:
        d = df[cols].dropna().copy()

        aft = LogNormalAFTFitter()
        aft.fit(
            d,
            duration_col=DURATION_COL,
            event_col=EVENT_COL,
        )

        # Save summary
        summary_path = OUTDIR / f"{name}_summary.csv"
        aft.summary.to_csv(summary_path, index=True)

        # quick metrics
        results.append({
            "model": name,
            "n": len(d),
            "log_likelihood": float(aft.log_likelihood_),
            "AIC": float(aft.AIC_),
        })

        # print key bits
        print("\n" + "=" * 80)
        print(name)
        print(f"n={len(d)}  logLik={aft.log_likelihood_:.3f}  AIC={aft.AIC_:.3f}")
        print(aft.summary[["coef", "se(coef)", "p"]])

    # save model comparison table
    res_df = pd.DataFrame(results).sort_values("AIC")
    res_df.to_csv(OUTDIR / "model_comparison.csv", index=False)
    print("\nSaved:", OUTDIR / "model_comparison.csv")


if __name__ == "__main__":
    main()
