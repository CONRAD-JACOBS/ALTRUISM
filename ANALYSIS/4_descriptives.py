#!/usr/bin/python3
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MPL_DIR = ROOT / "ANALYSIS" / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INFILE = ROOT / "ANALYSIS" / "2_simplified.csv"  
#INFILE = ROOT / "ANALYSIS" / "simulated_simplified.csv"   # change if needed
OUTDIR = ROOT / "ANALYSIS" / "descriptives"

# Columns to use for the sanity checks (adjust if you rename)
COMPLETIONS_COL = "captcha_post_completions"
TIME_COL = "captcha_post_total_time"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def save_hist(series: pd.Series, title: str, xlabel: str, outpath: Path, bins=30):
    s = series.dropna()
    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_scatter_with_corr(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, outpath: Path):
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    plt.figure()
    plt.scatter(df_xy["x"], df_xy["y"], s=15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    r = np.nan
    n = len(df_xy)
    if n >= 2:
        r = float(np.corrcoef(df_xy["x"], df_xy["y"])[0, 1])

    # annotate
    txt = f"n={n}\nr={r:.3f}" if (n >= 2 and np.isfinite(r)) else f"n={n}\nr=NA"
    plt.gca().text(0.02, 0.98, txt, transform=plt.gca().transAxes,
                   verticalalignment="top", horizontalalignment="left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_missingness_heatmap(df: pd.DataFrame, outpath: Path, max_rows=200):
    """
    Simple missingness heatmap: rows x columns (1=missing, 0=present).
    To keep it readable, only plot first max_rows rows.
    """
    miss = df.isna().astype(int)

    # sort columns by missingness (most missing on right)
    miss_prop = miss.mean(axis=0).sort_values(ascending=False)
    miss = miss[miss_prop.index]

    if len(miss) > max_rows:
        miss = miss.iloc[:max_rows].copy()

    w = max(8, 0.25 * miss.shape[1])
    w = min(w, 24)  # clamp width
    plt.figure(figsize=(w, 6))

    plt.imshow(miss.values, aspect="auto")
    plt.title(f"Missingness heatmap (1=missing) | first {len(miss)} rows")
    plt.xlabel("Columns (sorted by missingness)")
    plt.ylabel("Rows")
    # show at most ~30 labels
    ncol = miss.shape[1]
    step = max(1, ncol // 30)
    ticks = np.arange(0, ncol, step)
    plt.xticks(ticks=ticks, labels=miss.columns[ticks], rotation=90, fontsize=7)

    plt.yticks([])
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    ensure_dir(OUTDIR)

    df = pd.read_csv(INFILE, dtype=str)

    # Coerce key columns for sanity checks
    if COMPLETIONS_COL in df.columns:
        df[COMPLETIONS_COL] = safe_numeric(df[COMPLETIONS_COL])
    else:
        raise RuntimeError(f"Missing required column: {COMPLETIONS_COL}")

    if TIME_COL in df.columns:
        df[TIME_COL] = safe_numeric(df[TIME_COL])
    else:
        raise RuntimeError(f"Missing required column: {TIME_COL}")

    # ---------- Sanity metrics ----------
    completions = df[COMPLETIONS_COL]
    time_sec = df[TIME_COL]

    n_total = len(df)
    n_comp_nonmissing = int(completions.notna().sum())
    n_time_nonmissing = int(time_sec.notna().sum())

    zero_completion_mask = completions.fillna(np.nan) == 0
    n_zero_completion = int(zero_completion_mask.sum())
    pct_zero_completion = (n_zero_completion / n_total * 100.0) if n_total > 0 else np.nan

    # correlation computed on rows where both are present
    both = pd.DataFrame({"c": completions, "t": time_sec}).dropna()
    corr_ct = np.nan
    if len(both) >= 2:
        corr_ct = float(np.corrcoef(both["t"], both["c"])[0, 1])

    # ---------- Descriptives table ----------
    # Basic summary for all numeric-like columns
    # (convert everything possible to numeric for summary)


    # pick the numeric rows you actually want in descriptives
    NUM_ROWS = [
        "age",
        "captcha_post_attempts",
        "captcha_post_completions",
        "captcha_post_total_time",
        "captcha_post_mean_rt",
        "q_pre_captcha_fun",
        "q_pre_captcha_difficulty",
        "q_pre_idaq",
        "q_pre_2050_mean_futurism_score",
        "q_post_gators_pos",
        "q_post_gators_neg",
        "q_post_specific_mentacy_belief_scale",
        "q_post_specific_empathy",
    ]

    numeric_df = df[NUM_ROWS].copy()
    numeric_df = numeric_df.apply(lambda s: pd.to_numeric(s, errors="coerce"))


    numeric_df = numeric_df.copy()
    numeric_df = numeric_df.apply(lambda s: pd.to_numeric(s, errors="coerce"))


    desc = numeric_df.describe(include="all").T
    # Add missingness + zero rate for completions specifically
    miss_counts = df.isna().sum()
    miss_props = miss_counts / max(1, len(df))

    missing_table = pd.DataFrame({
        "column": df.columns,
        "n_missing": [int(miss_counts[c]) for c in df.columns],
        "prop_missing": [float(miss_props[c]) for c in df.columns],
    }).sort_values("prop_missing", ascending=False)

    # A single “topline” row block as a small table
    topline = pd.DataFrame([{
        "n_rows": n_total,
        "completions_nonmissing": n_comp_nonmissing,
        "time_nonmissing": n_time_nonmissing,
        "n_zero_completion": n_zero_completion,
        "pct_zero_completion": pct_zero_completion,
        "corr(time, completions)": corr_ct,
    }])

    # Save CSV outputs
    topline.to_csv(OUTDIR / "topline_descriptives.csv", index=False)
    missing_table.to_csv(OUTDIR / "missingness_table.csv", index=False)
    desc.to_csv(OUTDIR / "numeric_describe_table.csv")

    # Also write one combined “descriptives.csv” (easy to open)
    # by concatenating topline + missingness summary (as text-like blocks)
    combined_path = OUTDIR / "descriptives.csv"
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("TOPLINE\n")
        topline.to_csv(f, index=False)
        f.write("\nMISSINGNESS_TABLE\n")
        missing_table.to_csv(f, index=False)

    # ---------- Plots ----------
    # 1) distribution of completions (with zeros)
    save_hist(
        completions,
        title=f"Distribution: {COMPLETIONS_COL}",
        xlabel=COMPLETIONS_COL,
        outpath=OUTDIR / "dist_completions.png",
        bins=30
    )

    # 2) distribution of time (including structural zeros / idle)
    save_hist(
        time_sec,
        title=f"Distribution: {TIME_COL}",
        xlabel=TIME_COL,
        outpath=OUTDIR / "dist_time.png",
        bins=30
    )

    # 3) % zero-completion (simple bar-style plot)
    plt.figure()
    plt.bar(["nonzero", "zero"], [n_total - n_zero_completion, n_zero_completion])
    plt.title(f"Zero completion count | {pct_zero_completion:.1f}% zero")
    plt.ylabel("Participants")
    plt.tight_layout()
    plt.savefig(OUTDIR / "zero_completion_count.png", dpi=150)
    plt.close()

    # 4) correlation(time, completions) scatter
    save_scatter_with_corr(
        x=time_sec,
        y=completions,
        title=f"{TIME_COL} vs {COMPLETIONS_COL}",
        xlabel=TIME_COL,
        ylabel=COMPLETIONS_COL,
        outpath=OUTDIR / "scatter_time_vs_completions.png"
    )

    # 5) missingness heatmap
    save_missingness_heatmap(df, OUTDIR / "missingness_heatmap.png", max_rows=200)

    print(f"Wrote descriptives + plots to: {OUTDIR}")


if __name__ == "__main__":
    main()
