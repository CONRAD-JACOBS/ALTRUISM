#!/usr/bin/python3
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INFILE = Path(os.environ.get("ANALYSIS_INPUT_CSV", ROOT / "ANALYSIS" / "3_purified.csv"))
OUTDIR = ROOT / "ANALYSIS" / "6_correlations"

try:
    from scipy import stats
except Exception:
    stats = None


VARIABLES = {
    "mentism": "q_post_specific_mentism",
    "GAToRS_negative": "q_post_gators_neg",
    "GAToRS_positive": "q_post_gators_pos",
    "liking": "q_post_specific_likeability",
    "empathy": "q_post_specific_robot_empathy",
}

CORRELATION_PAIRS = [
    ("mentism", "GAToRS_negative"),
    ("mentism", "GAToRS_positive"),
    ("mentism", "liking"),
    ("mentism", "empathy"),
    ("GAToRS_negative", "liking"),
    ("GAToRS_positive", "liking"),
    ("liking", "empathy"),
]


def _pearson_ci(r, n, alpha=0.05):
    if n <= 3 or not np.isfinite(r) or abs(r) >= 1:
        return np.nan, np.nan

    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    if stats is not None:
        zcrit = stats.norm.ppf(1 - alpha / 2)
    else:
        zcrit = 1.959963984540054
    return float(np.tanh(z - zcrit * se)), float(np.tanh(z + zcrit * se))


def _pearson(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(d)
    if n < 3:
        return n, np.nan, np.nan, np.nan, np.nan

    if stats is not None:
        r, p = stats.pearsonr(d["x"], d["y"])
        r = float(r)
        p = float(p)
    else:
        r = float(np.corrcoef(d["x"], d["y"])[0, 1])
        p = np.nan

    ci_low, ci_high = _pearson_ci(r, n)
    return n, r, p, ci_low, ci_high


def run_correlations(infile=INFILE, outdir=OUTDIR):
    infile = Path(infile)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    missing = [col for col in VARIABLES.values() if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))

    dat = pd.DataFrame({
        label: pd.to_numeric(df[col], errors="coerce")
        for label, col in VARIABLES.items()
    })

    rows = []
    for x_label, y_label in CORRELATION_PAIRS:
        n, r, p, ci_low, ci_high = _pearson(dat[x_label], dat[y_label])
        rows.append({
            "x": x_label,
            "y": y_label,
            "x_column": VARIABLES[x_label],
            "y_column": VARIABLES[y_label],
            "method": "pearson",
            "n_pairwise": n,
            "r": r,
            "p_value": p,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        })

    pairwise = pd.DataFrame(rows)
    matrix = dat.corr(method="pearson")
    n_matrix = dat.notna().astype(int).T.dot(dat.notna().astype(int))

    pairwise_path = outdir / "zero_order_correlations.csv"
    matrix_path = outdir / "zero_order_correlation_matrix.csv"
    n_matrix_path = outdir / "zero_order_pairwise_n_matrix.csv"
    summary_path = outdir / "zero_order_correlations_summary.txt"

    pairwise.to_csv(pairwise_path, index=False)
    matrix.to_csv(matrix_path)
    n_matrix.to_csv(n_matrix_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Zero-order Pearson correlations\n")
        f.write("input: {}\n".format(infile))
        f.write("rows_in_input: {}\n".format(len(df)))
        f.write("variables:\n")
        for label, col in VARIABLES.items():
            f.write("- {}: {}\n".format(label, col))
        f.write("\nRequested correlations\n")
        f.write(pairwise.to_string(index=False))
        f.write("\n")

    print("Input rows: {}".format(len(df)))
    print("Wrote {}".format(pairwise_path))
    print("Wrote {}".format(matrix_path))
    print("Wrote {}".format(n_matrix_path))
    print("Wrote {}".format(summary_path))
    print("\nRequested correlations:")
    print(pairwise[["x", "y", "n_pairwise", "r", "p_value", "ci95_low", "ci95_high"]].to_string(index=False))

    return {
        "pairwise": pairwise,
        "matrix": matrix,
        "n_matrix": n_matrix,
        "pairwise_path": pairwise_path,
        "matrix_path": matrix_path,
        "n_matrix_path": n_matrix_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_correlations()
