#!/usr/bin/python3
"""Create zero-order Pearson correlations and a Word-ready lower-triangle table."""

import html
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


# Edit the right-hand value to change how a factor appears in the table.
# The left-hand value must match the column heading in 3_purified.csv.
# List order determines row/column order in the correlation table.
VARIABLES = [
    ("q_post_specific_mentism", "Robomentism"),
    ("q_post_specific_likeability", "Likeability"),
    ("q_pre_idaq", "IDAQ"),
    ("q_post_gators_pos", "GAToRS positive"),
    ("q_post_gators_neg", "GAToRS negative"),
    ("q_pre_captcha_fun", "reCATPCHA Fun"),
    ("q_pre_captcha_difficulty", "reCAPTCHA Difficulty"),
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
    if n < 3 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
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


def _stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_r(r, p):
    """Format an APA-style correlation (two decimals, no leading zero)."""
    if not np.isfinite(r):
        return ""
    if round(r, 2) == 0:
        return ".00" + _stars(p)
    value = "{:.2f}".format(r)
    if value.startswith("0"):
        value = value[1:]
    elif value.startswith("-0"):
        value = "-" + value[2:]
    return value + _stars(p)


def _make_lower_triangle(labels, r_matrix, p_matrix):
    table = pd.DataFrame("", index=range(len(labels)), columns=[str(i + 1) for i in range(len(labels))])
    table.insert(0, "Variable", ["{}. {}".format(i + 1, label) for i, label in enumerate(labels)])
    for row in range(len(labels)):
        for col in range(len(labels)):
            if row == col:
                table.iat[row, col + 1] = "\u2014"
            elif row > col:
                table.iat[row, col + 1] = _format_r(
                    r_matrix.iat[row, col], p_matrix.iat[row, col]
                )
    return table


def _write_word_ready_html(table, path, n_values):
    """Write a styled table that opens in Word and pastes with formatting intact."""
    n_values = sorted(set(int(n) for n in n_values if pd.notna(n)))
    if not n_values:
        n_text = "N unavailable"
    elif len(n_values) == 1:
        n_text = "N = {}".format(n_values[0])
    else:
        n_text = "Pairwise N = {}\u2013{}".format(n_values[0], n_values[-1])

    header_cells = "".join("<th>{}</th>".format(html.escape(str(col))) for col in table.columns)
    body_rows = []
    for _, row in table.iterrows():
        cells = "".join(
            "<td>{}</td>".format(html.escape(str(value)).replace("\u2014", "&mdash;"))
            for value in row
        )
        body_rows.append("<tr>{}</tr>".format(cells))

    document = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Zero-order correlations</title>
<style>
  body {{ font-family: "Times New Roman", serif; font-size: 12pt; color: #000; }}
  .table-number {{ margin: 0 0 3pt 0; }}
  .table-title {{ margin: 0 0 8pt 0; font-style: italic; }}
  table {{ border-collapse: collapse; border-top: 1.5pt solid #000; border-bottom: 1.5pt solid #000; }}
  th {{ border-bottom: 0.75pt solid #000; font-weight: normal; }}
  th, td {{ padding: 4pt 7pt; text-align: center; white-space: nowrap; }}
  th:first-child, td:first-child {{ text-align: left; padding-left: 2pt; padding-right: 14pt; }}
  .note {{ margin-top: 6pt; max-width: 8in; }}
</style>
</head>
<body>
<p class="table-number">Table 1</p>
<p class="table-title">Zero-order Pearson correlations among study factors</p>
<table>
<thead><tr>{header}</tr></thead>
<tbody>{body}</tbody>
</table>
<p class="note"><em>Note.</em> {n_text}. Correlations use pairwise-complete observations.
* <em>p</em> &lt; .05. ** <em>p</em> &lt; .01. *** <em>p</em> &lt; .001 (two-tailed).</p>
</body>
</html>
""".format(header=header_cells, body="\n".join(body_rows), n_text=n_text)
    path.write_text(document, encoding="utf-8")


def run_correlations(infile=INFILE, outdir=OUTDIR):
    infile = Path(infile)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(infile)
    raw_columns = [raw for raw, _ in VARIABLES]
    labels = [label for _, label in VARIABLES]
    missing = [col for col in raw_columns if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))
    if len(set(raw_columns)) != len(raw_columns) or len(set(labels)) != len(labels):
        raise ValueError("VARIABLES must contain unique raw columns and unique display headings")

    dat = pd.DataFrame({
        label: pd.to_numeric(df[raw], errors="coerce")
        for raw, label in VARIABLES
    })

    r_matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels)
    p_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    n_matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    rows = []
    for x_index, (x_raw, x_label) in enumerate(VARIABLES):
        n_matrix.iat[x_index, x_index] = int(dat[x_label].notna().sum())
        for y_index in range(x_index):
            y_raw, y_label = VARIABLES[y_index]
            n, r, p, ci_low, ci_high = _pearson(dat[x_label], dat[y_label])
            r_matrix.iat[x_index, y_index] = r_matrix.iat[y_index, x_index] = r
            p_matrix.iat[x_index, y_index] = p_matrix.iat[y_index, x_index] = p
            n_matrix.iat[x_index, y_index] = n_matrix.iat[y_index, x_index] = n
            rows.append({
                "x": x_label,
                "y": y_label,
                "x_column": x_raw,
                "y_column": y_raw,
                "method": "pearson",
                "n_pairwise": n,
                "r": r,
                "p_value": p,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })

    pairwise = pd.DataFrame(rows)
    lower_table = _make_lower_triangle(labels, r_matrix, p_matrix)

    pairwise_path = outdir / "zero_order_correlations.csv"
    matrix_path = outdir / "zero_order_correlation_matrix.csv"
    p_matrix_path = outdir / "zero_order_p_value_matrix.csv"
    n_matrix_path = outdir / "zero_order_pairwise_n_matrix.csv"
    lower_csv_path = outdir / "zero_order_correlations_lower_triangle.csv"
    lower_html_path = outdir / "zero_order_correlations_word_table.html"
    summary_path = outdir / "zero_order_correlations_summary.txt"

    pairwise.to_csv(pairwise_path, index=False)
    r_matrix.to_csv(matrix_path)
    p_matrix.to_csv(p_matrix_path)
    n_matrix.to_csv(n_matrix_path)
    lower_table.to_csv(lower_csv_path, index=False, encoding="utf-8-sig")
    _write_word_ready_html(lower_table, lower_html_path, pairwise["n_pairwise"])

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Zero-order Pearson correlations\n")
        f.write("input: {}\n".format(infile))
        f.write("rows_in_input: {}\n".format(len(df)))
        f.write("variables (raw column -> display heading):\n")
        for raw, label in VARIABLES:
            f.write("- {} -> {}\n".format(raw, label))
        f.write("\nLower-triangle publication table\n")
        f.write(lower_table.to_string(index=False))
        f.write("\n")

    print("Input rows: {}".format(len(df)))
    for path in (
        pairwise_path, matrix_path, p_matrix_path, n_matrix_path,
        lower_csv_path, lower_html_path, summary_path,
    ):
        print("Wrote {}".format(path))
    print("\nLower-triangle table:")
    print(lower_table.to_string(index=False))

    return {
        "pairwise": pairwise,
        "matrix": r_matrix,
        "p_matrix": p_matrix,
        "n_matrix": n_matrix,
        "lower_table": lower_table,
        "pairwise_path": pairwise_path,
        "matrix_path": matrix_path,
        "p_matrix_path": p_matrix_path,
        "n_matrix_path": n_matrix_path,
        "lower_csv_path": lower_csv_path,
        "lower_html_path": lower_html_path,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    run_correlations()
