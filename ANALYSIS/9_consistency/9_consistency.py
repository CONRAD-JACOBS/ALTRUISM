"""Psychometric diagnostics for the ALTRUISM study composites.

The purified file defines the analysis sample but contains composites only. Item
responses are recovered from 1_assembled.csv by exp_sid and stage, then their
means are checked against 3_purified.csv. No input is modified.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.stats import _qmvnt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "ANALYSIS" / "3_purified.csv"
RAW = ROOT / "ANALYSIS" / "1_assembled.csv"
OUT = Path(__file__).resolve().parent
SEED = 20260814
N_BOOT = 5000
N_PARALLEL = 1000
RNG = np.random.default_rng(SEED)
POLY_PAIR_CACHE: dict[tuple[str, str], tuple[float, str, int]] = {}

FUN = ["general_captcha_liking", "captcha_task_fun", "captcha_task_enjoyment"]
IDAQ = [f"idaq_{i}" for i in (3, 4, 7, 9, 11, 12, 13, 14, 17, 20, 21, 22, 23, 26, 29)]
S1 = [f"gators_{i}" for i in range(1, 6)]
S2 = [f"gators_{i}" for i in range(6, 11)]
S3 = [f"gators_{i}" for i in range(11, 16)]
S4 = [f"gators_{i}" for i in range(16, 21)]
MEASURES = {
    "Fun": FUN,
    "IDAQ": IDAQ,
    "GAToRS Personal Positive (S1)": S1,
    "GAToRS Personal Negative (S2)": S2,
    "GAToRS Societal Positive (S3)": S3,
    "GAToRS Societal Negative (S4)": S4,
    "GAToRS Positive": S1 + S3,
    "GAToRS Negative": S2 + S4,
}
RANGES = {"Fun": (1, 7), "IDAQ": (1, 10)}
for _m in MEASURES:
    if _m.startswith("GAToRS"):
        RANGES[_m] = (1, 7)


def alpha(x: np.ndarray) -> float:
    x = x[~np.isnan(x).any(axis=1)]
    if x.shape[0] < 3 or x.shape[1] < 2:
        return np.nan
    v = np.var(x, axis=0, ddof=1)
    total = np.var(x.sum(axis=1), ddof=1)
    return x.shape[1] / (x.shape[1] - 1) * (1 - v.sum() / total) if total > 0 else np.nan


def standardized_alpha(r: np.ndarray) -> float:
    k = len(r)
    vals = r[np.triu_indices(k, 1)]
    rb = np.nanmean(vals)
    return k * rb / (1 + (k - 1) * rb) if np.isfinite(rb) else np.nan


def nearest_corr(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    a = (a + a.T) / 2
    vals, vecs = np.linalg.eigh(np.nan_to_num(a, nan=0.0))
    vals = np.maximum(vals, 1e-6)
    b = (vecs * vals) @ vecs.T
    d = np.sqrt(np.diag(b))
    b = b / np.outer(d, d)
    np.fill_diagonal(b, 1.0)
    return b


def thresholds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cats, counts = np.unique(x, return_counts=True)
    if len(cats) < 2:
        return cats, np.array([])
    cum = np.cumsum(counts)[:-1] / counts.sum()
    return cats, stats.norm.ppf(np.clip(cum, 1e-5, 1 - 1e-5))


def polychoric_pair(x: np.ndarray, y: np.ndarray) -> tuple[float, str, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    cx, tx = thresholds(x)
    cy, ty = thresholds(y)
    if n < 10 or len(cx) < 2 or len(cy) < 2:
        return np.nan, "failed: insufficient observations/categories", n
    ix = np.searchsorted(cx, x)
    iy = np.searchsorted(cy, y)
    tab = np.zeros((len(cx), len(cy)), int)
    np.add.at(tab, (ix, iy), 1)
    bx = np.r_[-np.inf, tx, np.inf]
    by = np.r_[-np.inf, ty, np.inf]

    def nll(z):
        rho = np.tanh(float(z[0]))
        cov = [[1, rho], [rho, 1]]
        ll = 0.0
        for i, j in zip(*np.nonzero(tab)):
            lo = [bx[i], by[j]]
            hi = [bx[i + 1], by[j + 1]]
            p = _qmvnt._bvn(np.asarray(lo), np.asarray(hi), np.asarray(cov))
            ll += tab[i, j] * math.log(max(p, 1e-12))
        return -ll

    try:
        pr = stats.pearsonr(x, y).statistic
        z0 = np.arctanh(np.clip(pr, -0.8, 0.8)) if np.isfinite(pr) else 0.0
        fit = optimize.minimize(nll, [z0], method="Nelder-Mead", options={"maxiter": 250, "xatol": 1e-5})
        rho = float(np.tanh(fit.x[0]))
        if not fit.success or not np.isfinite(rho):
            return np.nan, "failed: numerical optimisation", n
        sparse = int((tab == 0).sum())
        return rho, f"ok ({sparse}/{tab.size} empty cells)", n
    except Exception as exc:
        return np.nan, f"failed: {type(exc).__name__}: {exc}", n


def polychoric_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[dict]]:
    k = frame.shape[1]
    r = np.eye(k)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            cache_key = tuple(sorted((str(frame.columns[i]), str(frame.columns[j]))))
            if cache_key not in POLY_PAIR_CACHE:
                POLY_PAIR_CACHE[cache_key] = polychoric_pair(
                    frame.iloc[:, i].to_numpy(float), frame.iloc[:, j].to_numpy(float))
            val, status, n = POLY_PAIR_CACHE[cache_key]
            r[i, j] = r[j, i] = val
            rows.append({"item1": frame.columns[i], "item2": frame.columns[j], "r": val, "n_pair": n, "status": status})
    return r, rows


def one_factor(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = nearest_corr(r)
    inv = np.linalg.pinv(r)
    communal = np.clip(1 - 1 / np.diag(inv), 0.05, 0.95)
    load = np.zeros(len(r))
    for _ in range(200):
        work = r.copy()
        np.fill_diagonal(work, communal)
        vals, vecs = np.linalg.eigh(work)
        load2 = vecs[:, -1] * math.sqrt(max(vals[-1], 0))
        if load2.sum() < 0:
            load2 *= -1
        new = np.clip(load2**2, 0.001, 0.999)
        load = load2
        if np.max(np.abs(new - communal)) < 1e-7:
            break
        communal = new
    return load, np.clip(load**2, 0, 1)


def omega_from_r(r: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    load, comm = one_factor(r)
    unique = np.clip(1 - comm, 0, 1)
    den = load.sum() ** 2 + unique.sum()
    return (load.sum() ** 2 / den if den > 0 else np.nan), load, comm


def bootstrap(frame: pd.DataFrame, fn, n_boot=N_BOOT) -> tuple[float, float, int]:
    x = frame.dropna().to_numpy(float)
    if len(x) < 3:
        return np.nan, np.nan, n_boot
    vals = []
    failed = 0
    for _ in range(n_boot):
        xb = x[RNG.integers(0, len(x), len(x))]
        try:
            v = fn(xb)
            if np.isfinite(v):
                vals.append(v)
            else:
                failed += 1
        except Exception:
            failed += 1
    if not vals:
        return np.nan, np.nan, failed
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), failed


def mean_r_array(x: np.ndarray) -> float:
    r = np.corrcoef(x, rowvar=False)
    return float(np.mean(r[np.triu_indices(len(r), 1)]))


def omega_array(x: np.ndarray) -> float:
    return omega_from_r(np.corrcoef(x, rowvar=False))[0]


def parallel_analysis(observed_r: np.ndarray, n: int) -> tuple[int, np.ndarray, np.ndarray]:
    k = len(observed_r)
    obs = np.linalg.eigvalsh(nearest_corr(observed_r))[::-1]
    null = np.empty((N_PARALLEL, k))
    for b in range(N_PARALLEL):
        null[b] = np.linalg.eigvalsh(np.corrcoef(RNG.normal(size=(n, k)), rowvar=False))[::-1]
    crit = np.percentile(null, 95, axis=0)
    return int(np.sum(obs > crit)), obs, crit


def extract_items(sample: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(RAW, low_memory=False)
    raw = raw[raw["exp_sid"].astype(str).isin(set(sample["exp_sid"].astype(str)))]
    stages = {"q_pre_captcha", "q_pre_idaq", "q_post_gators"}
    raw = raw[raw["stage_id"].isin(stages)]
    records = {sid: {} for sid in sample["exp_sid"].astype(str)}
    for _, row in raw.iterrows():
        sid = str(row["exp_sid"])
        try:
            payload = json.loads(row["questionnaire_json"])
        except Exception:
            continue
        for key, val in payload.items():
            records.setdefault(sid, {})[key] = pd.to_numeric(val, errors="coerce")
    items = pd.DataFrame.from_dict(records, orient="index")
    return items.reindex(sample["exp_sid"].astype(str)).reset_index(drop=True)


def fmt(v, digits=3):
    return "NA" if not np.isfinite(v) else f"{v:.{digits}f}"


def main():
    OUT.mkdir(exist_ok=True)
    sample = pd.read_csv(DATA)
    items = extract_items(sample)
    print("SCORING DEFINITIONS DETECTED")
    print(f"Fun: arithmetic mean of {FUN}; no reverse coding; expected 1-7")
    print(f"IDAQ: arithmetic mean of {IDAQ}; no reverse coding; expected 1-10")
    print(f"GAToRS Positive: mean of S1 personal positive {S1} and S3 societal positive {S3}; no reversal; expected 1-7")
    print(f"GAToRS Negative: mean of S2 personal negative {S2} and S4 societal negative {S4}; no reversal; expected 1-7")
    print("Because each GAToRS subscale has five items, pooling ten items equals averaging its two subscale means.")

    checks = []
    for name, cols, target in [
        ("Fun", FUN, "q_pre_captcha_fun"), ("IDAQ", IDAQ, "q_pre_idaq"),
        ("GAToRS Positive", S1 + S3, "q_post_gators_pos"),
        ("GAToRS Negative", S2 + S4, "q_post_gators_neg")]:
        reconstructed = items[cols].mean(axis=1)
        delta = reconstructed - pd.to_numeric(sample[target], errors="coerce")
        checks.append((name, int(delta.notna().sum()), float(delta.abs().max())))
        print(f"Scoring verification {name}: n compared={delta.notna().sum()}, max absolute difference={delta.abs().max():.12g}")

    desc_rows, itemdiag_rows, pearson_rows, poly_rows = [], [], [], []
    reliability_rows, loading_rows, dimension_rows = [], [], []
    poly_cache = {}
    for measure, cols in MEASURES.items():
        frame = items[cols].apply(pd.to_numeric, errors="coerce")
        lo, hi = RANGES[measure]
        for col in cols:
            s = frame[col]
            counts = {str(k): int(v) for k, v in s.value_counts(dropna=False).sort_index().items()}
            valid = s.dropna()
            floor = float((valid == lo).mean()) if len(valid) else np.nan
            ceiling = float((valid == hi).mean()) if len(valid) else np.nan
            flags = []
            if ((valid < lo) | (valid > hi)).any(): flags.append("impossible value")
            if valid.nunique() <= 1: flags.append("essentially no variance")
            if floor >= .80: flags.append("extreme floor concentration >=80%")
            if ceiling >= .80: flags.append("extreme ceiling concentration >=80%")
            desc_rows.append({"measure": measure, "item": col, "n": int(valid.count()),
                "missing_n": int(s.isna().sum()), "missing_pct": 100*s.isna().mean(), "mean": valid.mean(),
                "sd": valid.std(ddof=1), "median": valid.median(), "minimum": valid.min(), "maximum": valid.max(),
                "skewness": stats.skew(valid, bias=False, nan_policy="omit") if len(valid)>2 else np.nan,
                "floor_pct": 100*floor, "ceiling_pct": 100*ceiling,
                "frequency_distribution": json.dumps(counts), "flags": "; ".join(flags)})
        complete = frame.dropna()
        pr = frame.corr(method="pearson").to_numpy()
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                pearson_rows.append({"measure": measure, "item1": a, "item2": b, "r": pr[i,j],
                    "n_pair": int(frame[[a,b]].dropna().shape[0])})
        key = tuple(cols)
        pol, pairrows = polychoric_matrix(frame)
        poly_cache[key] = pol
        for row in pairrows:
            poly_rows.append({"measure": measure, **row})
        for i, a in enumerate(cols):
            poly_rows.append({"measure": measure, "item1": a, "item2": a, "r": 1.0,
                              "n_pair": int(frame[a].notna().sum()), "status": "diagonal"})
        ar = alpha(complete.to_numpy())
        ast = standardized_alpha(pr)
        arlo, arhi, afail = bootstrap(complete, alpha)
        mrlo, mrhi, mrfail = bootstrap(complete, mean_r_array)
        om_p, load_p, com_p = omega_from_r(pr)
        om_o = np.nan; load_o = np.full(len(cols), np.nan); com_o = load_o.copy()
        valid_poly = np.isfinite(pol).all()
        if valid_poly:
            om_o, load_o, com_o = omega_from_r(pol)
        omlo, omhi, omfail = bootstrap(complete, omega_array)
        offp = pr[np.triu_indices(len(cols), 1)]
        offpoly = pol[np.triu_indices(len(cols), 1)]
        pa_r = pol if valid_poly else pr
        nf, eig, crit = parallel_analysis(pa_r, len(complete))
        for i, item in enumerate(cols):
            others = [c for c in cols if c != item]
            citc = frame[[item] + others].dropna()
            corrected = citc[item].corr(citc[others].mean(axis=1)) if len(citc) else np.nan
            itemdiag_rows.append({"measure": measure, "item": item, "corrected_item_total_r": corrected,
                                  "alpha_if_deleted": alpha(complete[others].to_numpy())})
            loading_rows += [
                {"measure": measure, "basis": "Pearson", "item": item, "loading": load_p[i], "communality": com_p[i]},
                {"measure": measure, "basis": "Polychoric", "item": item, "loading": load_o[i], "communality": com_o[i]},]
        note = "Omega point estimate uses polychoric correlations; its bootstrap CI is Pearson-based." if valid_poly else "Polychoric omega unavailable; see pair failure statuses."
        if measure == "Fun": note += " Three indicators weakly identify dimensionality; alpha is scale-length sensitive."
        if measure == "IDAQ": note += " Validated scoring retained; diagnostics are not scale revalidation."
        if measure.startswith("GAToRS") and "(" not in measure: note += " Broad composite intentionally combines distinguishable validated subscales."
        reliability_rows.append({"measure": measure, "n_items": len(cols), "n_complete": len(complete),
            "alpha_raw": ar, "alpha_raw_CI_low": arlo, "alpha_raw_CI_high": arhi,
            "alpha_standardised": ast, "omega_total": om_o if valid_poly else om_p,
            "omega_pearson": om_p, "omega_CI_low": omlo, "omega_CI_high": omhi,
            "mean_interitem_r": np.nanmean(offp), "mean_interitem_r_CI_low": mrlo,
            "mean_interitem_r_CI_high": mrhi, "min_interitem_r": np.nanmin(offp), "max_interitem_r": np.nanmax(offp),
            "mean_polychoric_r": np.nanmean(offpoly), "min_polychoric_r": np.nanmin(offpoly),
            "max_polychoric_r": np.nanmax(offpoly), "suggested_n_factors_parallel_analysis": nf,
            "alpha_bootstrap_failed": afail, "omega_bootstrap_failed": omfail,
            "mean_r_bootstrap_failed": mrfail, "notes": note})
        dimension_rows.append({"measure": measure, "basis": "Polychoric" if valid_poly else "Pearson fallback",
            "n_complete": len(complete), "observed_eigenvalues": json.dumps(eig.tolist()),
            "parallel_95pct_eigenvalues": json.dumps(crit.tolist()), "suggested_n_factors": nf,
            "method": "one-factor principal-axis/common-factor; parallel analysis against 1000 normal null datasets"})

    # Suspicious identical pairs, assessed once across unique items.
    all_cols = list(dict.fromkeys(sum(MEASURES.values(), [])))
    identical = []
    for i, a in enumerate(all_cols):
        for b in all_cols[i+1:]:
            z = items[[a,b]].dropna()
            if len(z) and (z[a] == z[b]).all(): identical.append(f"{a}={b} (n={len(z)})")

    # GAToRS subscale-score diagnostics and two-component bootstrap.
    g_rows = []
    for broad, n1, c1, n2, c2 in [("GAToRS Positive", "Personal Positive (S1)", S1, "Societal Positive (S3)", S3),
                                   ("GAToRS Negative", "Personal Negative (S2)", S2, "Societal Negative (S4)", S4)]:
        sub = pd.DataFrame({n1: items[c1].mean(axis=1), n2: items[c2].mean(axis=1)}).dropna()
        r = sub.iloc[:,0].corr(sub.iloc[:,1], method="pearson")
        rs = sub.iloc[:,0].corr(sub.iloc[:,1], method="spearman")
        sb = 2*r/(1+r)
        sblo, sbhi, sbfail = bootstrap(sub, lambda x: 2*np.corrcoef(x,rowvar=False)[0,1]/(1+np.corrcoef(x,rowvar=False)[0,1]))
        g_rows.append({"measure": broad, "subscale_1": n1, "subscale_2": n2, "n_complete": len(sub),
            "subscale_1_mean": sub.iloc[:,0].mean(), "subscale_1_sd": sub.iloc[:,0].std(ddof=1),
            "subscale_2_mean": sub.iloc[:,1].mean(), "subscale_2_sd": sub.iloc[:,1].std(ddof=1),
            "pearson_r": r, "spearman_r": rs, "spearman_brown": sb,
            "spearman_brown_CI_low": sblo, "spearman_brown_CI_high": sbhi,
            "alpha_two_subscales_raw": alpha(sub.to_numpy()), "bootstrap_failed": sbfail})

    pd.DataFrame(reliability_rows).to_csv(OUT/"reliability_summary.csv", index=False)
    pd.DataFrame(desc_rows).to_csv(OUT/"item_descriptives.csv", index=False)
    pd.DataFrame(itemdiag_rows).to_csv(OUT/"item_total_diagnostics.csv", index=False)
    pd.DataFrame(pearson_rows).to_csv(OUT/"interitem_correlations_pearson.csv", index=False)
    pd.DataFrame(poly_rows).to_csv(OUT/"interitem_correlations_polychoric.csv", index=False)
    pd.DataFrame(loading_rows).to_csv(OUT/"factor_loadings.csv", index=False)
    pd.DataFrame(dimension_rows).to_csv(OUT/"dimensionality_summary.csv", index=False)
    pd.DataFrame(g_rows).to_csv(OUT/"gators_subscale_composite_diagnostics.csv", index=False)

    rel = pd.DataFrame(reliability_rows).set_index("measure")
    gdf = pd.DataFrame(g_rows).set_index("measure")
    ddf = pd.DataFrame(desc_rows)
    iddf = pd.DataFrame(itemdiag_rows)
    ldf = pd.DataFrame(loading_rows)
    lines = ["# Internal consistency and reliability assessment", "",
        f"Data: 3_purified.csv defines the sample (N={len(sample)}); item responses were recovered by exp_sid from 1_assembled.csv.",
        f"Random seed: {SEED}. Bootstrap samples: {N_BOOT}. Parallel-analysis samples: {N_PARALLEL}.", "",
        "## 1. Scoring verification", "",
        "Fun is the arithmetic mean of general_captcha_liking, captcha_task_fun, and captcha_task_enjoyment (1-7). IDAQ is the arithmetic mean of the 15 explicitly listed scoring items (1-10). GAToRS Positive averages Personal Positive S1 (items 1-5) and Societal Positive S3 (11-15); GAToRS Negative averages Personal Negative S2 (6-10) and Societal Negative S4 (16-20). All are unweighted item means and no reverse coding is specified or applied. Equal five-item GAToRS subscales mean the pooled item mean equals the mean of the two subscale means.", ""]
    for name,n,maxdiff in checks: lines.append(f"- {name}: {n} reconstructed scores checked; maximum absolute discrepancy {maxdiff:.12g}.")
    flags = ddf[ddf["flags"].astype(str).str.len()>0]
    lines += [f"- Impossible/variance/floor/ceiling flags: {len(flags)} item-measure rows (details in item_descriptives.csv).",
              f"- Suspicious exactly identical item pairs: {'; '.join(identical) if identical else 'none'}.", ""]

    def section(name, heading):
        row = rel.loc[name]
        return [f"## {heading}", "",
            f"Complete-case n={int(row.n_complete)}, k={int(row.n_items)}. Raw alpha={fmt(row.alpha_raw)} (bootstrap 95% CI {fmt(row.alpha_raw_CI_low)} to {fmt(row.alpha_raw_CI_high)}); standardised alpha={fmt(row.alpha_standardised)}. Omega total={fmt(row.omega_total)} (bootstrap 95% CI {fmt(row.omega_CI_low)} to {fmt(row.omega_CI_high)}; see method note). Mean Pearson inter-item r={fmt(row.mean_interitem_r)} (range {fmt(row.min_interitem_r)} to {fmt(row.max_interitem_r)}; bootstrap CI {fmt(row.mean_interitem_r_CI_low)} to {fmt(row.mean_interitem_r_CI_high)}). Mean polychoric r={fmt(row.mean_polychoric_r)} (range {fmt(row.min_polychoric_r)} to {fmt(row.max_polychoric_r)}). Parallel analysis suggested {int(row.suggested_n_factors_parallel_analysis)} factor(s).", "",
            str(row.notes), ""]

    lines += section("Fun", "2. Fun")
    pframe = pd.DataFrame(pearson_rows)
    ppframe = pd.DataFrame(poly_rows)
    order = {item: i for i, item in enumerate(FUN)}
    funpairs = pframe[(pframe.measure == "Fun") & pframe.apply(lambda z: order.get(z.item1, 99) < order.get(z.item2, -1), axis=1)]
    funpoly = ppframe[(ppframe.measure == "Fun") & ppframe.apply(lambda z: order.get(z.item1, 99) < order.get(z.item2, -1), axis=1)]
    fundesc = ddf[ddf.measure == "Fun"].set_index("item")
    fundiag = iddf[iddf.measure == "Fun"].set_index("item")
    funload = ldf[(ldf.measure == "Fun") & (ldf.basis == "Polychoric")].set_index("item")
    lines += ["The three Pearson correlations are " + "; ".join(f"{r.item1}/{r.item2}={fmt(r.r)}" for _,r in funpairs.iterrows()) + ".",
              "The three polychoric correlations are " + "; ".join(f"{r.item1}/{r.item2}={fmt(r.r)} ({r.status})" for _,r in funpoly.iterrows()) + ".", "",
              "Item-level results: " + "; ".join(
                  f"{item}: M={fmt(fundesc.loc[item,'mean'])}, SD={fmt(fundesc.loc[item,'sd'])}, median={fmt(fundesc.loc[item,'median'])}, corrected item-total r={fmt(fundiag.loc[item,'corrected_item_total_r'])}, ordinal loading={fmt(funload.loc[item,'loading'])}"
                  for item in FUN) + ".", "",
              "These intentionally different angles on enjoyment need not be redundant. Their shared variance supports consistency, while imperfect correlations can reflect construct breadth as well as error. Item-deletion results are diagnostics and are not grounds by themselves to remove an item. With only three indicators, dimensionality is weakly identified and saturated one-factor CFA fit would be uninformative.", ""]
    lines += section("IDAQ", "3. IDAQ")
    lines += ["IDAQ is treated as an established scale applied to this sample. The present exploratory diagnostics do not revalidate, redesign, shorten, or opportunistically redefine its published scoring structure.", ""]
    for broad, heading in [("GAToRS Positive", "4. GAToRS Positive"), ("GAToRS Negative", "5. GAToRS Negative")]:
        lines += section(broad, heading)
        g = gdf.loc[broad]
        lines += [f"At the constituent-subscale level, {g.subscale_1} and {g.subscale_2} correlated Pearson r={fmt(g.pearson_r)} and Spearman rho={fmt(g.spearman_r)}. Spearman-Brown={fmt(g.spearman_brown)} (bootstrap 95% CI {fmt(g.spearman_brown_CI_low)} to {fmt(g.spearman_brown_CI_high)}); raw two-component alpha={fmt(g.alpha_two_subscales_raw)}.",
                  "With two components, standardised alpha is mathematically determined by their correlation and adds little beyond that correlation and Spearman-Brown coefficient. Two subscales cannot establish a higher-order latent factor. Within-subscale consistency and between-subscale coherence answer different questions; multiple first-order dimensions do not automatically invalidate the broader composite.", ""]
    lines += ["## 6. Overall conclusions", "",
        "Cronbach's alpha is not a test of unidimensionality. It is affected by item count and average covariance, and can be distorted by violations of tau-equivalence, multidimensionality, redundancy, and correlated content/errors. High alpha can reflect redundancy or many items; low alpha can reflect short scales or deliberately heterogeneous facets.", "",
        "Omega is generally preferable when loadings differ, but it still requires a defensible latent-factor model. Polychoric estimates respect ordered responses but can be unstable with sparse categories; every failed pair is retained as missing and explicitly labelled in the polychoric CSV rather than silently replaced by Pearson. Item deletion is diagnostic, not an automatic editing rule. Reliability describes these scores in this sample, not an immutable property of a questionnaire.", "",
        "### Manuscript-oriented summary", "",
        f"- Fun: report alpha {fmt(rel.loc['Fun','alpha_raw'])}, omega {fmt(rel.loc['Fun','omega_total'])}, mean inter-item r {fmt(rel.loc['Fun','mean_interitem_r'])}, their CIs, and all three correlations; describe the intentionally broad three-item design and limited dimensionality evidence.",
        f"- IDAQ: report the established 15-item mean with alpha {fmt(rel.loc['IDAQ','alpha_raw'])} and omega {fmt(rel.loc['IDAQ','omega_total'])} in this sample, retaining the validated scoring definition.",
        f"- GAToRS Positive: report item-level alpha {fmt(rel.loc['GAToRS Positive','alpha_raw'])} and omega {fmt(rel.loc['GAToRS Positive','omega_total'])}, plus the S1-S3 Pearson correlation {fmt(gdf.loc['GAToRS Positive','pearson_r'])} and Spearman-Brown {fmt(gdf.loc['GAToRS Positive','spearman_brown'])}; acknowledge the two subdimensions.",
        f"- GAToRS Negative: report item-level alpha {fmt(rel.loc['GAToRS Negative','alpha_raw'])} and omega {fmt(rel.loc['GAToRS Negative','omega_total'])}, plus the S2-S4 Pearson correlation {fmt(gdf.loc['GAToRS Negative','pearson_r'])} and Spearman-Brown {fmt(gdf.loc['GAToRS Negative','spearman_brown'])}; acknowledge the two subdimensions.", "",
        "Full item descriptives, response frequencies, item-total diagnostics, correlation matrices, factor loadings, eigenvalues, bootstrap failures, and polychoric estimation statuses are in the accompanying machine-readable CSV files."]
    report = "\n".join(lines) + "\n"
    (OUT/"internal_consistency_report.md").write_text(report, encoding="utf-8")
    plain = report.replace("# ", "").replace("## ", "").replace("### ", "")
    (OUT/"internal_consistency_report.txt").write_text(plain, encoding="utf-8")
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
