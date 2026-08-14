# Internal consistency and reliability assessment

Data: 3_purified.csv defines the sample (N=100); item responses were recovered by exp_sid from 1_assembled.csv.
Random seed: 20260814. Bootstrap samples: 5000. Parallel-analysis samples: 1000.

## 1. Scoring verification

Fun is the arithmetic mean of general_captcha_liking, captcha_task_fun, and captcha_task_enjoyment (1-7). IDAQ is the arithmetic mean of the 15 explicitly listed scoring items (1-10). GAToRS Positive averages Personal Positive S1 (items 1-5) and Societal Positive S3 (11-15); GAToRS Negative averages Personal Negative S2 (6-10) and Societal Negative S4 (16-20). All are unweighted item means and no reverse coding is specified or applied. Equal five-item GAToRS subscales mean the pooled item mean equals the mean of the two subscale means.

- Fun: 100 reconstructed scores checked; maximum absolute discrepancy 4.4408920985e-16.
- IDAQ: 100 reconstructed scores checked; maximum absolute discrepancy 8.881784197e-16.
- GAToRS Positive: 100 reconstructed scores checked; maximum absolute discrepancy 0.
- GAToRS Negative: 100 reconstructed scores checked; maximum absolute discrepancy 0.
- Impossible/variance/floor/ceiling flags: 0 item-measure rows (details in item_descriptives.csv).
- Suspicious exactly identical item pairs: none.

## 2. Fun

Complete-case n=100, k=3. Raw alpha=0.868 (bootstrap 95% CI 0.808 to 0.909); standardised alpha=0.868. Omega total=0.891 (bootstrap 95% CI 0.812 to 0.910; see method note). Mean Pearson inter-item r=0.686 (range 0.652 to 0.739; bootstrap CI 0.583 to 0.771). Mean polychoric r=0.728 (range 0.698 to 0.784). Parallel analysis suggested 1 factor(s).

Omega point estimate uses polychoric correlations; its bootstrap CI is Pearson-based. Three indicators weakly identify dimensionality; alpha is scale-length sensitive.

The three Pearson correlations are general_captcha_liking/captcha_task_fun=0.668; general_captcha_liking/captcha_task_enjoyment=0.652; captcha_task_fun/captcha_task_enjoyment=0.739.
The three polychoric correlations are general_captcha_liking/captcha_task_fun=0.703 (ok (22/49 empty cells)); general_captcha_liking/captcha_task_enjoyment=0.698 (ok (22/49 empty cells)); captcha_task_fun/captcha_task_enjoyment=0.784 (ok (23/49 empty cells)).

Item-level results: general_captcha_liking: M=3.780, SD=1.345, median=4.000, corrected item-total r=0.708, ordinal loading=0.792; captcha_task_fun: M=3.360, SD=1.494, median=3.000, corrected item-total r=0.775, ordinal loading=0.888; captcha_task_enjoyment: M=3.750, SD=1.445, median=4.000, corrected item-total r=0.764, ordinal loading=0.882.

These intentionally different angles on enjoyment need not be redundant. Their shared variance supports consistency, while imperfect correlations can reflect construct breadth as well as error. Item-deletion results are diagnostics and are not grounds by themselves to remove an item. With only three indicators, dimensionality is weakly identified and saturated one-factor CFA fit would be uninformative.

## 3. IDAQ

Complete-case n=100, k=15. Raw alpha=0.811 (bootstrap 95% CI 0.724 to 0.864); standardised alpha=0.812. Omega total=0.856 (bootstrap 95% CI 0.701 to 0.868; see method note). Mean Pearson inter-item r=0.224 (range -0.103 to 0.655; bootstrap CI 0.150 to 0.299). Mean polychoric r=0.282 (range -0.164 to 0.731). Parallel analysis suggested 2 factor(s).

Omega point estimate uses polychoric correlations; its bootstrap CI is Pearson-based. Validated scoring retained; diagnostics are not scale revalidation.

IDAQ is treated as an established scale applied to this sample. The present exploratory diagnostics do not revalidate, redesign, shorten, or opportunistically redefine its published scoring structure.

## 4. GAToRS Positive

Complete-case n=100, k=10. Raw alpha=0.691 (bootstrap 95% CI 0.574 to 0.768); standardised alpha=0.722. Omega total=0.763 (bootstrap 95% CI 0.593 to 0.799; see method note). Mean Pearson inter-item r=0.206 (range -0.242 to 0.547; bootstrap CI 0.142 to 0.274). Mean polychoric r=0.234 (range -0.249 to 0.629). Parallel analysis suggested 2 factor(s).

Omega point estimate uses polychoric correlations; its bootstrap CI is Pearson-based. Broad composite intentionally combines distinguishable validated subscales.

At the constituent-subscale level, Personal Positive (S1) and Societal Positive (S3) correlated Pearson r=0.223 and Spearman rho=0.249. Spearman-Brown=0.365 (bootstrap 95% CI 0.020 to 0.597); raw two-component alpha=0.365.
With two components, standardised alpha is mathematically determined by their correlation and adds little beyond that correlation and Spearman-Brown coefficient. Two subscales cannot establish a higher-order latent factor. Within-subscale consistency and between-subscale coherence answer different questions; multiple first-order dimensions do not automatically invalidate the broader composite.

## 5. GAToRS Negative

Complete-case n=100, k=10. Raw alpha=0.785 (bootstrap 95% CI 0.708 to 0.835); standardised alpha=0.775. Omega total=0.816 (bootstrap 95% CI 0.698 to 0.834; see method note). Mean Pearson inter-item r=0.256 (range -0.035 to 0.675; bootstrap CI 0.187 to 0.326). Mean polychoric r=0.299 (range -0.041 to 0.724). Parallel analysis suggested 2 factor(s).

Omega point estimate uses polychoric correlations; its bootstrap CI is Pearson-based. Broad composite intentionally combines distinguishable validated subscales.

At the constituent-subscale level, Personal Negative (S2) and Societal Negative (S4) correlated Pearson r=0.521 and Spearman rho=0.493. Spearman-Brown=0.685 (bootstrap 95% CI 0.521 to 0.796); raw two-component alpha=0.656.
With two components, standardised alpha is mathematically determined by their correlation and adds little beyond that correlation and Spearman-Brown coefficient. Two subscales cannot establish a higher-order latent factor. Within-subscale consistency and between-subscale coherence answer different questions; multiple first-order dimensions do not automatically invalidate the broader composite.

## 6. Overall conclusions

Cronbach's alpha is not a test of unidimensionality. It is affected by item count and average covariance, and can be distorted by violations of tau-equivalence, multidimensionality, redundancy, and correlated content/errors. High alpha can reflect redundancy or many items; low alpha can reflect short scales or deliberately heterogeneous facets.

Omega is generally preferable when loadings differ, but it still requires a defensible latent-factor model. Polychoric estimates respect ordered responses but can be unstable with sparse categories; every failed pair is retained as missing and explicitly labelled in the polychoric CSV rather than silently replaced by Pearson. Item deletion is diagnostic, not an automatic editing rule. Reliability describes these scores in this sample, not an immutable property of a questionnaire.

### Manuscript-oriented summary

- Fun: report alpha 0.868, omega 0.891, mean inter-item r 0.686, their CIs, and all three correlations; describe the intentionally broad three-item design and limited dimensionality evidence.
- IDAQ: report the established 15-item mean with alpha 0.811 and omega 0.856 in this sample, retaining the validated scoring definition.
- GAToRS Positive: report item-level alpha 0.691 and omega 0.763, plus the S1-S3 Pearson correlation 0.223 and Spearman-Brown 0.365; acknowledge the two subdimensions.
- GAToRS Negative: report item-level alpha 0.785 and omega 0.816, plus the S2-S4 Pearson correlation 0.521 and Spearman-Brown 0.685; acknowledge the two subdimensions.

Full item descriptives, response frequencies, item-total diagnostics, correlation matrices, factor loadings, eigenvalues, bootstrap failures, and polychoric estimation statuses are in the accompanying machine-readable CSV files.
