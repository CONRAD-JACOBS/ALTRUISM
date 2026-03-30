import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower, GofChisquarePower
from itertools import chain, combinations
from patsy import dmatrices


# Load the CSV (adjust path if not in current directory)
data = pd.read_csv('model.csv')

# Basic inspections
print("\n :::::::::::::First 5 rows:::::::\n")
print(data.head())          # First 5 rows to visualize data
print("\n :::::::::::::Column names:::::::\n")
print(data.columns.tolist())  # Confirm all column names
print("\n :::::::::::::Data types:::::::::\n")
print(data.dtypes)          # Data types – expect float64/int64 for most, object for DATE/GENDER
print("\n :::::::::::::Missing values per column:::::::::::\n")
print(data.isnull().sum())  # Check for missing values (should be low in simulated data)

# Compute and save descriptive statistics for continuous variables
desc = data.describe()
# Save to CSV (transposed for better readability: stats as columns, variables as rows)
desc.T.to_csv('descriptive_stats.csv')

# # Define predictor columns (adjust to match your data; include all potential factors)
# predictors = ['IV_GATORS_pers_pos', 'IV_GATORS_soc_pos', 'IV_GATORS_pers_neg', 'IV_GATORS_soc_neg',
#               'IV_GATORS_pos', 'IV_GATORS_neg', 'IV_IDAQ', 'IV_EMPATHY', 'IV_MENTACY_CONFIDENCE']


# Define predictor columns (adjust to match your data; include all potential factors)
predictors = ['IV_GATORS_pers_pos', 'IV_GATORS_soc_pos', 'IV_GATORS_pers_neg', 'IV_GATORS_soc_neg',
              'IV_IDAQ', 'IV_EMPATHY', 'IV_MENTACY_CONFIDENCE']

# Add interaction term (hypothesized positive interaction)
data['MENTACY_CONFIDENCExEMPATHY'] = data['IV_MENTACY_CONFIDENCE'] * data['IV_EMPATHY']
predictors.append('MENTACY_CONFIDENCExEMPATHY')  # Now includes interaction; total ~10 predictors

# Filter for hurdle: binary outcome and positive time
binary_data = data.copy()
positive_time_data = data[data['DV_TOTAL_CAPTCHA_TIME'] > 0].copy()

# Function for best subset selection (exhaustive; returns best model by BIC)
def best_subset_selection(y, X, model_type='ols'):
    """
    Exhaustive best subset selection using BIC.
    model_type: 'logit' for logistic, 'ols' for normal linear, 'gamma' for Gamma GLM.
    """
    n_predictors = X.shape[1]
    best_bic = np.inf
    best_model = None
    best_subset = None
    
    # Generate all possible subsets (2^n - 1 non-empty)
    for k in range(1, n_predictors + 1):
        for subset in combinations(range(n_predictors), k):
            subset_cols = list(subset)
            X_subset = X.iloc[:, subset_cols]
            X_subset = sm.add_constant(X_subset)  # Add intercept
            
            if model_type == 'logit':
                model = sm.Logit(y, X_subset).fit(disp=0)
            elif model_type == 'ols':
                model = sm.OLS(y, X_subset).fit(disp=0)
            elif model_type == 'gamma':
                model = sm.GLM(y, X_subset, family=sm.families.Gamma()).fit(disp=0)
            else:
                raise ValueError("Invalid model_type")
            
            if model.bic < best_bic:
                best_bic = model.bic
                best_model = model
                best_subset = [X.columns[i] for i in subset_cols]
    
    return best_model, best_subset, best_bic

# Prepare design matrices
y_binary, X_binary = dmatrices('DV_CAPTCHA_DECISION ~ ' + ' + '.join(predictors), binary_data, return_type='dataframe')
y_time, X_time = dmatrices('DV_TOTAL_CAPTCHA_TIME ~ ' + ' + '.join(predictors), positive_time_data, return_type='dataframe')

# Step 1: Best subset for binary decision (logistic regression)
best_logit_model, best_logit_subset, best_logit_bic = best_subset_selection(y_binary['DV_CAPTCHA_DECISION'], X_binary.drop(columns=['Intercept']), model_type='logit')
print("Best Subset for Engagement Decision (Logistic):")
print(f"Selected predictors: {best_logit_subset}")
print(best_logit_model.summary())
print(f"BIC: {best_logit_bic}\n")

# Step 2: Best subset for time among engagers
# Option A: Assuming normality (OLS)
best_ols_model, best_ols_subset, best_ols_bic = best_subset_selection(y_time['DV_TOTAL_CAPTCHA_TIME'], X_time.drop(columns=['Intercept']), model_type='ols')
print("Best Subset for Engagement Time (OLS - Normal Assumption):")
print(f"Selected predictors: {best_ols_subset}")
print(best_ols_model.summary())
print(f"BIC: {best_ols_bic}\n")

# Option B: Non-normal (Gamma GLM for positive skewed time)
best_gamma_model, best_gamma_subset, best_gamma_bic = best_subset_selection(y_time['DV_TOTAL_CAPTCHA_TIME'], X_time.drop(columns=['Intercept']), model_type='gamma')
print("Best Subset for Engagement Time (Gamma GLM - Non-Normal):")
print(f"Selected predictors: {best_gamma_subset}")
print(best_gamma_model.summary())
print(f"BIC: {best_gamma_bic}\n")

# Power Analysis Section
# Adjust parameters: effect_size (Cohen's d for linear, f for logistic approx), alpha, power
# For logistic, approximate with chi-square GoF (simplified; for multi-var, simulate if needed)
alpha = 0.05
desired_power = 0.80

# Power for Step 1 (Logistic): Approximate for binary predictor effect
# Assume medium effect (odds ratio ~2.5 → logOR ~0.916, but use chi2 for multi-var approx)
power_logit = GofChisquarePower()
n_bins = 5
#n_bins = len(predictors)  # Degrees of freedom ~ num predictors
effect_size_logit = 0.3  # Medium; adjust based on expected (e.g., small=0.1, large=0.5)
n_logit = power_logit.solve_power(effect_size=effect_size_logit, alpha=alpha, power=desired_power, n_bins=n_bins)
print(f"Estimated sample size for Logistic Step (effect_size={effect_size_logit}): {np.ceil(n_logit)}")

# Power for Step 2 (Linear/OLS): t-test power for regression coefficient
power_linear = TTestIndPower()
effect_size_linear = 0.3  # Medium Cohen's f; adjust
n_linear = power_linear.solve_power(effect_size=effect_size_linear, alpha=alpha, power=desired_power, nobs1=None)
print(f"Estimated sample size for Linear Step (effect_size={effect_size_linear}, among engagers): {np.ceil(n_linear)}")

# Note: For Step 2, total N = n_linear / expected_proportion_engagers (e.g., if 50% engage, double total N)
# Refine with simulation if correlations/effects are known.

from statsmodels.stats.power import FTestPower

# Example: power for one coefficient (df_num=1) in model with k predictors total
k_total = 6  # e.g., your final best-subset model
effect_size_f2 = 0.001  # medium unique contribution
power_analysis = FTestPower()
n_needed = power_analysis.solve_power(effect_size=effect_size_f2,
                                      df_num=1,
                                      df_denom=None,  # solves for N
                                      alpha=0.05,
                                      power=0.80)

# Approximate total N (df_denom = N - k_total - 1)
N = n_needed + k_total + 1
print(f"Approximate N needed among engagers: {np.ceil(N)}")