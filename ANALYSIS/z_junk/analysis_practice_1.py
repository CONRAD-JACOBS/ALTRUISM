import pandas as pd
from scipy.stats import ttest_ind  # Import specifically for clarity

# Load the CSV (adjust path if not in current directory)
df = pd.read_csv('model.csv')

# # Basic inspections
# print("\n :::::::::::::First 5 rows:::::::\n")
# print(df.head())          # First 5 rows to visualize data
# print("\n :::::::::::::Column names:::::::\n")
# print(df.columns.tolist())  # Confirm all column names
# print("\n :::::::::::::Data types:::::::::\n")
# print(df.dtypes)          # Data types – expect float64/int64 for most, object for DATE/GENDER
# print("\n :::::::::::::Missing values per column:::::::::::\n")
# print(df.isnull().sum())  # Check for missing values (should be low in simulated data)

# # Compute and save descriptive statistics for continuous variables
# desc = df.describe()
# # Save to CSV (transposed for better readability: stats as columns, variables as rows)
# desc.T.to_csv('descriptive_stats.csv')

#############################
# # UNIVARIATE GENERAL LINEAR MODEL -> LINEAR REGRESSION -> t-TESTS

# # Ensure GENDER is category/string and clean (no 'O' based on your update)
# print(df['GENDER'].value_counts())  # Quick check: should show only M and F counts

# # Extract groups (drop any NaN if present)
# male_attempts = df[df['GENDER'] == 'M']['DV_CAPTCHAS_ATTEMPTED'].dropna()
# female_attempts = df[df['GENDER'] == 'F']['DV_CAPTCHAS_ATTEMPTED'].dropna()

# # Independent t-test
# t_stat, p_value = ttest_ind(male_attempts, female_attempts, equal_var=True)  # Or False for Welch's

# # Output results
# print(f"Male group (n={len(male_attempts)}): Mean = {male_attempts.mean():.2f}, SD = {male_attempts.std():.2f}")
# print(f"Female group (n={len(female_attempts)}): Mean = {female_attempts.mean():.2f}, SD = {female_attempts.std():.2f}")
# print(f"T-statistic: {t_stat:.3f}")
# print(f"P-value: {p_value:.4f}")
# print(f"Significant at p < 0.05: {'Yes' if p_value < 0.05 else 'No'}")

# # Optional: Save results to a text file or CSV for records
# results_dict = {
#     'Group': ['Male', 'Female'],
#     'N': [len(male_attempts), len(female_attempts)],
#     'Mean': [male_attempts.mean(), female_attempts.mean()],
#     'SD': [male_attempts.std(), female_attempts.std()]
# }
# results_df = pd.DataFrame(results_dict)
# results_df.to_csv('t_test_gender_summary.csv', index=False)
# print("\nGroup summaries saved to 't_test_gender_summary.csv'")

# ### CHECK NORMALITY

# import scipy.stats as stats
# import statsmodels.api as sm
# import matplotlib.pyplot as plt

# male = df[df['GENDER'] == 'M']['DV_CAPTCHAS_ATTEMPTED']
# female = df[df['GENDER'] == 'F']['DV_CAPTCHAS_ATTEMPTED']

# # Shapiro-Wilk
# print("Shapiro-Wilk (p < 0.05 rejects normality):")
# print("Male:", stats.shapiro(male))
# print("Female:", stats.shapiro(female))

# # Q-Q plots for visual check
# sm.qqplot(male, line='45')
# plt.title('Q-Q Plot: Male Group')
# plt.show()

# sm.qqplot(female, line='45')
# plt.title('Q-Q Plot: Female Group')
# plt.show()  

# ### CHECK HOMOSKEDASTICITY, AKA, EQUALITY OF WITHIN-GROUP VARIANCES

# # Levene's test
# levene_stat, levene_p = stats.levene(male, female)
# print(f"Levene's test: W={levene_stat:.4f}, p={levene_p:.4f}")
# print("If p < 0.05: Variances unequal → prefer Welch's t-test")

# ### Levene's passed but Shapiro-Wilk failed, so run Mann-Whitney U test instead of t-test
# from scipy.stats import mannwhitneyu

# male = df[df['GENDER'] == 'M']['DV_CAPTCHAS_ATTEMPTED']
# female = df[df['GENDER'] == 'F']['DV_CAPTCHAS_ATTEMPTED']

# mwu_stat, mwu_p = mannwhitneyu(male, female, alternative='two-sided')
# print(f"Mann-Whitney U test: statistic={mwu_stat:.2f}, p={mwu_p:.4f}")
# if mwu_p < 0.05:
#     print("  → Significant difference between groups")
# else:
#     print("  → No significant difference")

#### HURDLE MODEL
import statsmodels.api as sm          # Add this import for families and constants
import statsmodels.formula.api as smf
import pandas as pd

# Load data, create interaction
df = pd.read_csv('model.csv')
df['EMPATHY_x_MIND_BELIEF'] = df['EMPATHY'] * df['MIND_BELIEF'].astype(float)

# Formula (adjust names; all predictors in both parts initially)
formula = ('DV_CAPTCHAS_ATTEMPTED ~ IV_GATORS_POS + IV_GATORS_NEG + '
           'IV_IDAQ + EMPATHY + MIND_BELIEF + EMPATHY_x_MIND_BELIEF')

# Fit hurdle (via separate steps for flexibility; or use smf for integrated)
# Binary: logit on engagement (1 if DV>0, 0 else)
df['engaged'] = (df['DV_CAPTCHAS_ATTEMPTED'] > 0).astype(int)
logit_model = smf.logit('engaged ~ IV_GATORS_POS + IV_GATORS_NEG + '
                        'IV_IDAQ + EMPATHY + MIND_BELIEF + EMPATHY_x_MIND_BELIEF', 
                        data=df).fit()

# Positive: NB on DV | DV>0
positive_df = df[df['DV_CAPTCHAS_ATTEMPTED'] > 0]
nb_model = smf.glm('DV_CAPTCHAS_ATTEMPTED ~ IV_GATORS_POS + IV_GATORS_NEG + '
                   'IV_IDAQ + EMPATHY + MIND_BELIEF + EMPATHY_x_MIND_BELIEF', 
                   family=sm.families.NegativeBinomial(), data=positive_df).fit()

print("\n::::::::::::LOGISTIC MODEL::::::::::::::::\n")
print(logit_model.summary())
print("\n::::::::::::NEGATIVE BINOMIAL MODEL::::::::::::::::\n")
print(nb_model.summary())