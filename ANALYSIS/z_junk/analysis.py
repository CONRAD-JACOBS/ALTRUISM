import pandas as pd

# Load the CSV (adjust path if not in current directory)
df = pd.read_csv('model.csv')

# Basic inspections
print("\n :::::::::::::First 5 rows:::::::\n")
print(df.head())          # First 5 rows to visualize data
print("\n :::::::::::::Column names:::::::\n")
print(df.columns.tolist())  # Confirm all column names
print("\n :::::::::::::Data types:::::::::\n")
print(df.dtypes)          # Data types – expect float64/int64 for most, object for DATE/GENDER
print("\n :::::::::::::Missing values per column:::::::::::\n")
print(df.isnull().sum())  # Check for missing values (should be low in simulated data)

# Compute and save descriptive statistics for continuous variables
desc = df.describe()
# Save to CSV (transposed for better readability: stats as columns, variables as rows)
desc.T.to_csv('descriptive_stats.csv')