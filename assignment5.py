# Import libraries
import pandas as pd
from scipy import stats
from tabulate import tabulate

# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

# Import data
df = pd.read_csv('Titanic Crew.csv')

# Drop the url column
df.drop(columns="URL", inplace=True)

# Filter out those that are not LOST or SURVIVED
df = df[(df["Survived?"] == "LOST") | (df["Survived?"] == "SAVED")]



############################     Chi-squared Tests     ###############################
print("Contingency Table of Gender vs Survived?:\n")

contingency_table = pd.crosstab(df['Gender'], df['Survived?'])
# contingency_table.index.name = "Gender/Survived?"
print(tabulate(contingency_table, headers='keys', tablefmt='pretty', showindex=True))


# Conduct Chi squared
chi2_stat, p_value, dof = stats.chi2_contingency(contingency_table)[0:3]
# print("\nConducting Chi2 Analysis...")
# print(f"Chi-squared stat: {chi2_stat}")
# print(f"p-value: {p_value}")
# print(f"dof: {dof}")

results = [
    ["Chi-squared stat", chi2_stat],
    ["p-value", p_value],
    ["dof", dof],
]

# Display the results using tabulate
print(tabulate(results, headers=["Statistic", "Value"], tablefmt="fancy_grid"))


