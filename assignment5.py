# Import libraries
import pandas as pd
from scipy import stats
from tabulate import tabulate
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder

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

######################     Chi-squared Tests     ###############################
def conduct_chi2_print_table(independent, dependent):
    print(f"Perform a Chi-Squared test on the '{independent}' column using '{dependent}' as the \ndependent variable:")
    contingency_table = pd.crosstab(df[independent], df[dependent])
    contingency_table.index.name = f"{independent}/{dependent}"
    print(tabulate(contingency_table, headers='keys', tablefmt='pretty', showindex=True))

    # conducting the test
    chi2_stat, p_value, dof = stats.chi2_contingency(contingency_table)[0:3]

    # printing results
    results = [
        ["Chi-squared stat", chi2_stat],
        ["p-value", p_value],
        ["dof", dof],
    ]
    print(tabulate(results, headers=["Statistic", "Value"], tablefmt="pretty"))

print("""
################################################################################
#######################     CHI-SQUARED TESTS    ###############################
################################################################################
""")

# Chi2 of Gender vs Survived?
conduct_chi2_print_table('Gender', 'Survived?')

# analysis
print("""\nWhen conducting a Chi-squared test, the null hypothesis claims that 
there is no association between the columns Gender and Survived. Or in other 
words, that the columns are independent of each other. The alternative 
hypothesis claims that the columns are not independent and thus have some sort 
of association.

Based on the results from the Chi-squared test, we ended up with a very large 
chi-squared statistic resulting in an extremely small p-value. Given the 
standard significance level of 0.05, the p-value is significantly smaller than 
this providing evidence to reject the null hypothesis that the two columns are 
independent and assume that the columns are in fact not independent indicating 
some sort of association. In the context, of the survival rate of crew members, 
this test indicates that the expected frequencies of male and  female crew that 
survived versus lived versus the actual frequencies are very different to the 
point that they are unlikely to have differed strictly by chance. All in all, 
this provides evidence that there may be an association between crew member's 
gender and whether they were saved or died.
""")

# Chi squared on Class/Dept vs survived
conduct_chi2_print_table('Class/Dept', 'Survived?')

# analysis
print("""Similar to the chi-squared test conducted previously, testing the
independence between the 'Class/Dept' column and the 'Survived?' columnn yielded
a very large chi squared stat and an extremely small p-value. Again, at the 0.05
significance level, this provides compelling evidence to reject the null 
hypothesis that the two columns are independent in favor of the alternative that
they are in fact not independent. This means that the class/department that the
crew member belonged to does appear to have an association with whether they
lived or died. This means that given new data about a crew member's class/dept 
we could use this column as a factor in deciding the worker's life outcome.
""")

# Chi squared on Joined vs survived
conduct_chi2_print_table('Joined', 'Survived?')

# analysis
print("""Since the chi-squared statistic is zero and the p-value is very large,
1.0 essentially, this indicates that there is insufficient evidece to reject the
null hypothesis that the 'Joined' and 'Survived?' columns are independent. This
would imply that there is no association on which stop the crew joined the ship
and whether or not they survived.
""")

##########################      ANOVA     ####################################
print("""
################################################################################
#############################     ANOVA    #####################################
################################################################################
""")

print("Perform an ANOVA on 'Class/Dept' as it is given in the data set using\n'Age' as the dependent column")

# Rename the Class/Dept column
df.rename(columns={"Class/Dept": "Class_Dept"}, inplace=True)
model = ols('Age ~ C(Class_Dept)', data=df).fit()

# create the anova table
anova_table = sm.stats.anova_lm(model)
print('\nANOVA results:\n', anova_table)

print("""\nBased on the p-value from the ANOVA table, since the p-value is less
than the standard 0.05 significance level, this provides evidence that there is
a difference in crew member ages relative to the class/dept in which they
worked. We should then reject the null hypothesis that there is no difference
between ages of crew members given their class/dept.
""")

# Conduct Tukey HSD
print("""Since there is statistical significance, we shall now conduct a Tukey
HSD comparison and report which departments are statistically different from 
each other:
""")

mc = MultiComparison(df['Age'], df['Class_Dept'])
result = mc.tukeyhsd()
print(result)

# provide analysis
print("""Based on the table provided from the Tukey HSD, there appears to be
statistically significant differences between the ages of crew members belonging
to the Restaurant Staff versus other groups. While it is not know at this point
if the age of the restaurant staff is younger or older than the other crews, it
simply provides insight that there is a statistically significant difference
from the other groups.
""")

##################      AGE CORRELATION    #######################
print("""
################################################################################
#######################     AGE CORRELATION    #################################
################################################################################
""")

print("Was there a correlation between age and survival for the crew?")

# encode the survived column into 1s and zeros
le = preprocessing.LabelEncoder()

df['Lived_Died_int'] = le.fit_transform(df['Survived?'])

pearson_corr = df["Age"].corr(df["Lived_Died_int"], 'pearson')
spearman_corr = df["Age"].corr(df["Lived_Died_int"], 'spearman')

print(f"Pearson correlation: {pearson_corr}")
print(f"Spearman correlation: {spearman_corr}\n")

print("""Based on these correlation values, it does not appear like age and 
survival are highly correlated. I encoded the survival column to 1s for survived
and 0 for Lost. Since these values are not highly correlated, that might
indicate that age is not a great feature to choose for predicting survival rate.
""")

##################      BIVARIATE VISUALIZATIONS   #######################
print("""
################################################################################
###################     BIVARIATE VISUALIZATIONS    ############################
################################################################################
""")
print("Displayed bar graph of age vs survived....")
bin_list = [0, 10, 20, 30, 40, 50, 60, 70, 80]

survived = pd.cut(x=df[df['Survived?'] == 'SAVED']['Age'], bins=bin_list)
died = pd.cut(x=df[df['Survived?'] == 'LOST']['Age'], bins=bin_list)

x = np.arange(len(bin_list)-1)  # the label locations
width = 0.35  # the width of the bars

labels = [str(label) for label in died.unique()]
survived_values = [value for value in survived.value_counts(sort=False)]
died_values = [value for value in died.value_counts(sort=False)]

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x - width / 2, survived_values, width, label='Survived')
ax.bar(x + width / 2, died_values, width, label='Died')
plt.ylabel('Count')
plt.xlabel('Age Ranges')
plt.title('Histogram of Age Ranges of Titantic Crew members')
plt.legend()
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
plt.show()


print("Displayed bar graph of class/dept vs survived....")
grouped = df.groupby(['Class_Dept', 'Survived?']).size().unstack(fill_value=0)
grouped.plot(kind='bar', stacked=False, figsize=(10, 6))
plt.title('Survived Counts by Class/Dept')
plt.xlabel('Class/Dept')
plt.ylabel('Count')
plt.xticks(rotation=70)
plt.legend(title='Survived', labels=['SAVED', 'LOST'])
plt.tight_layout()
plt.show()

print("Displayed bar graph of joined vs survived....")
grouped = df.groupby(['Joined', 'Survived?']).size().unstack(fill_value=0)
grouped.plot(kind='bar', stacked=False, figsize=(10, 6))
plt.title('Survived Counts by Joined')
plt.xlabel('Joined')
plt.ylabel('Count')
plt.xticks(rotation=70)
plt.legend(title='Survived', labels=['SAVED', 'LOST'])
plt.tight_layout()
plt.show()

##################       MULTIVARIATE VISUALIZATIONS   #######################
print("""
################################################################################
##################     MULTIVARIATE VISUALIZATIONS    ##########################
################################################################################
""")

df['Age_Bins'] = pd.cut(x=df['Age'], bins=[0, 9, 19, 29, 39, 49, 59, 69, 79])

table = pd.pivot_table(df, values='Lived_Died_int', index=['Joined'], columns=['Class_Dept', 'Age_Bins'])

table.applymap(lambda x: 1 - x)  # the results are inverted for the heatmap

plt.figure(figsize=(11,5))
#ax = plt.axes()
plt.suptitle("Heatmap Comparing Age and Class_Dept")
sns.heatmap(table, annot=True, fmt='.2f')

plt.show()
print("Displayed heat map of multiple variables")

##################       Report which features Part 1   #######################
print("""
################################################################################
################     Report which features - Part 1    #########################
################################################################################
""")
print("""Based on visualizations and chi-squared analysis, I would use the
following features:
      1. Gender
      2. Class/Dept
      3. Age
Based off of the chi-squared analysis for gender and class/dept against survived
the test statistic was extermely large with a small p-value indicating that the
values of the Gender and Class/Dept imply some sort of association with whether
a crew member lived or died. This means that the variable values were not
independent. Based off of the visualizations for age and class/dept, there seems
to be some indication that certain values experienced higher rates of survival
versus others. Since there is a difference between different groups, this would
help build a more accurate model.
""")

##################       Report which features Part 2   #######################
print("""
################################################################################
################     Report which features - Part 2    #########################
################################################################################
""")

df = df.dropna(subset=["Lived_Died_int"])
df = df.apply(preprocessing.LabelEncoder().fit_transform)

X = df.drop(columns=["Lived_Died_int", 'Age_Bins', 'Survived?'])
y = df["Lived_Died_int"]

features = X.columns

model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, n_features_to_select=3)
X = df.loc[:, features]
rfe = rfe.fit(X, y)
print('RFE (Recursive Feature Elimination) gives the following results:')
# summarize the selection of the attributes:
print(rfe.support_)
print(rfe.ranking_)
print(f'\nIn other words, these are the top 3 features:')
new_features = features[rfe.support_]
print(new_features)
X = df.loc[:, new_features]
# y stays the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy = {str(accuracy_score(y_test, y_pred))}')


print("""RFE reported these top 3 features:
      1. Died
      2. Age
      3. Boat
""")

le = LabelEncoder()

# Encode features and target variable
df_encoded = df.copy()
for column in df_encoded.columns:
    df_encoded[column] = le.fit_transform(df_encoded[column])

X = df_encoded.drop(columns=["Lived_Died_int", 'Age_Bins', 'Survived?'])
y = df_encoded['Lived_Died_int']  # Target variable

selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(X, y)

# Get the top features
selected_features = X.columns[selector.get_support()]

print(f"""SelectKBest reported these top 3 features:
      1. {selected_features[0]}
      2. {selected_features[1]}
      3. {selected_features[2]}
""")

##################       Report which features Part 3   #######################
print("""
################################################################################
################     Report which features - Part 3    #########################
################################################################################
""")

print("""Based on the feature selection methods, it would appear that Boat and
Died are both good features to be able to build accurate predictive models.
Clearly, Died is essentially the same as the target variable of survived because
all the crew members who died in the Titanic crash would have their death year
as 1912. While SelectKBest didn't include it, RFE was able to find that age
was a significant feature. Thus, if I were to recommend a simple model using
certain features, I think a good model could include 'Died', 'Age', 'Boat', and
'Class/Dept'. I think class would be a good choice because based on my
chi-squared analysis from before, this variable seemed to not be independent of
whether the crew member lived or died. Even though I claimed Gender might be a 
significant feature to use, that doesn't seem to be a significant feature
according to the feature selection models.
""")