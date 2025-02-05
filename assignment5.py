# Import libraries
import pandas as pd
from scipy import stats
from tabulate import tabulate
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison

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

##################      SEPARATE GENDER COLUMN      #######################
print("""
################################################################################
#####################     SEPARATE GENDER COLUMN    ############################
################################################################################
""")

print("""What is the correlation of 'female' to survived? What is the correlation
of 'male' to survived?""")


