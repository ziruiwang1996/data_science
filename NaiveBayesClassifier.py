# Zirui Wang
# Hw 2
# Naive Bayes Classifier
import pandas as pd
from sklearn.model_selection import train_test_split
import statistics
import math

#preprocessing
df = pd.read_csv("adult_csv.csv")
# Feature selection: remove features ‘capitalgain’, ‘capitalloss’ and ‘native-country.’
df = df.drop(columns=['capitalgain', 'capitalloss', 'native-country'])
# Remove instances with at least one missing values
df = df.dropna(axis=0, how='any')
# Randomly split the dataset into 90% as train and 10% as test
train, test = train_test_split(df, test_size=0.1, random_state=10)
# Build the classifier
# Estimate the probabilities for continuous attributes (‘age’, ‘fnlwgt’, ‘education-num’ and ‘hoursperweek’) by fitting a Normal distribution.
class_1 = train.loc[train['class']=='<=50K']
age_mean_c1, age_stdev_c1 = statistics.mean(class_1['age']), statistics.stdev(class_1['age'])
fnlwgt_mean_c1, fnlwgt_stdev_c1 = statistics.mean(class_1['fnlwgt']), statistics.stdev(class_1['fnlwgt'])
edu_mean_c1, edu_stdev_c1 = statistics.mean(class_1['education-num']), statistics.stdev(class_1['education-num'])
hoursperweek_mean_c1, hoursperweek_stdev_c1 = statistics.mean(class_1['hoursperweek']), statistics.stdev(class_1['hoursperweek'])
class_2 = train.loc[train['class']=='>50K']
age_mean_c2, age_stdev_c2 = statistics.mean(class_2['age']), statistics.stdev(class_2['age'])
fnlwgt_mean_c2, fnlwgt_stdev_c2= statistics.mean(class_2['fnlwgt']), statistics.stdev(class_2['fnlwgt'])
edu_mean_c2, edu_stdev_c2 = statistics.mean(class_2['education-num']), statistics.stdev(class_2['education-num'])
hoursperweek_mean_c2, hoursperweek_stdev_c2 = statistics.mean(class_2['hoursperweek']), statistics.stdev(class_2['hoursperweek'])
def p_from_fitted_norm(x, mean, stdev):
    return (1/math.sqrt(2 * math.pi * stdev**2)) * math.exp(-1/2 * ((x-mean)/stdev)**2)

# Calculate the probabilities of discrete attributes (remaining attributes) using Laplace smoothing.
def laplace_smoothing(class_c, attribute, attribute_value_i, hyper_param):
    df_c = train.loc[train['class']==class_c]
    n_c = df_c.shape[0]
    df_ic = df_c.loc[df_c[attribute]==attribute_value_i]
    n_ic = df_ic.shape[0]
    return (n_ic + 1)/(n_c + hyper_param)

# Perform the classification
# Train and test with the corresponding train and test datasets
TP_TN = 0
for index, each in test.iterrows():
    p_age_c1 = p_from_fitted_norm(each['age'], age_mean_c1, age_stdev_c1)
    p_fnlwgt_c1 = p_from_fitted_norm(each['fnlwgt'], fnlwgt_mean_c1, fnlwgt_stdev_c1)
    p_edu_c1 = p_from_fitted_norm(each['education-num'], edu_mean_c1, edu_stdev_c1)
    p_hoursperweek_c1 = p_from_fitted_norm(each['hoursperweek'], hoursperweek_mean_c1, hoursperweek_stdev_c1)
    p_workclass_c1 = laplace_smoothing('<=50K', 'workclass', each['workclass'], 100)
    p_education_c1 = laplace_smoothing('<=50K', 'education', each['education'], 100)
    p_marital_status_c1 = laplace_smoothing('<=50K', 'marital-status', each['marital-status'], 100)
    p_relationship_c1 = laplace_smoothing('<=50K', 'relationship', each['relationship'], 100)
    p_race_c1 = laplace_smoothing('<=50K', 'race', each['race'], 100)
    p_sex_c1 = laplace_smoothing('<=50K', 'sex', each['sex'], 100)
    p_education_c1 = laplace_smoothing('<=50K', 'education', each['education'], 100)

    p_age_c2 = p_from_fitted_norm(each['age'], age_mean_c2, age_stdev_c2)
    p_fnlwgt_c2 = p_from_fitted_norm(each['fnlwgt'], fnlwgt_mean_c2, fnlwgt_stdev_c2)
    p_edu_c2 = p_from_fitted_norm(each['education-num'], edu_mean_c2, edu_stdev_c2)
    p_hoursperweek_c2 = p_from_fitted_norm(each['hoursperweek'], hoursperweek_mean_c2, hoursperweek_stdev_c2)
    p_workclass_c2 = laplace_smoothing('>50K', 'workclass', each['workclass'], 100)
    p_education_c2 = laplace_smoothing('>50K', 'education', each['education'], 100)
    p_marital_status_c2 = laplace_smoothing('>50K', 'marital-status', each['marital-status'], 100)
    p_relationship_c2 = laplace_smoothing('>50K', 'relationship', each['relationship'], 100)
    p_race_c2 = laplace_smoothing('>50K', 'race', each['race'], 100)
    p_sex_c2 = laplace_smoothing('>50K', 'sex', each['sex'], 100)
    p_education_c2 = laplace_smoothing('>50K', 'education', each['education'], 100)

    p_less_than_50K = p_age_c1*p_fnlwgt_c1*p_edu_c1*p_hoursperweek_c1*p_workclass_c1*p_education_c1*p_marital_status_c1*p_relationship_c1*p_race_c1*p_sex_c1*p_education_c1
    p_greater_than_50K = p_age_c2*p_fnlwgt_c2*p_edu_c2*p_hoursperweek_c2*p_workclass_c2*p_education_c2*p_marital_status_c2*p_relationship_c2*p_race_c2*p_sex_c2*p_education_c2

    if p_less_than_50K > p_greater_than_50K :
        class_pred = '<=50K'
    elif p_less_than_50K < p_greater_than_50K:
        class_pred = '>50K'
    # get TP and TN
    if class_pred == each['class']:
        TP_TN += 1

# Report the testing accuracy
accuracy = TP_TN/test.shape[0]
print("Accuracy is: ", accuracy) # Result: 0.768462206776716 when hyper_param is 100
# Sidenote: planning to tune hyper_param using a for loop, however due to short amount of time left, this part is not included in this submission
