
import numpy as np
import pandas as pd

admissions = pd.read_csv('binary.csv')
###print (admissions.head())

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
###print (data)

data = data.drop('rank', axis=1)
###print (data.head())

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:,field] = (data[field]-mean)/std

###print (data.head())
    
# Split off random 10% of the data for testing
np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

"""
print("data")
print (data)
print("test_data")
print (test_data)


data
     admit       gre       gpa  rank_1  rank_2  rank_3  rank_4
209      0 -0.066657  0.289305       0       1       0       0
280      0  0.625884  1.445476       0       1       0       0
33       1  1.837832  1.603135       0       0       1       0
....

test_data
     admit       gre       gpa  rank_1  rank_2  rank_3  rank_4
20       0 -0.759199 -0.577822       0       0       1       0
21       1  0.625884  0.630901       0       1       0       0
48       0 -1.278605 -2.390908       0       0       0       1
50       0  0.452749  1.235263       0       0       1       0
....

"""

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']

features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']
"""
print("features_test")
print (features_test)
print("targets_test")
print (targets_test)

features_test
          gre       gpa  rank_1  rank_2  rank_3  rank_4
20  -0.759199 -0.577822       0       0       1       0
21   0.625884  0.630901       0       1       0       0
48  -1.278605 -2.390908       0       0       0       1
50   0.452749  1.235263       0       0       1       0
....

targets_test
20     0
21     1
48     0
50     0
...

"""
