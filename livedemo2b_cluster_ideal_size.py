import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

data = pd.read_csv('demo2_unlabelled.csv')


print(data['Blood pressure'].unique())

dataSubset = data[['Age', 'Weight', 'Height']]

# Generate a new column that has the value True (1) IF
# the person has Elevated or Hypertension for blood pressure,
# False (0) otherwise
#
# This is a 1-hot encoding of our blood pressure column
dataSubset['HasHighBp'] = (data['Blood pressure'] == 'elevated') \
    | (data['Blood pressure'] == 'hypertension')
    
dataSubset['T1D'] = data['Diabetes'] == 'Type1'
dataSubset['T2D'] = data['Diabetes'] == 'Type2'

dataSubset['Fever'] = data['Fever'] == 'Y'
dataSubset['Cough'] = data['Cough'] == 'Y'
dataSubset['Smoker'] = data['Smoker'] == 'Y'

# Scale the data using standard scaler
# This will attempt to pre-process the data by subtracting the mean
# (centering) and dividing by the standard deviation
# Gives us a mean of 0 and stdev of 1
# 
# Since some units have different scales, it could mess with
# how the k-means clusters are formed
#
# So we transform the scales to be uniform
scaler = StandardScaler()
dataScaled = scaler.fit_transform(dataSubset)

clusterTest = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
clusterInertia = []

for clusterSize in clusterTest:
    kmeansTest = KMeans(n_clusters = clusterSize, random_state = 9)
    kmeansTest.fit(dataScaled)
    inertia = kmeansTest.inertia_
    
    print(f'Cluster size {clusterSize}, inertia is {inertia}')
    clusterInertia.append(inertia)

plt.plot(clusterTest, clusterInertia)
plt.xlabel('Cluster Size')
plt.ylabel('Inertia')
plt.show()

