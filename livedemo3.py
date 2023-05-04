import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv('demo3_cat_species.csv')

dataSubset = data[['WeightLb', 'SizeCm']]

dataSubset['CanRoar'] = data['CanRoar'] == 'Yes'
dataSubset['CanPurr'] = data['CanPurr'] == 'Yes'
dataSubset['CanMeow'] = data['CanMeow'] == 'Yes'

labels = data['Genus']

trainData, testData, trainLabels, testLabels = train_test_split(
    dataSubset,
    labels,
    test_size = 0.5,
    random_state = 23)

model = DecisionTreeClassifier()
model.fit(trainData, trainLabels)

testPredictLabels = model.predict(testData)
accuracy = accuracy_score(testLabels, testPredictLabels)

plt.figure(figsize = (48, 24))
plot_tree(model, filled = True, feature_names = dataSubset.columns, 
          class_names = labels.unique())
plt.show()