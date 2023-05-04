# R-equivalent
# library(pandas)
import pandas as pd

# Module not found error?
# pip install sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


data = pd.read_csv('demo1_labeled.csv')

# Write out summary stats for our data frame
print(data.describe(include = 'all'))

dataSubset = data[['age', 'cholesterol_level']]

trainData, testData, trainLabels, testLabels = \
    train_test_split(dataSubset, 
                     data['has_disease'],
                     test_size = 0.5,
                     random_state=3)

model = LogisticRegression()
# Trains our model
model.fit(trainData, trainLabels)

# Time to test the accuracy of our model
testPredictionLabels = model.predict(testData)

modelAccuracy = accuracy_score(testLabels, 
                               testPredictionLabels)

print(modelAccuracy)

cm = confusion_matrix(testLabels, testPredictionLabels)

# Convert this to a percentage
cm_percentage = cm / cm.sum()

# Actual = 1
predictedData = [40, 183]

testPredict = model.predict([predictedData])
print(testPredict)

testPredictProb = model.predict_proba([predictedData])
print(testPredictProb)




