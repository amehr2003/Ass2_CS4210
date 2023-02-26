#-------------------------------------------------------------------------
# AUTHOR: Anita Mehrazarin
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 90 min
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# reading the training data in a csv file
X = []
Y = []
weather_training = []
X2 = []
Y2 = []
weather_test = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            weather_training.append(row)

# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
for row in weather_training:
    data = []
    for i, features in enumerate(row):
        if i == 1:
            data.append(1 if features == 'Sunny' else 2 if features == 'Overcast' else 3)
        elif i == 2:
            data.append(1 if features == 'Hot' else 2 if features == 'Mild' else 3)
        elif i == 3:
            data.append(1 if features == 'High' else 2)
        elif i == 4:
            data.append(1 if features == 'Weak' else 2)
    X.append(data)

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
for row in weather_training:
    Y.append(1 if row[5] == 'Yes' else 2)

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
# printing the header os the solution
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            weather_test.append(row)

for row in weather_test:
    data2 = []
    for i, features in enumerate(row):
        if i == 1:
            data2.append(1 if features == 'Sunny' else 2 if features == 'Overcast' else 3)
        elif i == 2:
            data2.append(1 if features == 'Hot' else 2 if features == 'Mild' else 3)
        elif i == 3:
            data2.append(1 if features == 'High' else 2)
        elif i == 4:
            data2.append(1 if features == 'Weak' else 2)
    X2.append(data2)
    Y2.append(1 if row[5] == 'Yes' else 2)
    # use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
    prediction = clf.predict([[data2[0], data2[1], data2[2], data2[3]]])[0]
    PROB = clf.predict_proba([[data2[0], data2[1], data2[2], data2[3]]])[0]
    confidence = PROB[prediction - 1]
    play_tennis = 'Yes' if prediction == 1 else 'No'
    if confidence >= 0.75:
        print(row[0], row[1], row[2], row[3], row[4], play_tennis, confidence)
