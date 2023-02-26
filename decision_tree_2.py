#-------------------------------------------------------------------------
# AUTHOR: Anita Mehrazarin
# FILENAME: decision_tree_2.py
# SPECIFICATION: creates a decision tree considering provided data
# FOR: CS 4210- Assignment #2
# TIME SPENT: 90 minutes
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to
# work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']
for ds in dataSets:

    dbTraining = []
    X = []
    Y = []
    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young =
    # 1, Prepresbyopic = 2, Presbyopic = 3 so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    dictionary = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, 'Myope': 1, 'Yes': 1, 'No': 2,
                  'Normal': 1, 'Reduced': 2, 'Astigmatism': 1, 'Hypermetrope': 2}
    for row in dbTraining:
        data2 = []
        for i in range(4):
            data2.append(dictionary[row[i]])
        X.append(data2)

    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1,
    # No = 2, so Y = [1, 1, 2, 2, ...]
    for row in dbTraining:
        Y.append(dictionary[row[4]])

    accuracy = 0
    # loop your training and test tasks 10 times here
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)
        dbTest = []

        # read the test data and add this data to dbTest
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)

        predicted = 0
        for row in dbTest:
            # transform the features of the test instances to numbers following the same strategy done during
            # training, and then use the decision tree to make the class prediction. For instance: class_predicted =
            # clf.predict([[3, 1, 2, 1]])[0] where [0] is used to get an integer as the predicted class label so that
            # you can compare it with the true label
            data3 = []
            for i in range(4):
                data3.append(dictionary[row[i]])
            class_predicted = clf.predict([data3])[0]
            # compare the prediction with the true label (located at data[4]) of the test instance to start calculating
            if class_predicted == dictionary[row[4]]:
                predicted += 1

    # find the average of this model during the 10 runs (training and test set)
        AVG = predicted / len(dbTest)
        accuracy += AVG

    # print the average accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    accuracy /= 10
    print("final accuracy when training on " + ds + ": " + str(accuracy))