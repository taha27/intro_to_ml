#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from os import path
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
start_time = time()

clf = DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
training_time = time()

accuracy = clf.score(features_test, labels_test)
end_time = time()

print(f"Accuracy of the Decision Tree Classifier : {accuracy}")
print(f"Training time: {round(training_time - start_time, 3)}s")
print(f"Testing time: {round(end_time - training_time, 3)}s")
#########################################################
