#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("G:/Programming/Udacity/ud120-projects/")
from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

start_time = time()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)

training_time = time()

accuracy = clf.score(features_test, labels_test)

end_time = time()

print(f"Accuracy of the Gaussian Naive Bayes classifier: {accuracy}")
print(f"Training time: {round(training_time - start_time, 3)}s")
print(f"Testing time: {round(end_time - training_time, 3)}s")
#########################################################
