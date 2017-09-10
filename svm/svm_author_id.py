#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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
from sklearn.svm import SVC

# features_train = features_train[:len(features_train)//100]
# labels_train = labels_train[:len(labels_train)//100]

start_time = time()
clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
training_time = time()

accuracy = clf.score(features_test, labels_test)
end_time = time()

print(f"Accuracy of the SVM SVC: {accuracy}")
print(f"Training time: {round(training_time - start_time, 3)}s")
print(f"Testing time: {round(end_time - training_time, 3)}s")

pred = clf.predict(features_test)

import numpy as np
class_counts = np.bincount(pred)
print(f"{class_counts[1]} test sets are predicted to be in the Chris(1) class.")
#########################################################
