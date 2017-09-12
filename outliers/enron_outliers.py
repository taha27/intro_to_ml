#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
from os import path
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from tools.feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below



