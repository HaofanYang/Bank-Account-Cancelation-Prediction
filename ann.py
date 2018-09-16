#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 23:18:11 2018

@author: yanghaofan
"""
# ANN

# Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# We don't have any missing data here. Great!

# Dealing with categorical varibles 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X1 = LabelEncoder()
X[:, 1] = labelEncoder_X1.fit_transform(X[:, 1])
labelEncoder_X2 = LabelEncoder()
X[:, 2] = labelEncoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Since the number of countries are fixed, So the dummy variables we just Created are not necessarily independent
# A.K.A dummy variable trap. To aleviate this problem, we need to delete the first column
X = X[:, 1:]

# Split the dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

# Feature scalling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)