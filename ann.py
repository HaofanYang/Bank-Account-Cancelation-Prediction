#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 23:18:11 2018

@author: yanghaofan
"""

                                        # Part 1. Data Preprocessing
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








                                # Part 2. Building ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
# Adding hidden layers
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
for i in range(3):
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer 
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
# Predicting on the test set
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Constructing confusion matrix
"""
matrix = [[0, 0],[0, 0]]
i = 0
while i < len(y_pred):
    if (y_test[i] == y_pred[i]):
        if (y_pred[i] == 1):
            matrix[1][1] += 1
        else:
            matrix[0][0] += 1
    else:
        if (y_pred[i] == 0):
            matrix[1][0] += 1
        else:
            matrix[0][1] += 1
    i = i + 1
"""
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc_test = 1.0 * (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])

# Test with a single prediction
new_pred = classifier.predict(np.array(sc.transform([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_pred = (new_pred > 0.5)








                        # Part 3. Evaluating the ANN (K fold cross validation)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    # Adding hidden layers
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    for i in range(3):
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the output layer 
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
var = accuracies.var()

import vpython as vs
vs.ModelLearning(X_train, y_train)



                        # Part 4. Tunning the model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    # Adding hidden layers
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    for i in range(3):
        classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the output layer 
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32], 
              'epochs': [100, 500],
              'optimizer': []}