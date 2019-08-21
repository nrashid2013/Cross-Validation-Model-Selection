# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:28:23 2019

@author:
"""


import numpy as np
import pandas as pd

df=pd.read_csv('Purchased_Dataset.csv')
x=df[['Age', 'EstimatedSalary']]
y=df['Purchased']



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=5)
Knnclassifier = KNeighborsClassifier(n_neighbors=5)
Knnclassifier.fit(X_train, y_train)
y_pred = Knnclassifier.predict(X_test)
accuracy_score1=metrics.accuracy_score(y_test, y_pred)
print(X_train.head())



from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=7)
Knnclassifier = KNeighborsClassifier(n_neighbors=5)
Knnclassifier.fit(X_train, y_train)
y_pred = Knnclassifier.predict(X_test)
accuracy_score2=metrics.accuracy_score(y_test, y_pred)
print(X_train.head())




from sklearn.model_selection import cross_val_score
Knnclassifier=KNeighborsClassifier(n_neighbors=4)
Z_KNN_cross_val_score = cross_val_score(Knnclassifier, x,  y, cv=10, scoring='accuracy').mean()


from sklearn.linear_model import LogisticRegression
logreg =  LogisticRegression()
Z_Logi_cross_val_score = cross_val_score(logreg, x,  y, cv=10, scoring='accuracy').mean()

