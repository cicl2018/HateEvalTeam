# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import svm, model_selection, cross_validation
import numpy as np
import pandas as pd

FILE = 'C:/Users/Denise/Documents/Studium/WS 1819/Vhallenges WS1819/Data_Files/Data Files/1_Practice/trial_en.tsv'

data = pd.read_csv(FILE, sep='\t',index_col=0)

X = np.array(data['text'])
Y = np.array(data['HS'])

print(data)


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size=0.1)

clf = svm.SVC()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print(accuracy)
