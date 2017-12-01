#ML-SVMSKLearn.py
import numpy as np 
from sklearn import preprocessing, model_selection, neighbors, svm 
import pandas as pd 

'''
SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, 
	coef0=0.0, shrinking=True, probability=False, tol=0.001, 
	cache_size=200, class_weight=None, verbose=False, 
	max_iter=-1, decision_function_shape=’ovr’, random_state=None)
gmma, donot mess with it
decision_function_shape=’ovr’ default 'one vs rest'
'''
df = pd.read_csv('/Users/shaobinmin/Documents/python/homecodes/ML/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
clf = svm.SVC()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)
