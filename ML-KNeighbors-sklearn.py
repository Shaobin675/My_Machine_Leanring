#ML-KNeighbors-sklearn.py
import numpy as np 
from sklearn import preprocessing, model_selection, neighbors 
import pandas as pd 

df = pd.read_csv('/Users/shaobinmin/Documents/python/homecodes/ML/breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,5,1,6,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
predicts = clf.predict(example_measures)
print(predicts)