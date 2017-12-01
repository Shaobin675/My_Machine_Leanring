#ML-RandomForest.py
'''
Pros
Automatically model non-linear relations and interactions between variables. Perfect collinearity doesn't matter.
Easy to tune
Relatively easy to understand everything about them
Flexible enough to handle regression and classification tasks
Is useful as a step in exploratory data analysis
Can handle high dimensional data
Have a built in method of checking to see model accuracy
In general, beats most models at most prediction tasks
'''
import numpy as np
# Load the Boston Housing dataset
from sklearn.datasets import load_boston
# Make train and test datasets
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

X, y = load_boston().data, load_boston().target
np.random.seed(100)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)

#LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print ("R^2:", model.score(X_test, y_test).round(2))

#DecisionTree
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
print ("R^2:", model.score(X_test, y_test).round(2))

#RandomForest
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print ("R^2:", model.score(X_test, y_test).round(2))
