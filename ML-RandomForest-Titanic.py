#ML-RandomForest-Titanic.py
'''
The Random Forest Algorithm
The big idea: Combine a bunch of terrible decision trees into one awesome model.
For each tree in the forest:
Take a bootstrap sample of the data
Randomly select some variables.
For each variable selected, find the split point which minimizes MSE (or Gini Impurity or Information Gain if classification).
Split the data using the variable with the lowest MSE (or other stat).
Repeat step 2 through 4 (randomly selecting new sets of variables at each split) until some stopping condition is satisfied or all the data is exhausted.
Repeat this process to build several trees.
To make a prediction, run an observation down several trees and average the predicted values from all the trees (for regression) or find the most popular class predicted (if classification).
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd

# Import the data
X = pd.read_csv("./Titanic.csv")
y = X.pop("Survived")
# Impute Age with mean
X["Age"].fillna(X.Age.mean(), inplace=True)
# Get just the numeric variables by selecting only the variables that are not "object" datatypes.
numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
X[numeric_variables].head()
'''
###Parameters that will make your model better
n_estimators: The number of trees in the forest. Choose as high of a number as your computer can handle.
It is a good idea to increase n_estimators to a number higher than the default

max_features: The number of features to consider when looking for the best split. Try ["auto", "None", "sqrt", "log2", 0.9, and 0.2]
min_samples_leaf: The minimum number of samples in newly created leaves.Try [1, 2, 3]. If 3 is the best, try higher numbers.

###Parameters that will make it easier to train your model
n_jobs: Determines if multiple processors should be used to train and test the model. Always set this to -1 and %%timeit vs. if it is set to 1. It should be much faster (especially when many trees are trained).
random_state: Set this to 42 if you want to be cool AND want others to be able to replicate your results.
oob_score: THE BEST THING EVER. Random Forest's custom validation method: out-of-bag predictions.
oob_score Always True
'''
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
# I only use numeric_variables because I have yet to dummy out the categorical variables
model.fit(X[numeric_variables], y)
# For regression, the oob_score_ attribute gives the R^2 based on the oob predictions. We want to use c-stat, but I mention this 
# for awareness. By the way, attributes in sklearn that have a trailing underscore are only available after the model has been fit.
print(model.oob_score_) #0.1361695005913669
y_oob = model.oob_prediction_
print("c-stat: ", roc_auc_score(y, y_oob))  #0.73995515504

X.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)
# Change the Cabin variable to be only the first letter or None
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"

X["Cabin"] = X.Cabin.apply(clean_cabin)

categorical_variables = ['Sex', 'Cabin', 'Embarked']
#Change Categorical data to Numeric Data
for variable in categorical_variables:
	# Fill missing data with the word "Missing"
	X[variable].fillna("Missing", inplace=True)
	# Create array of dummies
	dummies = pd.get_dummies(X[variable], prefix=variable)
	# Update X to include dummies and drop the main variable
	X = pd.concat([X, dummies], axis=1)
	X.drop([variable], axis=1, inplace=True)
model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)
print ("C-stat: ", roc_auc_score(y, model.oob_prediction_)) #0.863521128261, btter than before
print(model.feature_importances_)
# Simple version that shows all of the variables
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind="barh", figsize=(7,6));



''' ************************
Just for Fun
To test how parameters have influnce on RandomForest Prediction
'''
#n_jobs
%%timeit
model = RandomForestRegressor(1000, oob_score=True, n_jobs=1, random_state=42)
model.fit(X, y)  #1 loop, best of 3: 1.21 s per loop
%%timeit
model = RandomForestRegressor(1000, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)   #1 loop, best of 3: 708 ms per loop

#n_estimators
results = []
n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000]
for trees in n_estimator_options:
    model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
    model.fit(X, y)
    print (trees, "trees")
    roc = roc_auc_score(y, model.oob_prediction_)
    print ("C-stat: ", roc)
    results.append(roc)
    print ("")    
pd.Series(results, n_estimator_options).plot();  #after 1000, stay the same

#max_features
results = []
max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]
for max_features in max_features_options:
    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
    model.fit(X, y)
    print (max_features, "option")
    roc = roc_auc_score(y, model.oob_prediction_)
    print ("C-stat: ", roc)
    results.append(roc)
    print ("")
pd.Series(results, max_features_options).plot(kind="barh", xlim=(.85,.88));  #auto is the best


#min_samples_leaf
results = []
min_samples_leaf_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for min_samples in min_samples_leaf_options:
    model = RandomForestRegressor(n_estimators=1000, 
                                  oob_score=True, 
                                  n_jobs=-1, 
                                  random_state=42, 
                                  max_features="auto", 
                                  min_samples_leaf=min_samples)
    model.fit(X, y)
    print (min_samples, "min samples")
    roc = roc_auc_score(y, model.oob_prediction_)
    print ("C-stat: ", roc)
    results.append(roc)
    print ("")
pd.Series(results, min_samples_leaf_options).plot();  #5 is the best, but we have to test one by one


'''
To sum up, the best model I found out is 
model = RandomForestRegressor(n_estimators=1000, 
                              oob_score=True, 
                              n_jobs=-1, 
                              random_state=42, 
                              max_features="auto", 
                              min_samples_leaf=5)
model.fit(X, y)
roc = roc_auc_score(y, model.oob_prediction_)
print ("C-stat: ", roc)
'''
