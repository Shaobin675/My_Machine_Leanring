#ML-v2.py
import pandas as pd 
import quandl, math, datetime
import numpy as np 
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
from matplotlib import style
import pickle #classifier, avoid training

style.use('ggplot')

df = pd.DataFrame()
df = quandl.get('WIKI/GOOGL') 
#print(df)
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Low'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']] 
#print(df.head())
df.fillna('-99999', inplace = True)
forecast_col = 'Adj. Close'
forecast_out  = int(math.ceil(0.01*len(df))) 
#往上挪forecast_out行
df['label'] = df[forecast_col].shift(-forecast_out)


print(df.tail())
#print(df['Adj. Close'][forecast_out:forecast_out+5])

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace = True)
y = np.array(df['label'])
print(df.tail())

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
'''
for k in ['linear', 'poly', 'rbf', 'sigmoid']:
	clf = svm.SVR(kernel = k)
	clf.fit(X_train, y_train)
	acurracy = clf.score(X_test, y_test)
	print(acurracy)
'''
clf = LinearRegression(n_jobs = 10) 
#clf = svm.SVR()
clf.fit(X_train, y_train)

#create a pickle to save classifier
with open('LinearRegression.pickle', 'wb') as f:
	pickle.dump(clf, f)

#open a pickle
pickleOpen = open('LinearRegression.pickle', 'rb')
clf = pickle.load(pickleOpen)

#predict test dataset
acurracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
#print(forecast_set, acurracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #60x60x24 seconds
next_unix = last_unix + one_day
for i in forecast_set:
	next_date = datetime.datetime.fromtimestamp(next_unix)
	next_unix += one_day
	df.loc[next_date] = [np.nan for _ in range(len(df.columns) -1)] + [i] #i is the df['Forecast']

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc=4)
plt.show()
