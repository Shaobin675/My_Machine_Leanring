#ML-KMeansSKLearn-TitanicData.py
#how do people transfer text data to numeric data?

'''Titanic.xls columns meaning
#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''
import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans

#load data
df = pd.read_excel('titanic.xls')

#remove NAN, drop meaningless columns
df.drop(['body'], 1, inplace= True)
pd.convert_objects(df, downcast='float', errors = 'ignore')
df.fillna(-99999, inplace = True )
#print(df.head())


#handle non-numeric data, convert text to numeric data
def texttoNumeric():
	columns = df.columns.values
	for column in columns:
		text_digit_vals = {}
		def convert_to_int(val):
			return text_digit_vals[val]
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in text_digit_vals:
					#print(unique, x) 每个column都从0开始计数，每个不同string都有不同数值
					text_digit_vals[unique] = x
				x += 1
			# now we map the new "id" vlaue
           	# to replace the string.
			df[column] = list(map(convert_to_int, df[column]))		
	return df

df = texttoNumeric()
X = np.array(df.drop(['survived'], 1).astype(float)) 
X = preprocessing.scale(X)
y = np.array(df['survived'])

#create clf, train
clf = KMeans(n_clusters = 2)
clf.fit(X)
#test, predict, accuracy
length = len(X)
correct = 0
for i in range(length):
	pre = np.array(X[i].astype(float))
	pre = pre.reshape(-1, len(pre))
	prediction = clf.predict(pre)
	if prediction == y[i]:
		correct += 1

print('accuracy', correct/length)

