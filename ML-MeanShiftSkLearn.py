#ML-MeanShiftSkLearn.py

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
import time
from sklearn import preprocessing, model_selection
from sklearn.cluster import KMeans, MeanShift

#load data
df = pd.read_excel('./data/titanic.xls')
original_df = pd.DataFrame.copy(df)
#remove NAN, drop meaningless columns
df.drop(['body'], 1, inplace= True)
#pd.convert_objects(df, downcast='float', errors = 'ignore')
df.fillna(0, inplace = True )
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
df.drop(['ticket','home.dest'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float)) 
X = preprocessing.scale(X)
y = np.array(df['survived'])

#create clf, train
clf = MeanShift()
clf.fit(X)
labels = clf.labels_ #0,1,2,3
cluster_centers = clf.cluster_centers_
#print(np.unique(labels), 'fsdfagfs,', cluster_centers)
original_df['cluster_group'] = np.nan
for i in range(len(X)):
	#original_df['cluster_group'].iloc[i] cluster_group的第i行
	#把clf.fit的classification放在cluster_group里面
	original_df['cluster_group'].iloc[i] = labels[i]
	#print(original_df)
	

n_clusters_ = len(np.unique(labels))
survival_rates = {}
for i in range(n_clusters_):
	#temp_df按照cluster_group把数据分离出来
	temp_df = original_df[(original_df['cluster_group'] == float(i))]
	#print(temp_df)
	survival_cluster = temp_df[(temp_df['survived'] == 1)]
	survival_rate = len(survival_cluster) / len(temp_df)
	survival_rates[i] = survival_rate
	
print(survival_rates)
print(original_df[ (original_df['cluster_group']==1) ])
