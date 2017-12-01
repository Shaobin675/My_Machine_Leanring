# K-Means from Scratch in Python
'''
1.Choose value for K
2.Randomly select K featuresets to start as your centroids
3.Calculate distance of all other featuresets to centroids
4.Classify other featuresets as same as closest centroid
5.Take mean of each class (mean of all featuresets by class), making that mean the new centroid
6.Repeat steps 3-5 until optimized (centroids no longer moving)
'''
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import preprocessing
import matplotlib.pyplot as plt
style.use('ggplot')
import time

#data
# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11]])
#plot
# plt.scatter(X[:,0], X[:,1], color = 'b')
# plt.show()
# class KMeans
class KMeansClass:
	#init()
	def __init__(self, k=5, thre = 0.001, max_iter = 300):
		self.k = k
		self.thre = thre
		self.max_iter = max_iter

	#fit method
	def fit(self, data):
		#initiate classifications dict
		self.centroids = {}
		for i in range(self.k):#随机放两个点到centriod中
			self.centroids[i] = data[i] 
			print('original centroid', data[i])
		#max_iter iteration
		for i in range(self.max_iter):
			self.classifications = {}
			for j in range(self.k):
				self.classifications[j] = []
			for feature in data: #根据centroid分类
				distances = [np.linalg.norm(feature - self.centroids[centroid]) for centroid in self.centroids]
				classification = distances.index(min(distances))
				self.classifications[classification].append(feature)
				#print('第几个classification', classification, 'data', feature)
			
			# Take mean of each class (mean of all featuresets by class), making that mean the new centroid
			pre_centroids = dict(self.centroids)
			#更新centroid，如果变化小于0.001，就break
			for classification in self.classifications:
				self.centroids[classification] = np.average(self.classifications[classification], axis = 0)
			print('updated centroid', self.centroids)
			time.sleep(10)
			
			optimized = True
			for c in self.centroids:
				original_centroid = pre_centroids[c]
				current_centroid = self.centroids[c]
				if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.thre:
					print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
					optimized = False
			if optimized:
				break;

	#predict method
	def predict(self, data):
		distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
		classification = distances.index(min(distances))
		return classification

	def loadData(self, drect):
		df = pd.read_excel(drect)
		df.drop(['body'], 1, inplace=True)
		df.fillna(-99999, inplace = True)

		columns = df.columns.values
		for column in columns:
			text_digit_vals = {}
			#pd.to_numeric(df[column], downcast = 'float', errors = 'ignore')
			def convert_to_int(val):
				return text_digit_vals[val]
			if df[column].dtype != np.int64 and df[column].dtype != np.float64:
				elements = df[column]
				unique = set(elements)
				x = 0
				for uni in unique:
					if uni not in text_digit_vals:
						text_digit_vals[uni] = x
					x += 1
				# now we map the new "id" vlaue
            	# to replace the string.
				df[column] = list(map(convert_to_int, df[column]))
		return df

#run KMeans
clf = KMeansClass()
#sklearn
df = clf.loadData('titanic.xls')

X = np.array(df.drop(['survived'], 1).astype(float)) 
X = preprocessing.scale(X)
y = np.array(df['survived'])

#from skrech
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

#show plot with prediction
# for centroid in clf.centroids:
# 	plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], 
# 		marker = 'o', color = 'k', s = 150, linewidths = 5)
# colors = 10*["g","r","c","b","k"]
# for classification in clf.classifications:
# 	color = colors[classification]
# 	for feature in clf.classifications[classification]:
# 		plt.scatter(feature[0], feature[1],
# 			marker = 'x', color = color, s = 150, linewidths = 5)
# plt.show()
