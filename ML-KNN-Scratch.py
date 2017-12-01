#selfkNN.py
#ML-KNN-Scratch
import pandas as pd 
import numpy as np 
import time, random, warnings
from sklearn import preprocessing, model_selection, neighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
from math import sqrt
from collections import Counter
from scipy.spatial import distance
from sklearn.cluster import KMeans

style.use('fivethirtyeight')

def getData():
	df = open('datingTestSet.txt')
	features = []#np.zeros((len(df),3)) 
	labels = []
	fullData = []
	for line in df.readlines():
		line = line.strip().split('\t')
		# print(line)
		features.append(line[0:3])
		labels.append(line[-1])
		fullData.append(line)
		# print(features)
		# time.sleep(5)
		# print(labels)
	return features, labels, fullData

#sklearn
def sklearnMethod(X, y):
#test model, print accuracy
	x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
	clf = neighbors.KNeighborsClassifier()
	clf.fit(x_train, y_train)
	accuracy = clf.score(x_test, y_test)
	print(accuracy)

	kMeansMethod(X, y)
	#plot3D(x_train, x_test, y_train, y_test)
	#predict 
	# example_measures = np.array([32534, 10.4534, 0.5345])
	# example_measures = example_measures.reshape(1, -1)
	# y_pre = clf.predict(example_measures)
	# print(y_pre)

	#自己写的整理Data方法
	#sameSeperation(x_train, x_test, y_train, y_test)

def kMeansMethod(X, y):
	clf = KMeans(n_clusters = 3)
	clf.fit(X)
	#handle non-numeric data
	y_num = y
	unique = set(y)
	iteration = 0
	for i in unique:
		for j in range(len(y)):
			if y[j] == i:
				y_num[j] = iteration
		iteration += 0
	#calculate accucary
	correct = 0
	for i in range(len(X)):
		pre = np.array(X[i]).astype(float)
		pre = pre.reshape(-1, len(pre))
		prediction = clf.predict(pre)
		if prediction == y_num[i]:
			correct+=1
	print(correct/len(X), 'for kmeans')




#画3D图
def plot3D(x_train, x_test, y_train, y_test):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, projection='3d')
	train_set , test_set = sameSeperation(x_train, x_test, y_train, y_test)
	
	xs = []
	ys = []
	zs = []
	for i in train_set['largeDoses']:
		xs.append(float(i[0]))
		ys.append(float(i[1]))
		zs.append(float(i[2]))
	ax.scatter(xs, ys, zs, s=20, c='r', depthshade=True)
	for i in train_set['smallDoses']:
		xs.append(float(i[0]))
		ys.append(float(i[1]))
		zs.append(float(i[2]))
	ax.scatter(xs, ys, zs, s=20, c='b', depthshade=True)
	for i in train_set['didntLike']:
		xs.append(float(i[0]))
		ys.append(float(i[1]))
		zs.append(float(i[2]))
	ax.scatter(xs, ys, zs, s=20, c='g', depthshade=True)
	plt.show()





def sameSeperation(x_train, x_test, y_train, y_test, k = 5):
	train_set = {'largeDoses':[], 'smallDoses':[], 'didntLike':[]}
	test_set = {'largeDoses':[], 'smallDoses':[], 'didntLike':[]}

	lentrain = len(x_train)
	lentest = len(x_test)
	#creating dataset looks like {'largeDoses': [], 'smallDoses': [], 'didntLike': []}
	for i in range(lentrain):
		train_set[ y_train[i] ].append( x_train[i] )
		#print('rrrrrr', train_set)
	for i in range(lentest):
		test_set[ y_test[i]].append( x_test[i])
		# print('dsfsggf', test_set)
		# time.sleep(3)

	correct = 0
	total = 0
	for group in test_set:
		for predicts in test_set[group]:
			#每一个test里面数据，like ['38343', '7.241614', '1.661627'] 与 train数据算距离，得出分类
			vote, confidence = k_nearest_neighbors(train_set, predicts)
			if group == vote:
				correct += 1
			total += 1
	print('Same Train and Test, Accuracy', correct/total)
	return train_set, test_set


#自己实现
def seperateData(fullData, k):
	random.shuffle(fullData)
	test_size = 0.2
	train_set = {'largeDoses':[], 'smallDoses':[], 'didntLike':[]}
	test_set = {'largeDoses':[], 'smallDoses':[], 'didntLike':[]}

	train_data = fullData[:-int(test_size * len(fullData))]
	test_data = fullData[-int(test_size * len(fullData)):]
	#creating dataset looks like {'largeDoses': [], 'smallDoses': [], 'didntLike': []}
	for i in train_data:
		train_set[i[-1]].append(i[:-1])
		#print('rrrrrr', train_set)
	for i in test_data:
		test_set[i[-1]].append(i[:-1])
		#print('dsfsggf', test_set)

	correct = 0
	total = 0
	for group in test_set:
		for predicts in test_set[group]:
			#每一个test里面数据，like ['38343', '7.241614', '1.661627'] 与 train数据算距离，得出分类
			vote, confidence = k_nearest_neighbors(train_set, predicts)
			if group == vote:
				correct += 1
			total += 1
	print('Accuracy', correct/total)
	

def k_nearest_neighbors(data, predict, k = 5):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups!')

	distances = []
	for group in data:
		for features in data[group]:
			euclidean_diatance = np.linalg.norm(np.array(features, dtype = float) - np.array(predict, dtype = float))
			distances.append([euclidean_diatance, group])
			
	votes = [i[1] for i in sorted(distances)[:k]]
	vote_result = Counter(votes).most_common(1)[0][0]
	#Counter(votes).most_common(1) looks like this [('smallDoses', 4)]
	# 4/k is confidence
	confidence = Counter(votes).most_common(1)[0][1] / k
	return vote_result, confidence

if __name__ == '__main__':
	X, y, fullData = getData()
	sklearnMethod(X, y)
	#seperateData(fullData, 5)

	
