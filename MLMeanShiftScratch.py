#MLMeanShiftScratch.py
'''
Make all datapoints centroids
Take mean of all featuresets within centroid's radius, setting this mean as new centroid.
Repeat step #2 until convergence.
'''
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import style
style.use('ggplot')
import numpy as np
import time
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

style.use('ggplot')

X, y = make_blobs(n_samples=150, centers=3, n_features=2)

# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8 ],
#               [8, 8],
#               [1, 0.6],
#               [9,11],
#               [8,2],
#               [10,2],
#               [9,3],])

# plt.scatter(X[:,0], X[:,1], s=150)
# plt.show()


def SKLearn_fit(X):
	bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=150)
	print(bandwidth)
	ms = MeanShift(bandwidth = bandwidth, bin_seeding = True)
	ms.fit(X)
	labels = ms.labels_
	cluster_centers = ms.cluster_centers_
	labels_unique = np.unique(labels)
	#number of estimated clusters
	n_clusters = len(labels_unique)
	print('estimated', cluster_centers, 'numbers', n_clusters)
	return cluster_centers


class Mean_Shift(object):
	"""docstring for Mean_Shift"""
	def __init__(self, radius = None, radius_norm_step = 100):
		super(Mean_Shift, self).__init__()
		self.radius = radius
		self.radius_norm_step = radius_norm_step

	def fit(self, data):
		if self.radius == None:
			all_data_centroid = np.average(data, axis = 0)
			#所有点的中心到原点距离
			all_data_norm = np.linalg.norm(all_data_centroid)
			self.radius = all_data_norm / self.radius_norm_step

		centroids = {}

		for i in range(len(data)):
			centroids[i] = data[i]

		weights = [i for i in range(self.radius_norm_step)][::-1]

		while True:
			new_centroids = []
			for i in centroids:
				in_bandwidth = []
				centroid = centroids[i]

				for features in data:
					distances = np.linalg.norm(features - centroid)

					#未知radius情况，需要自己optimize
					if distances == 0:
						distances = 0.000000001
					#以中心点为标准，分类features
					weight_index = int(distances / self.radius)

					if weight_index > self.radius_norm_step - 1:
						weight_index = self.radius_norm_step - 1
					#根据离centroid[i]的距离来确定分类的点/以及数量
					#离centroid[i]越近的点，权重越大，算average时候，更靠近
					#to_add = (weights[weight_index])*[features]
					to_add = (weights[weight_index]**2)*[features]#更精准
					#print(to_add, 'gdfgdfshfgh', len(to_add), 'fffffffff', weights[weight_index])	
					in_bandwidth += to_add
					'''已知radius的情况
					if distances < self.radius:
						in_bandwidth.append(features)
					'''
				#每一个点都找到已为中心的bandwidth，然后计算新的centroid
				new_centroid = np.average(in_bandwidth, axis = 0)
				#将计算出来的新centroid放在一起
				new_centroids.append(tuple(new_centroid))

			#将所有点得出来的centroid进行Set，找出独一的
			unique = sorted(list(set(new_centroids)))
			to_pop = []
			for i in unique:
				for ii in [i for i in unique]:
					if i == ii:
						pass
					elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
						#print(np.array(i), np.array(ii), 'sgdsfgh', self.radius)
						#time.sleep(5)
						to_pop.append(ii)
						break
			for i in to_pop:
				try:
					unique.remove(i)
				except Exception as e:
					pass
				
			pre_centroids = dict(centroids)
			centroids = {}
			for i in range(len(unique)):
				centroids[i] = np.array(unique[i])

			optimized = True
			for i in centroids:
				if not np.array_equal(centroids[i], pre_centroids[i]):
					optimized = False
				if not optimized:
					break
			if optimized:
				break
		self.centroids = centroids	

		#classification
		self.classifications = {}
		for i in range(len(self.centroids)):
			self.classifications[i] = []
		for featureset in data:
			#compare distnace to either centroid
			distances = [np.linalg.norm(features - centroids[cen]) for cen in centroids]
			classification = distances.index(min(distances))
			self.classifications[classification].append(featureset)

	def predict(self, data):
		distances = [np.linalg.norm(features - centroids[cen]) for cen in centroids]
		classification = distances.index(min(distances))
		return classification

clf = Mean_Shift()
cluster_centers = SKLearn_fit(X)
clf.fit(X)

colors = 10*["b","m","g","c"]
centroids = clf.centroids
for classification in clf.classifications:
	color = colors[classification]
	print(color,'the color is ', len(clf.classifications))
	for featureset in clf.classifications[classification]:
		plt.scatter(featureset[0],featureset[1], marker = "x", color=color, s=70, linewidths = 5, zorder = 10)

for c in centroids:
	plt.scatter(centroids[c][0], centroids[c][1], color = 'k', marker = "*", s =150)

for cen in cluster_centers:
	plt.scatter(cen[0], cen[1], color = 'r', marker = 'o', s = 100)
plt.show()
		
