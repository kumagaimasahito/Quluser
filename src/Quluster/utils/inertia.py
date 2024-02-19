import numpy as np
import collections
from scipy.spatial import distance_matrix, distance

def get_center(data, label):
	label_clusters = list(collections.Counter(label).keys())
	
	label2center = {Ca:
		np.array(
			[
				data[i]
				for i,Ci in enumerate(label)
				if Ca == Ci
			]
		).mean(axis=0)
		for Ca in label_clusters
	}

	return label_clusters, label2center

def inertia(data, label):
	label_clusters, label2center = get_center(data, label)
	dist = distance.cdist(data, list(label2center.values()))
	leng = len(data)	

	sse = sum(
		[
			dist[j,i]**2
			for i,Ci in enumerate(label_clusters)
			for j in range(leng)
			if label[j] == Ci
		]
	)

	return sse