from scipy.spatial import distance
from . import min_max

def cost(data, label, scaling='normal'):
	dist = distance.squareform(distance.pdist(data))
	if scaling == 'normal':
		dist = min_max(dist)

	leng = len(data)
	cost = sum(
		[
			dist[i,j]
			for i in range(0,leng)
			for j in range(i+1,leng)
			if label[i] == label[j]
		]
	)

	return cost