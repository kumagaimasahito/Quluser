from pyqubo import Array
from scipy.spatial import distance
import numpy as np
from .utils import min_max, standardization, Base, pyqubo_to_qubo, measure

class BinaryClustering(Base):
    @measure
    def __init__(self, *, data, metric='euclidean'):
        self.data = data
        self.n_points = len(data)
        self.dist = distance.cdist(data, data, metric=metric)
        self.timing = {}

    @measure
    def set_pyqubo(self, scaling="invalid"):
        dist = self._scale_dist(scaling=scaling)
        
        spin = Array.create('spin', shape=self.n_points, vartype='SPIN')
        H = 0.5 * sum(dist[i,j] * spin[i] * spin[j] for i in range(self.n_points) for j in range(self.leng))
        model = H.compile()
        qubo_by_pyqubo, _ = model.to_qubo()
        self.indexed_qubo = self.pyqubo_to_qubo(qubo_by_pyqubo)

    @measure
    def set_qubo(self, scaling="invalid", mode="numpy"):
        dist = self._scale_dist(scaling=scaling)

        if mode == "iterance":
            self.indexed_qubo = np.array([
                [
                    4*dist[i,j] if (i<j)
                    else (-2)*sum(
                        [
                            dist[i,k] 
                            for k in range(0,self.n_points)
                        ]
                    ) if i==j
                    else 0
                    for j in range(0,self.n_points) 
                ]
                for i in range(0,self.n_points)
            ])
        else:
            diag = np.diag(np.sum(dist, axis=0))
            triu = np.triu(dist, k=1)
            self.indexed_qubo = -2 * diag + 4 * triu 

    def _scale_dist(self, scaling=None):
        dist = self.dist
        if scaling == 'nomal' or None:
            dist = min_max(dist)
        elif scaling == 'divbymax':
            dist = dist/dist.max()
        elif scaling == 'normal_i':
            dist = min_max(dist, axis=1)
        elif scaling == 'standard':
            dist = standardization(dist)
        elif scaling == 'invalid':
            pass
        else :
            print("scaling failed.")
        return dist