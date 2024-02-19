from pyqubo import Array, Constraint
from scipy.spatial import distance
from .utils import min_max, standardization, BaseSolver, pyqubo_to_qubo, measure

class BinaryClustering(BaseSolver):
    @measure
    def __init__(self, *, data, metric='euclidean'):
        self.data = data
        self.leng = len(data)
        self.dist = distance.cdist(data, data, metric=metric)
        self.timing = {}

    @measure
    def set_pyqubo(self, scaling="invalid"):
        dist = self._scale_dist(scaling=scaling)
        
        spin = Array.create('spin', shape=self.leng, vartype='SPIN')
        H = 0.5 * sum(dist[i,j] * spin[i] * spin[j] for i in range(self.leng) for j in range(self.leng))
        model = H.compile()
        self.qubo, _ = model.to_qubo()
        self.qubo = pyqubo_to_qubo(self.qubo)
        return self.qubo

    @measure
    def set_qubo(self, scaling="invalid"):
        dist = self._scale_dist(scaling=scaling)

        self.qubo = {
            (i,j) :
            4*dist[i,j] if (i<j)
            else (-2)*sum(
                [
                    dist[i,k] 
                    for k in range(0,self.leng)
                ]
            )
            for i in range(0,self.leng) 
            for j in range(i,self.leng)
        }
        return self.qubo

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