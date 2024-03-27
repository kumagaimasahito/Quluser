from pyqubo import Array, Constraint
from scipy.spatial import distance
from .utils import min_max, standardization, Base, pyqubo_to_qubo, measure
from .utils import get_kronecker_qubo_euclidean
# from .utils import get_kronecker_pubo_euclidean

class CombinatorialClustering(Base):
    @measure
    def __init__(self, *, data, n_clusters, metric='euclidean'):
        self.data = data
        self.n_points = len(data)
        self.n_clusters = n_clusters
        self.dist = distance.cdist(data, data, metric=metric)
        self.lam = self.n_points - n_clusters
        self.label2index = {
            (i,a):i*n_clusters+a 
            for i in range(self.n_points) 
            for a in range(n_clusters)
        }
        self.index2label = {
            i*n_clusters+a:(i,a) 
            for i in range(self.n_points) 
            for a in range(n_clusters)
        }
        self.timing = {}

    @measure
    def set_pyqubo(self, scaling="normal", lam_rate = 1):
        self.lam *= lam_rate
        dist, M = self._scale_dist(scaling=scaling)

        qbit = Array.create('qbit', shape=(self.n_points, self.n_clusters), vartype='BINARY')
        H = 0.5 * sum(dist[i,j] * qbit[i,a] * qbit[j,a] for i in range(self.n_points) for j in range(self.n_points) for a in range(self.n_clusters)) + M * Constraint(sum((sum(qbit[i,a] for a in range(self.n_clusters)) - 1) ** 2 for i in range(self.n_points)), label='ONE_HOT')
        model = H.compile()
        qubo_by_pyqubo, _ = model.to_qubo()
        self.qubo = pyqubo_to_qubo(qubo_by_pyqubo)
        self.indexed_qubo = self.qubo_to_indexed_qubo()

    @measure
    def set_qubo(self, scaling="normal", lam_rate=1, mode="numpy"):
        self.lam *= lam_rate
        dist, M = self._scale_dist(scaling=scaling)
        if mode == "iterance":
            self.qubo = {
                ((i,a),(j,b)) :
                dist[i,j] if (i<j and a==b)
                else (-1)*M if (i==j and a==b)
                else 2*M if (i==j and a<b)
                else 0
                for i in range(0,self.n_points) 
                for j in range(i,self.n_points)
                for a in range(0,self.n_clusters)
                for b in range(a,self.n_clusters)
            }
            self.qubo = self._omit_zero_coefficients(self.qubo) # 非ゼロ要素だけのQUBO
            self.indexed_qubo = self.qubo_to_indexed_qubo()
        else:
            self.indexed_qubo = get_kronecker_qubo_euclidean(dist, M, self.n_points, self.n_clusters)

    def _scale_dist(self, scaling):
        M = self.lam
        dist = self.dist
        if scaling == 'normal':
            dist = min_max(dist)
        elif scaling == 'divbymax':
            dist = dist/dist.max()
        elif scaling == 'normal_i':
            dist = min_max(dist, axis=1)
        elif scaling == 'standard':
            dist = standardization(dist)
        elif scaling == 'invalid':
            M *= dist.max()
        else :
            print("scaling failed.")
        return dist, M
        
    def _omit_zero_coefficients(self, qb):
        return {k: v for k, v in qb.items() if v != 0}