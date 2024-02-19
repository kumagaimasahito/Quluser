from . import ConstrainedClustering
from .utils import measure
from .utils import get_kronecker_qubo_kernel, get_kronecker_pubo_kernel
from .utils.kernel import get_gram_matrix
from .utils.kernel.core import get_gaussian_matrix, get_square_distance_matrix

class KernelClustering(ConstrainedClustering):
    @measure
    def __init__(self, *, data, n_clusters):
        self.data = data
        self.n_points = len(data)
        self.n_clusters = n_clusters
        self.sq_dist = get_square_distance_matrix(data)
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
    def set_gaussian_matrix(self, sigma=0.2, mode="numpy"):
        self.kernel_matrix = get_gaussian_matrix(self.sq_dist, sigma=sigma, mode=mode)

    @measure
    def set_gram_matrix(self, mode="numpy"):
        self.gram_matrix = get_gram_matrix(self.kernel_matrix, self.n_points, mode=mode)

    @measure
    def set_qubo(self, mode="numpy"):
        gram_min = self.gram_matrix.min()
        lam = -2*gram_min
        if mode == "iterance":
            self.qubo = {
                ((i,a),(j,b)) :
                -(self.gram_matrix[i,j]+self.gram_matrix[j,i]) if (i<j and a==b)
                else -(self.gram_matrix[i,j]+lam) if (i==j and a==b)
                else 2*lam if (i==j and a<b)
                else 0
                for i in range(0,self.n_points) 
                for j in range(i,self.n_points)
                for a in range(0,self.n_clusters)
                for b in range(a,self.n_clusters)
            }
        else:
            self.indexed_qubo = get_kronecker_qubo_kernel(self.gram_matrix, lam, self.n_points, self.n_clusters, mode=mode)

    @measure
    def set_pubo(self, mode="numpy"):
        if mode == "iterance":
            self.qubo = {
                ((i,a),(j,a)) : 
                -(self.gram_matrix[i,j]+self.gram_matrix[j,i]) if i<j
                else -self.gram_matrix[i,j]
                for i in range(0,self.n_points) 
                for j in range(i,self.n_points)
                for a in range(0,self.n_clusters)
            }
        else:
            self.indexed_qubo = get_kronecker_pubo_kernel(self.gram_matrix, self.n_clusters, mode=mode)
