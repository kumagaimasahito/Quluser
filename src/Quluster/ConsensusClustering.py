from .utils import BaseSolver, measure

class ConsensusClustering(BaseSolver):
    @measure
    def __init__(self, *, clusterings, n_clusters):
        self.clusterings = clusterings
        self.n_points = len(clusterings[0])
        self.n_clusterings = len(clusterings)
        self.n_clusters = n_clusters
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
        self.similarity_matrix = {
            (i,j) : sum([
                1 for labels_m in self.clusterings if labels_m[i]==labels_m[j]
            ]) / self.n_clusterings
            for i in range(0,self.n_points)
            for j in range(i,self.n_points)
        }
        self.timing = {}

    @measure
    def set_qubo(self, model=None, lam_rate=1):
        self.lam *= lam_rate

        if model=="pairwise_similarity-based":
            self.qubo = {
                ((i,a),(j,b)) :
                1-self.similarity_matrix[(i,j)] if (i<j and a==b)
                else (-1)*self.lam if (i==j and a==b)
                else 2*self.lam if (i==j and a<b)
                else 0
                for i in range(0,self.n_points) 
                for j in range(i,self.n_points)
                for a in range(0,self.n_clusters)
                for b in range(a,self.n_clusters)
            }
            return self.qubo
        elif model=="partition_difference":
            self.qubo = {
                ((i,a),(j,b)) :
                1-self.similarity_matrix[(i,j)] if (i<j and a==b)
                else self.similarity_matrix[(i,j)] if  (i<j and a<b)
                else (-1)*self.lam if (i==j and a==b)
                else 2*self.lam if (i==j and a<b)
                else 0
                for i in range(0,self.n_points) 
                for j in range(i,self.n_points)
                for a in range(0,self.n_clusters)
                for b in range(a,self.n_clusters)
            }
            return self.qubo
        else:
            print("Please select the Ising model of pairwise_similarity-based or partition_difference")

    @measure
    def set_pubo(self, model):
        if model=="pairwise_similarity-based":
            self.qubo = {
                ((i,a),(j,a)) : 1-self.similarity_matrix[(i,j)]
                for i in range(0,self.n_points-1) 
                for j in range(i+1,self.n_points)
                for a in range(0,self.n_clusters)
            }
            return self.qubo
        elif model=="partition_difference":
            self.qubo = {
                ((i,a),(j,b)) :
                1-self.similarity_matrix[(i,j)] if (i<j and a==b)
                else self.similarity_matrix[(i,j)] if  (i<j and a<b)
                else 0
                for i in range(0,self.n_points-1) 
                for j in range(i+1,self.n_points)
                for a in range(0,self.n_clusters)
                for b in range(a,self.n_clusters)
            }
            return self.qubo
        else:
            print("Please select the Ising model of pairwise_similarity-based or partition_difference")
