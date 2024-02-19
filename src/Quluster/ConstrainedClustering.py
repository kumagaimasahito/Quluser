from . import CombinatorialClustering
from .utils import measure

class ConstrainedClustering(CombinatorialClustering):
    @measure
    def add_must_link_to_qubo(self, must_link_instances, weight_must_link=None):
        """
        For example, must_link_instances is [(0, 1), (4, 6), (5, 9), ... , (20, 35)] as a list.
        """
        self.weight_must_link = self.lam if weight_must_link == None else weight_must_link
        self.must_link_qubo = {
            ((i, k), (i, k)) if inc == 0 else ((j, k), (j, k)) if inc == 1 else ((i, k), (j, k)) : 
            self.weight_must_link if inc != 2 else 2*self.weight_must_link
            for k in range(self.n_clusters) 
            for (i, j) in must_link_instances
            for inc in range(3)
        }
        self.qubo = self._merge_dicts(self.qubo, self.must_link_qubo)

    @measure
    def add_cannot_link_to_qubo(self, cannot_link_instances, weight_cannot_link=None):
        self.weight_cannot_link = self.lam if weight_cannot_link == None else weight_cannot_link
        self.cannot_link_qubo = {
            ((i, k), (j, k)) : self.weight_cannot_link
            for k in range(self.n_clusters) 
            for (i, j) in cannot_link_instances
        }
        self.qubo = self._merge_dicts(self.qubo, self.cannot_link_qubo)

    @measure
    def add_partition_level_to_qubo(self, partition_level_instances, weight_partition_level=None):
        self.weight_partition_level = self.lam if weight_partition_level == None else weight_partition_level
        self.partition_level_qubo = {
            ((i, k), (i, k)) : self.weight_partition_level
            for k in range(self.n_clusters)
            for m, instances in partition_level_instances.items()
            for i in instances
            if k != m
        }
        self.qubo = self._merge_dicts(self.qubo, self.partition_level_qubo)

    @measure
    def add_non_partition_level_to_qubo(self, non_partition_level_instances, weight_non_partition_level=None):
        self.weight_non_partition_level = self.lam if weight_non_partition_level == None else weight_non_partition_level
        self.non_partition_level_qubo = {
            ((i, m), (i, m)) : self.weight_non_partition_level
            for m, instances in non_partition_level_instances.items()
            for i in instances
        }
        self.qubo = self._merge_dicts(self.qubo, self.non_partition_level_qubo)
    
    @measure
    def add_balanced_sizes_to_qubo(self, weight_balanced_size=None):
        self.weight_balanced_size = self.lam if weight_balanced_size == None else weight_balanced_size
        self.balanced_sizes_qubo = {
            ((i, k), (j, k)) : self.weight_balanced_size*(1-2*self.n_points/self.n_clusters) if i == j else 2*self.weight_balanced_size
            for i in range(self.n_points)
            for j in range(i, self.n_points)
            for k in range(self.n_clusters)
        }
        self.qubo = self._merge_dicts(self.qubo, self.balanced_sizes_qubo)

    @measure
    def add_limited_sizes_to_qubo(self, limited_sizes, weight_limited_sizes=None):
        self.weight_limited_sizes = self.lam if weight_limited_sizes == None else weight_limited_sizes
        self.limited_sizes_qubo = {
            ((i, k), (j, k)) : self.weight_limited_sizes*(1-2*limited_sizes) if i == j else 2*self.weight_limited_sizes
            for i in range(self.n_points)
            for j in range(i, self.n_points)
            for k in range(self.n_clusters)
        }
        self.qubo = self._merge_dicts(self.qubo, self.limited_sizes_qubo)

    def _merge_dicts(self, a, b, func=lambda x, y: x+y):
        d1 = a.copy()
        d2 = b.copy()
        d1 = {
            k: func(d2[k], v) if k in d2 else v 
            for k, v in d1.items()
        }
        d2.update(d1)
        return d2