from . import sol2lab, calc_inertia, calc_cost
from amplify import VariableGenerator, solve
from .measure import measure
import time
import numpy as np
from datetime import timedelta

class Base:
    @measure
    def set_BinaryMatrix(self):
        if not hasattr(self, "index2label"):
            gen = VariableGenerator()
            self.model = gen.matrix("Binary", self.n_points)
            self.model.quadratic = self.indexed_qubo
            # self.q = gen.array("Binary", self.n_points)
        else:
            gen = VariableGenerator()
            self.model = gen.matrix("Binary", self.n_clusters*self.n_points)
            self.model.quadratic = self.indexed_qubo
            # self.q = gen.array("Binary", self.n_clusters*self.n_points)
    
    def solve(self, client):
        self.set_BinaryMatrix()
        self.result = solve(self.model, client)
        self.__set_timing('run_annealings', self.result.execution_time.microseconds * 10**-6)
        self.indexed_solution = [v for v in self.result.best.values.values()]
        self.get_labels(self.indexed_solution)
    
    @measure
    def get_labels(self, indexed_solution):
        if not hasattr(self, "index2label"):
            self.labels = indexed_solution
        else:
            solutions = {
                self.index2label[i]:q 
                for i,q in enumerate(indexed_solution)
            }
            self.labels = sol2lab(solution=solutions, n_clusters=self.n_clusters, n_points=self.n_points)
        
    def __set_timing(self, fn, time):
        self.timing[fn] = time if fn not in self.timing else self.timing[fn]+time

    def calc_cost(self, scaling='normal'):
        self.cost = calc_cost(self.data, self.labels, scaling=scaling)
        return self.cost
    
    def calc_inertia(self):
        self.inertia = calc_inertia(self.data, self.labels)
        return self.inertia

    def qubo_to_indexed_qubo(self, given_qubo=None):
        if given_qubo is None:
            indexed_qubo = np.array(
                [
                    [
                        0 if j < i
                        else 0 if (self.index2label[i], self.index2label[j]) not in self.qubo
                        else self.qubo[(self.index2label[i], self.index2label[j])]
                        for j in range(self.n_clusters * self.n_points)
                    ]
                    for i in range(self.n_clusters * self.n_points)
                ]
            )
        else:
            indexed_qubo = np.array(
                [
                    [
                        0 if j < i
                        else 0 if (self.index2label[i], self.index2label[j]) not in given_qubo
                        else given_qubo[(self.index2label[i], self.index2label[j])]
                        for j in range(self.n_clusters * self.n_points)
                    ]
                    for i in range(self.n_clusters * self.n_points)
                ]
            )
        return indexed_qubo