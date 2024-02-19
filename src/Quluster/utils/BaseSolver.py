from . import sol2lab, calc_inertia, calc_cost
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import openjij as oj
from neal import SimulatedAnnealingSampler
from dwave_qbsolv import QBSolv
from dwave.system import LeapHybridSampler
from amplify import BinaryMatrix, Solver
from amplify.client import FixstarsClient, HitachiClient
from .measure import measure
import time
import numpy as np

class BaseSolver:
    """
    Solvers using QUBO to obtain solutions and labels
    """
    def DWaveQuantumAnnealer(self, token, endpoint='https://cloud.dwavesys.com/sapi', solver='Advantage_system4.1', settings={}, parameters={}):
        sampler = EmbeddingComposite(DWaveSampler(endpoint=endpoint, token=token, solver=solver, **settings))
        type_sampler = 'DWaveSampler'
        return self.__DWaveBaseSolver(solver=sampler, type_sampler=type_sampler, **parameters)

    def DWaveSimulatedAnnealer(self, settings={}, parameters={}):
        sampler = SimulatedAnnealingSampler(**settings)
        type_sampler = 'SimulatedAnnealingSampler'
        return self.__DWaveBaseSolver(solver=sampler, type_sampler=type_sampler, **parameters)

    def LeapHybridSolver(self, token, endpoint='https://cloud.dwavesys.com/sapi', settings={}, parameters={}):
        sampler = LeapHybridSampler(endpoint=endpoint, token=token, **settings)
        type_sampler = 'LeapHybridSampler'
        return self.__DWaveBaseSolver(solver=sampler, type_sampler=type_sampler, **parameters)

    def __DWaveBaseSolver(self, solver, type_sampler, **parameters):
        if type_sampler == 'SimulatedAnnealingSampler':
            start_time = time.perf_counter()
            self.response = solver.sample_qubo(self.qubo, **parameters)
            end_time = time.perf_counter()
            self.__set_timing('run_annealings', end_time - start_time)
            return self.get_labels(list(self.response.first.sample.values()))
        self.response = solver.sample_qubo(self.qubo, **parameters)
        self.info_dwave = self.response.info
        if type_sampler == 'LeapHybridSampler':
            self.__set_timing('run_annealings', self.info_dwave['run_time']*10**(-6)) # Only for LeapHybridSampler
        elif type_sampler == 'DWaveSampler':
            # self.__set_timing('run_annealings', self.info_dwave['timing']['total_real_time']*10**(-6)) # Only for DWaveSampler
            self.__set_timing('run_annealings', self.info_dwave['timing']['qpu_access_time']*10**(-6)) # Only for DWaveSampler
        return self.get_labels(list(self.response.first.sample.values()))

    def QBSolvQuantumAnnealer(self, token, endpoint='https://cloud.dwavesys.com/sapi', solver='Advantage_system4.1', settings={}, parameters={}):
        sampler = EmbeddingComposite(DWaveSampler(endpoint=endpoint, token=token, solver=solver, **settings))
        return self.__QBSolvBaseSolver(solver=sampler, **parameters)

    def QBSolvSimulatedAnnealer(self, settings={}, parameters={}):
        sampler = SimulatedAnnealingSampler(**settings)
        return self.__QBSolvBaseSolver(solver=sampler, **parameters)

    def QBSolvTabuSearchSolver(self, **parameters):
        return self.__QBSolvBaseSolver(solver=None, **parameters)

    def __QBSolvBaseSolver(self, solver, **parameters):
        start_time = time.perf_counter()
        self.response = QBSolv().sample_qubo(self.qubo, solver=solver, **parameters)
        end_time = time.perf_counter()
        self.__set_timing('run_annealings', end_time - start_time)
        return self.get_labels(list(self.response.first.sample.values()))

    def OpenJijSimulatedAnnealer(self, settings={}, parameters={}):
        sampler = oj.SASampler(**settings)
        return self.__OpenJijBaseSolver(solver=sampler, **parameters)

    def OpenJijSimulatedQuantumAnnealer(self, settings={}, parameters={}):
        sampler = oj.SQASampler(**settings)
        return self.__OpenJijBaseSolver(solver=sampler, **parameters)

    def OpenJijContinuousTimeSimulatedQuantumAnnealer(self, settings={}, parameters={}):
        sampler = oj.CSQASampler(**settings)
        return self.__OpenJijBaseSolver(solver=sampler, **parameters)

    def __OpenJijBaseSolver(self, solver, **parameters):
        self.response = solver.sample_qubo(self.qubo, **parameters)
        self.info_openjij = self.response.info
        self.__set_timing('run_annealings', sum(self.info_openjij['list_exec_times'])*10**(-6))
        return self.get_labels(list(self.response.min_samples.values()))

    @measure
    def get_labels(self, indexed_solutions):
        if hasattr(self, 'indexed_qubo'):
            if type(self.indexed_qubo) == np.ndarray:
                solutions = {
                    self.index2label[i]:q 
                    for i,q in enumerate(indexed_solutions)
                }
                self.labels = sol2lab(solution=solutions, n_clusters=self.n_clusters, n_points=self.n_points)
            elif type(list(self.qubo.keys())[0][0]) == tuple:
                solutions = {
                    self.index2label[i]:q 
                    for i,q in enumerate(indexed_solutions)
                }
                self.labels = sol2lab(solution=solutions, n_clusters=self.n_clusters, n_points=self.n_points)
        elif type(list(self.qubo.keys())[0][0]) == int: #この辺怪しい
            solutions = {
                self.index2label[i]:q 
                for i,q in enumerate(indexed_solutions)
            }
            self.labels = sol2lab(solution=solutions, n_clusters=self.n_clusters, n_points=self.n_points)
        else : #この辺怪しい
            solutions = {
                self.index2label[i]:q 
                for i,q in enumerate(indexed_solutions)
            }
            self.labels = sol2lab(solution=solutions, n_clusters=self.n_clusters, n_points=self.n_points)
        return self.labels
        
    def __set_timing(self, fn, time):
        self.timing[fn] = time if fn not in self.timing else self.timing[fn]+time

    def calc_cost(self, scaling='normal'):
        self.cost = calc_cost(self.data, self.labels, scaling=scaling)
        return self.cost
    
    def calc_inertia(self):
        self.inertia = calc_inertia(self.data, self.labels)
        return self.inertia