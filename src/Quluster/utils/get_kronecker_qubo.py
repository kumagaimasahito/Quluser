import numpy as np

def get_kronecker_qubo_euclidean(dist, lagr, n_points, n_clusters):
    # Generate the Objective QUBO 
    iden = np.identity(n_clusters)
    obje = np.kron(dist, iden)

    # Generate the Constraint QUBO
    lamb = np.diag(np.full(n_points, lagr))
    uden = np.tri(n_clusters, k=-1).T
    coe2 = 2 * uden - iden
    cons = np.kron(lamb, coe2)

    qubo = obje + cons
    return qubo

# def get_kronecker_pubo_euclidean(dist, n_clusters):
#     iden = np.identity(n_clusters)
#     pubo = np.kron(dist, iden)
#     return pubo

def get_kronecker_qubo_kernel(gram, lagr, n_points, n_clusters):
    # Generate the Objective QUBO 
    iden = np.identity(n_clusters)
    diag = np.diag(np.diag(gram))
    triu = np.triu(gram, k=1)
    coe1 = - diag - 2 * triu
    obje = np.kron(coe1, iden)

    # Generate the Constraint QUBO
    lamb = np.diag(np.full(n_points, lagr))
    uden = np.tri(n_clusters, k=-1).T
    coe2 = 2 * uden - iden
    cons = np.kron(lamb, coe2)

    qubo = obje + cons
    return qubo

# def get_kronecker_pubo_kernel(gram, n_clusters):
#     iden = np.identity(n_clusters)
#     diag = np.diag(np.diag(gram))
#     triu = np.triu(gram, k=1)
#     coef = - diag - 2 * triu
#     pubo = np.kron(coef, iden)
#     return pubo