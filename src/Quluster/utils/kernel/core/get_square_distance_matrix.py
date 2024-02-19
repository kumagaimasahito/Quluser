import numpy as np
from scipy.spatial import distance

def get_square_distance_matrix(data, mode="scipy"):

    if mode == "numpy":
        diffs = np.expand_dims(data, axis=1) - np.expand_dims(data, axis=0)
        dists = np.sum(diffs**2, axis=-1)
    elif mode == "scipy":
        dists = distance.cdist(data, data, 'sqeuclidean')
    
    return dists