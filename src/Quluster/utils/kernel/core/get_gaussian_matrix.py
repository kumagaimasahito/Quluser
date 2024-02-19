import numpy as np

def get_gaussian_matrix(sq_dists, sigma=0.2, mode="numpy"):

    if mode == "numpy":
        gauss = np.exp(-sq_dists/(2*sigma**2))
    elif mode == "iterance":
        gauss = list(map(lambda d: np.exp(-d/(2*sigma**2)), sq_dists))
        
    return gauss
