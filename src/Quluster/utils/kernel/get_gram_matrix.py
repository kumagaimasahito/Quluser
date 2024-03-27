import numpy as np

def get_gram_matrix(kernel_matrix, n_points, mode="numpy"):
    if mode == "numpy":
        mean_row = np.mean(kernel_matrix, axis=1).reshape(n_points,1)
        mean_col = np.mean(kernel_matrix, axis=0).reshape(1,n_points)
        mean_all = np.mean(kernel_matrix)
        gram_matrix = kernel_matrix - mean_row - mean_col + mean_all

    elif mode == "iterance":
        gram_matrix = np.array([
            [
                kernel_matrix[i,j] - sum(
                    [
                        kernel_matrix[k,j]
                        for k in range(n_points)
                    ]
                ) / n_points - sum(
                    [
                        kernel_matrix[i,l]
                        for l in range(n_points)
                    ]
                ) / n_points + sum(
                    [
                        kernel_matrix[k,l]
                        for k in range(n_points)
                        for l in range(n_points)
                    ]
                ) / n_points**2
                for i in range(n_points)
            ]
            for j in range(n_points)
        ])

    return gram_matrix
