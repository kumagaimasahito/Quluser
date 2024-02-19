import collections

def solution2labels(solution, n_clusters, n_points):
    labels = [
        -1 if (
            collections.Counter(
                [
                    solution[i,j]
                    for j in range(n_clusters)
                ]
            )[1]!=1
        ) 
        else [
            solution[i,j]
            for j in range(n_clusters)
        ].index(1)
        for i in range(n_points)
    ]
    return labels