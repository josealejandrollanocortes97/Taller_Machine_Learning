import numpy as np

def k_medoids(X, n_clusters, max_iter=100):
    # Initialize medoids randomly
    medoids_idx = np.random.choice(X.shape[0], n_clusters, replace=False)
    medoids = X[medoids_idx]
    
    for i in range(max_iter):
        # Calculate distance matrix
        D = np.sum((X[:, np.newaxis, :] - medoids)**2, axis=2)
        
        # Assign each point to the closest medoid
        labels = np.argmin(D, axis=1)
        
        # Update medoids
        for j in range(n_clusters):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_medoid = cluster_points[np.argmin(np.sum((cluster_points[:, np.newaxis, :] - cluster_points)**2, axis=2))]
                medoids[j] = new_medoid
                
    return labels, medoids


    '''
    import numpy as np

class KMedoids:
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        
    def fit(self, X):
        self.X = X
        m, n = X.shape

        # Initialize the medoids randomly
        self.medoids = X[np.random.choice(m, self.k, replace=False)]
        
        # Initialize the cluster labels and distances
        self.labels = np.zeros(m, dtype=int)
        self.distances = np.zeros((m, self.k))
        
        # Iterate until convergence or maximum iterations
        for it in range(self.max_iter):
            # Update the distances to the medoids
            for j in range(self.k):
                self.distances[:,j] = np.sum(np.abs(X - self.medoids[j]), axis=1)
            
            # Assign each data point to the closest medoid
            self.labels = np.argmin(self.distances, axis=1)
            
            # Update the medoids
            for j in range(self.k):
                self.medoids[j] = X[self.labels == j].mean(axis=0)
        
        return self.labels, self.medoids

    '''