"""
K-Means Clustering implementation using NumPy
"""

import numpy as np


class TFKMeans:

    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, seed=42):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.seed = seed
        self.centroids = None
        self.inertia_ = None

    def fit(self, X):
        np.random.seed(self.seed)
        X = np.array(X, dtype=np.float32)
        n_samples = X.shape[0]

        # สุ่มจุด Centroids เริ่มต้น
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iters):
            diff = X[:, None, :] - self.centroids[None, :, :]
            distances = np.sum(diff**2, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centroids = []
            for i in range(self.k):
                points = X[labels == i]
                if len(points) > 0:
                    new_centroids.append(points.mean(axis=0))
                else:
                    new_centroids.append(self.centroids[i])

            new_centroids = np.array(new_centroids)
            shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        diff = X[:, None, :] - self.centroids[None, :, :]
        distances = np.sum(diff**2, axis=2)
        self.inertia_ = float(np.sum(np.min(distances, axis=1)))
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        diff = X[:, None, :] - self.centroids[None, :, :]
        distances = np.sum(diff**2, axis=2)
        return np.argmin(distances, axis=1)