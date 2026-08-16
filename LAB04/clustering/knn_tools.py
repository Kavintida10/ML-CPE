"""
KNN Tools: หาข้อมูลตัวอย่างที่เป็นจุดศูนย์กลางและเพื่อนบ้านใกล้เคียงที่สุดในแต่ละ Cluster
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_cluster_representatives(X_scaled, labels, n_clusters, n_samples=3):
    """หาดัชนีของตัวอย่างที่ใกล้จุดศูนย์กลางของแต่ละกลุ่มมากที่สุด"""
    representatives = {}
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = X_scaled[cluster_indices]

        if len(cluster_points) == 0:
            continue

        center = np.mean(cluster_points, axis=0, keepdims=True)
        
        # ใช้ NearestNeighbors หาระยะทางที่ใกล้จุดกึ่งกลางที่สุด
        nn = NearestNeighbors(n_neighbors=min(n_samples, len(cluster_points)))
        nn.fit(cluster_points)
        _, nearest_idx = nn.kneighbors(center)

        representatives[cluster_id] = cluster_indices[nearest_idx[0]]

    return representatives