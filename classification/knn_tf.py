"""
KNN Classifier Implementation using NumPy (compatible with all Python versions)
"""

import numpy as np


class TFKNNClassifier:

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float32)
        self.y_train = np.array(y, dtype=np.int32)
        self.n_classes = int(y.max()) + 1
        return self

    def _distance(self, X_new):
        diff = X_new[:, None, :] - self.X_train[None, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))

    def predict(self, X):
        X = np.array(X, dtype=np.float32)
        dist = self._distance(X)

        # หา index ของเพื่อนบ้านที่ใกล้ที่สุด k ตัว
        idx = np.argsort(dist, axis=1)[:, : self.k]
        neighbor_labels = self.y_train[idx]

        # โหวตเลือกคลาสที่คะแนนสูงสุด
        preds = []
        for row in neighbor_labels:
            counts = np.bincount(row, minlength=self.n_classes)
            preds.append(np.argmax(counts))

        return np.array(preds)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))