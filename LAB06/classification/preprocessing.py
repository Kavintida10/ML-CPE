import numpy as np

def preprocess_features(X):
    # ปรับสเกลให้อยู่ในช่วง 0 ถึง 1
    return X.astype('float32') / 255.0
