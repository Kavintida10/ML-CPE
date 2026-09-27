import numpy as np

def preprocess_features(X):
    # ปรับสเกลค่าพิกเซล 0-255 ให้อยู่ในช่วง 0.0 - 1.0
    return X.astype('float32') / 255.0