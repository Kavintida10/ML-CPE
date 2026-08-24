import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

def split_and_save_data(X, y, categories, output_dir="classification/outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # แบ่ง 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # แบ่ง Temp เป็น Val 15% และ Test 15%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # เซฟไฟล์ .npy
    np.save(os.path.join(output_dir, "features.npy"), X)
    np.save(os.path.join(output_dir, "labels.npy"), y)
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    with open(os.path.join(output_dir, "classes.json"), "w") as f:
        json.dump(categories, f)
        
    return X_train, X_val, X_test, y_train, y_val, y_test