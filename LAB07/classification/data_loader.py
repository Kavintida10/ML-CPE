import os
import cv2
import numpy as np

def load_data(data_dir="../PetImages", img_size=(64, 64)):
    categories = ["Cat", "Dog"]
    data = []
    labels = []
    
    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            print(f"Warning: Directory not found: {path}")
            continue
            
        print(f"Loading images from {category}...")
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                # อ่านรูปภาพและข้ามไฟล์ที่เสียหาย
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # แปลงสี BGR เป็น RGB และ Resize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                data.append(img)
                labels.append(label)
            except Exception:
                continue

    return np.array(data), np.array(labels), categories
