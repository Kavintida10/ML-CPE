import cv2
import numpy as np

# ตั้งขนาด 128x128 เพื่อให้เบาเครื่อง คอมสเปกเริ่มต้นรันได้สบาย
IMG_SIZE = (128, 128)

def preprocess_images(images):
    processed = []
    for img in images:
        # ปรับขนาดภาพ
        resized = cv2.resize(img, IMG_SIZE)
        # แปลงจาก OpenCV BGR เป็น RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        processed.append(rgb)
        
    # แปลงเป็น NumPy array และ Normalize ให้อยู่ในช่วง 0 - 1
    X = np.array(processed, dtype=np.float32) / 255.0
    return X