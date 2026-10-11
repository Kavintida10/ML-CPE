import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import preprocess_images

def main():
    # กำหนด Path (อิงจากโฟลเดอร์โปรเจกต์)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'FlowersImage')
    output_dir = os.path.join(base_dir, 'outputs')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Loading raw images...")
    images, labels, class_names = load_data(data_dir)
    
    print("2. Preprocessing images...")
    X = preprocess_images(images)
    y = np.array(labels)
    
    # 3. บันทึก classes.json
    classes_path = os.path.join(output_dir, 'classes.json')
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    print(f"Saved: {classes_path}")
    
    # 4. แบ่งชุดข้อมูล Train (70%), Validation (15%), Test (15%)
    print("3. Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    
    # 5. บันทึกไฟล์ .npy ลงโฟลเดอร์ outputs/
    print("4. Saving arrays to outputs/...")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    np.save(os.path.join(output_dir, 'labels.npy'), y)
    
    print("\nData preparation completed!")
    print(f"Train size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

if __name__ == '__main__':
    main()