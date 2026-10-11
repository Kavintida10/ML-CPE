import os
import cv2

def load_data(data_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    # กรองเฉพาะโฟลเดอร์คลาส
    class_names = [cls for cls in class_names if os.path.isdir(os.path.join(data_dir, cls))]
    
    print(f"Found classes: {class_names}")
    
    for label_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, class_name)
        file_list = os.listdir(class_folder)
        print(f"Loading '{class_name}' ({len(file_list)} images)...")
        
        for file_name in file_list:
            file_path = os.path.join(class_folder, file_name)
            
            try:
                # อ่านรูปภาพ
                img = cv2.imread(file_path)
                
                # ข้ามไฟล์ที่เสียหรือไม่สามารถเปิดได้
                if img is None:
                    continue
                
                images.append(img)
                labels.append(label_idx)
            except Exception as e:
                print(f"Skipping corrupted file: {file_path}")
                continue

    return images, labels, class_names