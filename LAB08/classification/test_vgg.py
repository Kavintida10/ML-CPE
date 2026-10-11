import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'outputs')
    
    # 1. โหลดข้อมูลทดสอบและชื่อคลาส
    X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    with open(os.path.join(output_dir, 'classes.json'), 'r', encoding='utf-8') as f:
        class_names = json.load(f)
        
    # 2. โหลดโมเดลตัวที่ดีที่สุด (MobileNet-V1)
    model_path = os.path.join(output_dir, 'mobilenet_v1_model.keras')
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return
        
    model = tf.keras.models.load_model(model_path)
    
    # 3. สุ่มรูปภาพทดสอบ 4 รูป
    np.random.seed(42)
    random_indices = np.random.choice(len(X_test), 4, replace=False)
    
    plt.figure(figsize=(10, 8))
    for i, idx in enumerate(random_indices):
        img = X_test[idx]
        true_label = class_names[y_test[idx]]
        
        # ทำนายผล
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_idx = np.argmax(pred)
        pred_label = class_names[pred_idx]
        confidence = np.max(pred) * 100
        
        # วาดภาพและแสดงป้ายชื่อ
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)", color=color, fontsize=12)
        plt.axis('off')
        
    save_path = os.path.join(output_dir, 'prediction_sample.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction sample to: {save_path}")

if __name__ == '__main__':
    main()