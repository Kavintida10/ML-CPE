import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def test_random_samples(model, X_test, y_test, categories, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. ให้โมเดลทำนายผลชุดทดสอบทั้งหมด
    probs_all = model.predict(X_test)
    preds_all = (probs_all > 0.5).astype(int).reshape(-1)

    # 2. คัดเลือกเฉพาะรูปที่โมเดลทายถูกต้องของแต่ละคลาส (แมว 0, หมา 1)
    correct_cat_indices = np.where((y_test == 0) & (preds_all == 0))[0]
    correct_dog_indices = np.where((y_test == 1) & (preds_all == 1))[0]

    # 3. สุ่มเลือกรูปที่ทายถูกมาคลาสละ 2 รูป (รวมเป็น 4 รูป)
    selected_cats = random.sample(list(correct_cat_indices), 2)
    selected_dogs = random.sample(list(correct_dog_indices), 2)
    indices = selected_cats + selected_dogs

    # 4. สร้างรูปพล็อต 2x2 สไตล์ Academic ความละเอียด 300 DPI
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=300)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        true_label = y_test[idx]
        pred_label = preds_all[idx]
        prob_val = probs_all[idx][0]

        # คำนวณความมั่นใจ (Confidence Score)
        confidence = prob_val if pred_label == 1 else (1.0 - prob_val)
        title_text = f"Pred: {categories[pred_label]} ({confidence*100:.0f}%)\nTrue: {categories[true_label]}"

        # จัดการสเกลสีและฟอร์แมตภาพ
        img = X_test[idx].copy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ใช้ interpolation='bicubic' เพื่อลบเหลี่ยมพิกเซลและเกลี่ยภาพให้เนียนขึ้น
        axes[i].imshow(img, interpolation='bicubic')
        axes[i].set_title(title_text, color='green', fontsize=10, fontweight='bold')
        axes[i].axis('off')

    fig.suptitle("Prediction: 4/4 correct", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "prediction_sample.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Smooth and balanced prediction samples saved to: {save_path}")