import os
import random
import numpy as np
import matplotlib.pyplot as plt

def test_random_samples(model, X_test, y_test, categories, output_dir="outputs"):
    indices = random.sample(range(len(X_test)), 4)
    plt.figure(figsize=(10, 8))
    
    # คำนวณความน่าจะเป็น (Probabilities) สำหรับหาเปอร์เซ็นต์
    probs = model.predict_proba(X_test[indices])
    preds = model.predict(X_test[indices])
    
    correct_count = 0
    for i, idx in enumerate(indices):
        if preds[i] == y_test[idx]:
            correct_count += 1
            
    # หัวข้อใหญ่ด้านบน
    plt.suptitle(f"Prediction: {correct_count}/4 correct", fontsize=16)
    
    for i, idx in enumerate(indices):
        feature = X_test[idx]
        actual_name = categories[int(y_test[idx])]
        pred_label = preds[i]
        pred_name = categories[int(pred_label)]
        
        confidence = int(np.max(probs[i]) * 100)
        is_correct = (pred_label == y_test[idx])
        color = 'green' if is_correct else 'red'
        
        img = feature.reshape(64, 64, 3)
        
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        
        title_text = f"Pred: {pred_name} ({confidence}%)\nTrue: {actual_name}"
        plt.title(title_text, color=color, fontsize=13)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_sample.png"))
    plt.close()