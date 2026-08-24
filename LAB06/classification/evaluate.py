import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_plot(model, history, X_test, y_test, categories, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. พล็อต Loss Curve ของ Scikit-Learn
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(6, 4))
        plt.plot(model.loss_curve_, label='Training Loss')
        plt.title('Loss vs Iterations')
        plt.xlabel('Iterations (Epochs)')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_history.png"))
        plt.close()
        
        # บันทึก loss ลง history.json
        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump({"loss": model.loss_curve_}, f)

    # 2. Confusion Matrix & Report
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=categories))