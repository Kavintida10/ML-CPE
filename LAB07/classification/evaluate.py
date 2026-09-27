import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def plot_history(history, output_path="outputs/training_history.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # สไตล์วิชาการแบบ IEEE
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    # 1. กราฟ Model Accuracy
    ax1 = axes[0]
    final_train_acc = history.history['accuracy'][-1] * 100
    final_val_acc = history.history['val_accuracy'][-1] * 100

    ax1.plot(epochs, [v * 100 for v in history.history['accuracy']], 
             color='#1f77b4', marker='o', markersize=4, linewidth=1.5, 
             label=f"Training ({final_train_acc:.2f}%)")
    ax1.plot(epochs, [v * 100 for v in history.history['val_accuracy']], 
             color='#2ca02c', marker='^', markersize=4, linewidth=1.5, 
             label=f"Validation ({final_val_acc:.2f}%)")
    
    ax1.set_title("Training Performance\n(a) Model Accuracy", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("Epochs", fontsize=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax1.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)

    # 2. กราฟ Model Loss
    ax2 = axes[1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    ax2.plot(epochs, history.history['loss'], 
             color='#d62728', marker='o', markersize=4, linewidth=1.5, 
             label=f"Training ({final_train_loss:.4f})")
    ax2.plot(epochs, history.history['val_loss'], 
             color='#9467bd', marker='s', markersize=4, linewidth=1.5, 
             label=f"Validation ({final_val_loss:.4f})")
    
    ax2.set_title("Loss Curve\n(b) Model Loss", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("Epochs", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax2.legend(loc='upper right', frameon=True, edgecolor='black', fancybox=False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Academic styled graph saved to: {output_path}")

def evaluate_and_plot(model, history, X_test, y_test, categories, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    # บันทึกกราฟ Accuracy & Loss
    plot_history(history, output_path=os.path.join(output_dir, "training_history.png"))
    
    # ทำนายผล
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
    
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=categories)
    print("\n--- Classification Report ---")
    print(report)
    
    # บันทึกรูป Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix saved to: {cm_path}")