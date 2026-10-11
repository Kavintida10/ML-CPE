import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def plot_history(history, model_name, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in [(ax1, 'loss', 'Loss'), (ax2, 'accuracy', 'Accuracy')]:
        ax.plot(history.history[metric], label=f'Train {title}')
        ax.plot(history.history[f'val_{metric}'], label=f'Val {title}')
        ax.set_title(f'{model_name} - {title}'), ax.set_xlabel('Epoch'), ax.legend(), ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_history_{model_name}.png'))
    plt.close()

def plot_all_models_comparison(all_histories, output_dir):
    colors = ['#c2185b', '#e65100', '#1976d2', '#2e7d32']
    markers = ['o', 's', '^', 'd']
    
    total_epochs = len(next(iter(all_histories.values())).history['accuracy'])
    epochs = list(range(1, total_epochs + 1))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor='white')
    plt.subplots_adjust(hspace=0.35, wspace=0.2)
    
    sub_plots = [
        (axes[0, 0], 'accuracy', 'Training Performance', '(a) Training accuracy of models.', 'Accuracy (%)', (0, 105), 'lower right', True),
        (axes[0, 1], 'val_accuracy', 'Training Performance', '(b) Validation accuracy of models.', 'Validation Accuracy (%)', (0, 105), 'lower right', True),
        (axes[1, 0], 'loss', 'Loss Curve', '(c) Training loss of models.', 'Loss', None, 'upper right', False),
        (axes[1, 1], 'val_loss', 'Loss Curve', '(d) Validation loss of models.', 'Validation Loss', None, 'upper right', False)
    ]

    markevery = max(1, total_epochs // 15)

    for ax, key, super_title, sub_title, ylabel, ylim, leg_loc, is_acc in sub_plots:
        ax.set_facecolor('#eaeaf2')
        for i, (name, h) in enumerate(all_histories.items()):
            vals = [v * 100 for v in h.history[key]] if is_acc else h.history[key]
            lbl = f"{name}: {vals[-1]:.2f}%" if is_acc else f"{name}: {vals[-1]:.4f}"
            ax.plot(epochs, vals, label=lbl, color=colors[i % 4], marker=markers[i % 4], 
                    markevery=markevery, linewidth=1.8, markersize=4.5)
        
        ax.set_title(f"{super_title}\n{sub_title}", fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Epochs', fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_xlim(1, total_epochs)
        if ylim:
            ax.set_ylim(ylim)
            
        ax.grid(True, color='white', linewidth=1.2, linestyle='-')
        ax.legend(loc=leg_loc, frameon=True, facecolor='white', edgecolor='#cccccc', framealpha=0.95, fontsize=8.5)

    compare_all_path = os.path.join(output_dir, 'all_models_comparison_paper.png')
    plt.savefig(compare_all_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved IEEE-Style Comparison Graph to: {compare_all_path}")

def evaluate_model(model, X_test, y_test, class_names, model_name, output_dir):
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n--- Classification Report: {model_name} ---\n", classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.colorbar(), plt.xticks(range(len(class_names)), class_names, rotation=45), plt.yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.ylabel('True label'), plt.xlabel('Predicted label')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()