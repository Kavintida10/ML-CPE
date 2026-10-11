import os
import json
import numpy as np
import matplotlib.pyplot as plt
from vgg_model import build_model
from evaluate import plot_history, evaluate_model, plot_all_models_comparison

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'outputs')
    
    print("Loading prepared dataset...")
    X_train = np.load(os.path.join(output_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(output_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(output_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(output_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(output_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(output_dir, 'y_test.npy'))
    
    with open(os.path.join(output_dir, 'classes.json'), 'r', encoding='utf-8') as f:
        class_names = json.load(f)

    BATCH_SIZE = 32
    num_classes = len(class_names)

    # 4 โมเดลที่ต้องการเปรียบเทียบ
    models_to_train = ['mobilenet_v1', 'mobilenet_v2', 'vgg16', 'efficientnet_b0']
    EPOCHS = 20  # ตั้งค่าเป็น 20 Epochs เพื่อความโค้งสวยสมบูรณ์แบบ
    
    results_models = {}
    all_histories = {}

    print(f"\n>>> TRAINING 4 MODELS ({EPOCHS} EPOCHS) <<<")
    for model_name in models_to_train:
        print(f"\n==========================================")
        print(f"Training Model: {model_name.upper()} ({EPOCHS} Epochs)")
        print(f"==========================================")
        
        model = build_model(model_name=model_name, num_classes=num_classes, input_shape=(128, 128, 3))
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        all_histories[model_name] = history

        # บันทึกโมเดล
        model_save_path = os.path.join(output_dir, f'{model_name}_model.keras')
        model.save(model_save_path)
        
        # วาดกราฟเดี่ยวและ Matrix
        plot_history(history, model_name, output_dir)
        evaluate_model(model, X_test, y_test, class_names, model_name, output_dir)
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        results_models[model_name] = {'test_loss': test_loss, 'test_acc': test_acc}

    # บันทึกกราฟรวม 4 โมเดลสไตล์ IEEE Paper (20 Epochs)
    plot_all_models_comparison(all_histories, output_dir)

    # ==============================================================
    # ทำกราฟเปรียบเทียบ 10 vs 20 Epochs ของ MobileNet-V1
    # (ดึงผลจาก history ของ 20 Epochs มาเทียบได้เลย ไม่ต้องเสียเวลารันใหม่)
    # ==============================================================
    hist_mb = all_histories['mobilenet_v1']
    val_acc_10 = hist_mb.history['val_accuracy'][9] * 100
    val_acc_20 = hist_mb.history['val_accuracy'][-1] * 100
    val_loss_10 = hist_mb.history['val_loss'][9]
    val_loss_20 = hist_mb.history['val_loss'][-1]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), hist_mb.history['val_loss'][:10], label=f'10 Epochs (Loss: {val_loss_10:.4f})', linestyle='--')
    plt.plot(range(1, 21), hist_mb.history['val_loss'], label=f'20 Epochs (Loss: {val_loss_20:.4f})')
    plt.title('Epochs Comparison: Val Loss')
    plt.xlabel('Epoch'), plt.ylabel('Validation Loss'), plt.legend(), plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 11), [x*100 for x in hist_mb.history['val_accuracy'][:10]], label=f'10 Epochs (Acc: {val_acc_10:.2f}%)', linestyle='--')
    plt.plot(range(1, 21), [x*100 for x in hist_mb.history['val_accuracy']], label=f'20 Epochs (Acc: {val_acc_20:.2f}%)')
    plt.title('Epochs Comparison: Val Accuracy')
    plt.xlabel('Epoch'), plt.ylabel('Validation Accuracy (%)'), plt.legend(), plt.grid(True)

    epochs_plot_path = os.path.join(output_dir, 'epochs_comparison.png')
    plt.tight_layout()
    plt.savefig(epochs_plot_path)
    plt.close()
    print(f"Saved Epochs comparison plot to: {epochs_plot_path}")

    # ==============================================================
    # ตารางสรุปผลลัพธ์
    # ==============================================================
    print("\n\n================ 1. MODEL CONFIGURATION RESULTS (20 Epochs) ================")
    print(f"{'Model':<20} | {'Test Accuracy':<15} | {'Test Loss':<10}")
    print("-" * 52)
    for name, metrics in results_models.items():
        print(f"{name:<20} | {metrics['test_acc']*100:.2f}%{'':<9} | {metrics['test_loss']:.4f}")
    
    print("\n================ 2. EPOCHS COMPARISON (MobileNet-V1) ================")
    print(f"{'Epochs':<10} | {'Val Accuracy':<15} | {'Val Loss':<10}")
    print("-" * 42)
    print(f"{'10 Epochs':<10} | {val_acc_10:.2f}%{'':<9} | {val_loss_10:.4f}")
    print(f"{'20 Epochs':<10} | {val_acc_20:.2f}%{'':<9} | {val_loss_20:.4f}")
    print("=====================================================================")

if __name__ == '__main__':
    main()